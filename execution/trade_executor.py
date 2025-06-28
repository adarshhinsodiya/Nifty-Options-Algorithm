"""
Trade execution module for handling order placement and position management.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, time
import pytz

from core.models import (
    Order, OrderStatus, OrderType, Position, PositionStatus, 
    TradeSignal, TradeSignalType, Portfolio
)
from core.config_manager import ConfigManager
from data.data_provider import DataProvider

class TradeExecutor:
    """Handles trade execution and position management."""
    
    def __init__(self, config: ConfigManager, data_provider: DataProvider, portfolio: Portfolio):
        """Initialize trade executor.
        
        Args:
            config: Configuration manager instance
            data_provider: Data provider instance
            portfolio: Portfolio instance to manage positions
        """
        self.config = config
        self.data_provider = data_provider
        self.portfolio = portfolio
        self.trading_config = config.get_trading_config()
        self.options_config = config.get_options_config()
        self.backtest_config = config.get_backtest_config()
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.logger = self._setup_logger()
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.last_trade_time = None
        
        # Trade and position tracking
        self.current_date = datetime.now(self.ist_tz).date()
        self.daily_trades_count = 0
        
        # Signal cooldown tracking
        self.last_signal_time = None
        self.min_candle_gap = self.trading_config.get('min_candle_gap_between_signals', 5)  # Default to 5 candles if not set
        self.logger.info(f"Signal cooldown set to {self.min_candle_gap} candles")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the trade executor."""
        logger = logging.getLogger(__name__)
        return logger
        
    def _calculate_fill_price(
        self,
        symbol: str,
        order_type: OrderType,
        price: float,
        quantity: int,
        is_options: bool = False
    ) -> float:
        """Calculate fill price with slippage and spread for backtesting.
        
        Args:
            symbol: Trading symbol
            order_type: Type of order (MARKET, LIMIT, etc.)
            price: Requested price (0 for market orders)
            quantity: Order quantity (positive for buy, negative for sell)
            is_options: Whether this is an options trade
            
        Returns:
            float: Fill price with slippage and spread applied
        """
        try:
            # Get current market data
            current_price = self.data_provider.get_latest_price(symbol, 'NFO' if is_options else 'NSE')
            
            # For market orders, use the current market price
            if order_type == OrderType.MARKET:
                fill_price = current_price
            else:
                # For limit/sl orders, use the requested price if it's better than the current price
                if (quantity > 0 and price >= current_price) or (quantity < 0 and price <= current_price):
                    fill_price = price
                else:
                    fill_price = current_price
            
            # Apply spread (bid-ask spread simulation)
            # For options, use a wider spread (0.5% each side)
            # For stocks, use a tighter spread (0.1% each side)
            spread_pct = 0.005 if is_options else 0.001
            
            if quantity > 0:  # Buy order - pay the ask price (higher)
                fill_price *= (1 + spread_pct)
            else:  # Sell order - get the bid price (lower)
                fill_price *= (1 - spread_pct)
            
            # Apply slippage (random between 0 and slippage config, in basis points)
            slippage_bps = self.backtest_config.slippage  # Slippage in basis points (e.g., 5 = 0.05%)
            if slippage_bps > 0:
                slippage_pct = (random.uniform(0, slippage_bps) / 10000)  # Convert bps to decimal
                fill_price *= (1 + slippage_pct) if quantity > 0 else (1 - slippage_pct)
            
            # Ensure price is valid and non-negative
            fill_price = max(0.05, fill_price)  # Minimum price of 0.05 to avoid negative or zero prices
            
            # Round to appropriate decimal places
            fill_price = round(fill_price, 2 if is_options else 2)  # 2 decimal places for both stocks and options
            
            return fill_price
            
        except Exception as e:
            self.logger.error(f"Error calculating fill price: {str(e)}")
            return price  # Fallback to requested price if there's an error
    
    def _generate_order_id(self) -> str:
        """Generate a unique order ID."""
        return f"ORD_{datetime.now(self.ist_tz).strftime('%Y%m%d_%H%M%S_%f')}"
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours (9:15 AM to 3:30 PM IST)."""
        now = datetime.now(self.ist_tz)
        market_open = time(9, 15)
        market_close = time(15, 30)
        
        # Check if it's a weekday (0 = Monday, 6 = Sunday)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
            
        # Check if within market hours
        current_time = now.time()
        return market_open <= current_time <= market_close
    
    def _get_option_instrument(
        self, 
        symbol: str, 
        strike: int, 
        option_type: str, 
        expiry_date: datetime
    ) -> Dict[str, Any]:
        """Get option instrument details."""
        return self.data_provider.get_option_instrument(
            symbol=symbol,
            strike=strike,
            option_type=option_type,
            expiry_date=expiry_date
        )
    
    def place_order(
        self,
        symbol: str,
        order_type: OrderType,
        quantity: int,
        price: float = 0.0,
        product: str = 'options',
        exchange_code: str = 'NFO',
        validity: str = 'day',
        stop_loss: float = 0.0,
        trailing_stop_loss: float = 0.0,
        target_price: float = 0.0,
        tag: str = '',
        signal: Optional[TradeSignal] = None
    ) -> Optional[Order]:
        """Place an order.
        
        Args:
            symbol: Trading symbol
            order_type: Type of order (MARKET, LIMIT, SL, SLM)
            quantity: Order quantity
            price: Limit price (for LIMIT/SL orders)
            product: Product type (cash, options, futures)
            exchange_code: Exchange code (NSE, NFO)
            validity: Order validity (day, ioc, etc.)
            stop_loss: Stop loss price
            trailing_stop_loss: Trailing stop loss amount
            target_price: Target price for bracket orders
            tag: Optional order tag for tracking
            signal: Optional TradeSignal object for options orders
            
        Returns:
            Order object if successful, None otherwise
        """
        try:
            # Generate order ID
            order_id = self._generate_order_id()
            
            # Get current price for market orders
            if order_type == OrderType.MARKET:
                price = self.data_provider.get_latest_price(symbol, exchange_code)
                if price <= 0:
                    self.logger.error(f"Failed to get market price for {symbol}")
                    return None
            
            # Create order object
            order = Order(
                order_id=order_id,
                symbol=symbol,
                order_type=order_type,
                quantity=quantity,
                price=price,
                status=OrderStatus.PENDING
            )
            
            # In live mode, place the actual order
            if self.data_provider.mode == 'live':
                if not self._is_market_hours():
                    self.logger.warning("Order not placed: Market is closed")
                    return None
                
                # Prepare order parameters
                order_params = {
                    'stock_code': symbol,
                    'exchange_code': exchange_code,
                    'product': product,
                    'action': 'buy' if quantity > 0 else 'sell',
                    'order_type': order_type.value.lower(),
                    'stoploss': stop_loss,
                    'quantity': abs(quantity),
                    'price': price,
                    'validity': validity,
                    'disclosed_quantity': 0,
                    'trader_id': '',
                    'tag': tag
                }
                
                # Add option-specific parameters if this is an options order
                if product.lower() == 'options':
                    if not signal:
                        self.logger.error("Signal object is required for options orders")
                        return None
                        
                    if not signal.expiry_date:
                        self.logger.error("expiry_date is required for options orders")
                        return None
                        
                    # Format expiry date as 'DD-MM-YYYY' for Breeze API
                    expiry_date = signal.expiry_date.strftime('%d-%m-%Y')
                    
                    order_params.update({
                        'expiry_date': expiry_date,
                        'right': signal.option_type.upper(),  # CE or PE
                        'strike_price': signal.strike
                    })
                
                # Place order via Breeze API with retry/backoff
                try:
                    # Use the data provider's retry mechanism for the API call
                    response = self.data_provider._api_call_with_retry(
                        self.data_provider.breeze.place_order,
                        **order_params
                    )
                    
                    if not response or 'Success' not in response:
                        self.logger.error(f"Order placement failed: {response}")
                        order.status = OrderStatus.REJECTED
                        return None
                        
                except Exception as e:
                    self.logger.error(f"Failed to place order after retries: {str(e)}")
                    order.status = OrderStatus.REJECTED
                    return None
                
                # Update order with exchange details
                order.exchange_order_id = response['Success']['order_id']
                order.status = OrderStatus.OPEN
                
                # For bracket orders, place SL and target orders
                if stop_loss > 0 or target_price > 0:
                    self._place_bracket_orders(order, stop_loss, target_price, signal)
            
            # In backtest mode, simulate order execution with slippage and spread
            else:
                # Calculate fill price with slippage and spread
                fill_price = self._calculate_fill_price(
                    symbol=symbol,
                    order_type=order_type,
                    price=price,
                    quantity=quantity,
                    is_options=(product.lower() == 'options')
                )
                
                # Simulate immediate fill at the calculated price
                order.status = OrderStatus.COMPLETE
                order.filled_quantity = quantity
                order.average_price = fill_price
                order.filled_timestamp = datetime.now(self.ist_tz)
                
                self.logger.debug(
                    f"Backtest fill - Requested: {price:.2f}, "
                    f"Filled: {fill_price:.2f}, "
                    f"Slippage: {abs(fill_price - price):.2f}"
                )
            
            # Store the order
            self.orders[order_id] = order
            self.last_trade_time = datetime.now(self.ist_tz)
            
            self.logger.info(
                f"Order placed: {order_id} | {symbol} | {order_type.value} | "
                f"Qty: {quantity} @ {price} | Status: {order.status.value}"
            )
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return None
    
    def _place_bracket_orders(
        self,
        main_order: Order,
        stop_loss: float,
        target_price: float,
        signal: Optional[TradeSignal] = None
    ) -> None:
        """Place bracket orders (SL and target) for the main order.
        
        Args:
            main_order: The main order to place brackets for
            stop_loss: Stop loss price
            target_price: Target price
            signal: Optional TradeSignal object for options orders
        """
        try:
            # Common order parameters
            order_params = {
                'exchange_code': 'NFO',
                'product': 'options',
                'quantity': abs(main_order.quantity),
                'validity': 'day',
                'disclosed_quantity': 0,
                'trader_id': ''
            }
            
            # Add option-specific parameters if signal is provided
            if signal and signal.expiry_date:
                order_params.update({
                    'expiry_date': signal.expiry_date.strftime('%d-%m-%Y'),
                    'right': signal.option_type.upper(),
                    'strike_price': signal.strike
                })
            
            if stop_loss > 0:
                # Place stop loss order
                sl_order_id = f"{main_order.order_id}_SL"
                sl_order = Order(
                    order_id=sl_order_id,
                    symbol=main_order.symbol,
                    order_type=OrderType.SL,
                    quantity=main_order.quantity,
                    price=stop_loss,
                    status=OrderStatus.PENDING,
                    parent_order_id=main_order.order_id
                )
                
                # In live mode, place actual SL order
                if self.data_provider.mode == 'live':
                    sl_params = order_params.copy()
                    sl_params.update({
                        'stock_code': main_order.symbol,
                        'action': 'sell' if main_order.quantity > 0 else 'buy',
                        'order_type': 'stoploss',
                        'stoploss': stop_loss,
                        'price': 0,  # Market order for SL
                        'tag': f"SL_{main_order.order_id}"
                    })
                    
                    sl_response = self.data_provider.breeze.place_order(**sl_params)
                    
                    if sl_response and 'Success' in sl_response:
                        sl_order.exchange_order_id = sl_response['Success']['order_id']
                        sl_order.status = OrderStatus.OPEN
                    else:
                        self.logger.error(f"Failed to place SL order: {sl_response}")
                
                self.orders[sl_order_id] = sl_order
                
            if target_price > 0:
                # Place target order
                tgt_order_id = f"{main_order.order_id}_TGT"
                tgt_order = Order(
                    order_id=tgt_order_id,
                    symbol=main_order.symbol,
                    order_type=OrderType.LIMIT,
                    quantity=main_order.quantity,
                    price=target_price,
                    status=OrderStatus.PENDING,
                    parent_order_id=main_order.order_id
                )
                
                # In live mode, place actual target order
                if self.data_provider.mode == 'live':
                    tgt_params = order_params.copy()
                    tgt_params.update({
                        'stock_code': main_order.symbol,
                        'action': 'sell' if main_order.quantity > 0 else 'buy',
                        'order_type': 'limit',
                        'stoploss': 0,
                        'price': target_price,
                        'tag': f"TGT_{main_order.order_id}"
                    })
                    
                    tgt_response = self.data_provider.breeze.place_order(**tgt_params)
                    
                    if tgt_response and 'Success' in tgt_response:
                        tgt_order.exchange_order_id = tgt_response['Success']['order_id']
                        tgt_order.status = OrderStatus.OPEN
                    else:
                        self.logger.error(f"Failed to place target order: {tgt_response}")
                
                self.orders[tgt_order_id] = tgt_order
                
        except Exception as e:
            self.logger.error(f"Error placing bracket orders: {str(e)}")
    
    def _is_in_cooldown(self) -> bool:
        """Check if we're in a cooldown period after the last signal.
        
        Returns:
            bool: True if in cooldown, False otherwise
        """
        if self.last_signal_time is None:
            return False
            
        # Calculate time difference in minutes
        time_diff = (datetime.now(self.ist_tz) - self.last_signal_time).total_seconds() / 60
        
        # Convert candle gap to minutes (assuming 1m candles)
        cooldown_minutes = self.min_candle_gap * 1  # 1 minute per candle
        
        if time_diff < cooldown_minutes:
            remaining = cooldown_minutes - time_diff
            self.logger.debug(f"In cooldown period: {remaining:.1f} minutes remaining")
            return True
            
        return False

    def _check_daily_trades_limit(self) -> bool:
        """Check if we've reached the maximum number of daily trades.
        
        Returns:
            bool: True if we can place more trades today, False otherwise
        """
        current_date = datetime.now(self.ist_tz).date()
        
        # Reset daily counter if it's a new day
        if current_date != self.current_date:
            self.current_date = current_date
            self.daily_trades_count = 0
            self.logger.debug(f"New trading day: {self.current_date}")
        
        # Check against max daily trades limit
        max_daily = self.trading_config.max_daily_trades
        if max_daily > 0 and self.daily_trades_count >= max_daily:
            self.logger.warning(
                f"Daily trade limit reached: {self.daily_trades_count}/{max_daily} "
                "trades used today"
            )
            return False
            
        return True
    
    def _check_max_positions(self) -> bool:
        """Check if we can open a new position based on max_open_positions.
        
        Returns:
            bool: True if we can open a new position, False otherwise
        """
        max_positions = self.trading_config.max_open_positions
        if max_positions <= 0:  # No limit if zero or negative
            return True
            
        current_positions = len(self.portfolio.get_open_positions())
        if current_positions >= max_positions:
            self.logger.warning(
                f"Maximum open positions limit reached: {current_positions}/{max_positions}"
            )
            return False
            
        return True
        
    def _place_bracket_orders(
        self,
        main_order: Order,
        stop_loss: float,
        target_price: float,
        signal: Optional[TradeSignal] = None
    ) -> None:
        """Place bracket orders (SL and target) for the main order.
        
        Args:
            main_order: The main order to place brackets for
            stop_loss: Stop loss price
            target_price: Target price
            signal: Optional TradeSignal object for options orders
        """
        try:
            # Common order parameters
            order_params = {
                'exchange_code': 'NFO',
                'product': 'options',
                'quantity': abs(main_order.quantity),
                'validity': 'day',
                'disclosed_quantity': 0,
                'trader_id': ''
            }
            
            # Add option-specific parameters if signal is provided
            if signal and signal.expiry_date:
                order_params.update({
                    'expiry_date': signal.expiry_date.strftime('%d-%m-%Y'),
                    'right': signal.option_type.upper(),
                    'strike_price': signal.strike
                })
            
            if stop_loss > 0:
                # Place stop loss order
                sl_order_id = f"{main_order.order_id}_SL"
                sl_order = Order(
                    order_id=sl_order_id,
                    symbol=main_order.symbol,
                    order_type=OrderType.SL,
                    quantity=main_order.quantity,
                    price=stop_loss,
                    status=OrderStatus.PENDING,
                    parent_order_id=main_order.order_id
                )
                
                # In live mode, place actual SL order
                if self.data_provider.mode == 'live':
                    sl_params = order_params.copy()
                    sl_params.update({
                        'stock_code': main_order.symbol,
                        'action': 'sell' if main_order.quantity > 0 else 'buy',
                        'order_type': 'stoploss',
                        'stoploss': stop_loss,
                        'price': 0,  # Market order for SL
                        'tag': f"SL_{main_order.order_id}"
                    })
                    
                    sl_response = self.data_provider.breeze.place_order(**sl_params)
                    
                    if sl_response and 'Success' in sl_response:
                        sl_order.exchange_order_id = sl_response['Success']['order_id']
                        sl_order.status = OrderStatus.OPEN
                    else:
                        self.logger.error(f"Failed to place SL order: {sl_response}")
                
                self.orders[sl_order_id] = sl_order
                
            if target_price > 0:
                # Place target order
                tgt_order_id = f"{main_order.order_id}_TGT"
                tgt_order = Order(
                    order_id=tgt_order_id,
                    symbol=main_order.symbol,
                    order_type=OrderType.LIMIT,
                    quantity=main_order.quantity,
                    price=target_price,
                    status=OrderStatus.PENDING,
                    parent_order_id=main_order.order_id
                )
                
                # In live mode, place actual target order
                if self.data_provider.mode == 'live':
                    tgt_params = order_params.copy()
                    tgt_params.update({
                        'stock_code': main_order.symbol,
                        'action': 'sell' if main_order.quantity > 0 else 'buy',
                        'order_type': 'limit',
                        'stoploss': 0,
                        'price': target_price,
                        'tag': f"TGT_{main_order.order_id}"
                    })
                    
                    tgt_response = self.data_provider.breeze.place_order(**tgt_params)
                    
                    if tgt_response and 'Success' in tgt_response:
                        tgt_order.exchange_order_id = tgt_response['Success']['order_id']
                        tgt_order.status = OrderStatus.OPEN
                    else:
                        self.logger.error(f"Failed to place target order: {tgt_response}")
                
                self.orders[tgt_order_id] = tgt_order
                
        except Exception as e:
            self.logger.error(f"Error placing bracket orders: {str(e)}")
    
    def _is_in_cooldown(self) -> bool:
        """Check if we're in a cooldown period after the last signal.
        
        Returns:
            bool: True if in cooldown, False otherwise
        """
        if self.last_signal_time is None:
            return False
                
        # Calculate time difference in minutes
        time_diff = (datetime.now(self.ist_tz) - self.last_signal_time).total_seconds() / 60
            
        # Convert candle gap to minutes (assuming 1m candles)
        cooldown_minutes = self.min_candle_gap * 1  # 1 minute per candle
            
        if time_diff < cooldown_minutes:
            remaining = cooldown_minutes - time_diff
            self.logger.debug(f"In cooldown period: {remaining:.1f} minutes remaining")
            return True
                
        return False

    def _check_daily_trades_limit(self) -> bool:
        """Check if we've reached the maximum number of daily trades.
        
        Returns:
            bool: True if we can place more trades today, False otherwise
        """
        current_date = datetime.now(self.ist_tz).date()
            
        # Reset daily counter if it's a new day
        if current_date != self.current_date:
            self.current_date = current_date
            self.daily_trades_count = 0
            self.logger.debug(f"New trading day: {self.current_date}")
            
        # Check against max daily trades limit
        max_daily = self.trading_config.max_daily_trades
        if max_daily > 0 and self.daily_trades_count >= max_daily:
            self.logger.warning(
                f"Daily trade limit reached: {self.daily_trades_count}/{max_daily} "
                "trades used today"
            )
            return False
                
        return True
        
    def _check_max_positions(self) -> bool:
        """Check if we can open a new position based on max_open_positions.
        
        Returns:
            bool: True if we can open a new position, False otherwise
        """
        max_positions = self.trading_config.max_open_positions
        if max_positions <= 0:  # No limit if zero or negative
            return True
                
        current_positions = len(self.portfolio.get_open_positions())
        if current_positions >= max_positions:
            self.logger.warning(
                f"Maximum open positions limit reached: {current_positions}/{max_positions}"
            )
            return False
                
        return True
        
    def _is_trading_day(self, date: datetime) -> bool:
        """Check if the given date is a trading day (weekday and not a holiday).
        
        Args:
            date: Date to check
                
        Returns:
            bool: True if it's a trading day, False otherwise
        """
        # Check if it's a weekend
        if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False
                
        # Check if it's a holiday (you'll need to implement holiday checking)
        # For now, we'll assume no holidays - you should replace this with actual holiday checking
        # Example: if date.date() in self._get_market_holidays():
        #     return False
                
        return True
            
    def _get_commission_rate(self) -> float:
        """Get the commission rate from backtest config.
        
        Returns:
            float: Commission rate as a decimal (e.g., 0.0005 for 0.05%)
        """
        return self.backtest_config.commission if hasattr(self.backtest_config, 'commission') else 0.0

    def execute_signal(self, signal: TradeSignal) -> Optional[Position]:
        """Execute a trading signal.
        
        Args:
            signal: Trading signal to execute
                
        Returns:
            Position object if successful, None otherwise
        """
        try:
            # Skip if it's not a trading day (for backtest)
            if self.data_provider.mode == 'backtest' and not self._is_trading_day(signal.timestamp):
                self.logger.debug(f"Skipping signal on non-trading day: {signal.timestamp.date()}")
                return None
                    
            # Check position limits
            if not self._check_max_positions():
                return None
                    
            # Check daily trade limits
            if not self._check_daily_trades_limit():
                return None
                    
            # Check cooldown period
            if self._is_in_cooldown():
                self.logger.debug("Skipping signal: In cooldown period")
                return None
                    
            # Check if market is open (for live trading)
            if self.data_provider.mode == 'live' and not self._is_market_hours():
                self.logger.debug("Skipping signal: Market is closed")
                return None
                    
            # Get option instrument details
            expiry_date = signal.expiry_date or self._get_next_expiry()
            option_instrument = self._get_option_instrument(
                symbol=signal.symbol,
                strike=signal.strike,
                option_type=signal.option_type,
                expiry_date=expiry_date
            )
            
            if not option_instrument:
                self.logger.error(f"Could not find option instrument for {signal}")
                return None
            
            # Calculate position size based on risk
            risk_amount = self.portfolio.current_cash * (self.trading_config.risk_per_trade / 100)
            position_size = int(risk_amount / (signal.entry_price * self.options_config.lot_size))
            
            if position_size < 1:
                self.logger.warning(f"Position size too small: {position_size} (risk: {risk_amount:.2f})")
                return None
                    
            # Calculate quantity (positive for long, negative for short)
            quantity = position_size * self.options_config.lot_size
            if signal.signal_type == TradeSignalType.SHORT:
                quantity = -quantity
            
            # Place entry order
            order = self.place_order(
                symbol=option_instrument['tradingSymbol'],
                order_type=OrderType.MARKET,
                quantity=quantity,
                product='options',
                exchange_code='NFO',
                tag=f"ENTRY_{signal.signal_type.value}",
                signal=signal
            )
            
            if not order or order.status != OrderStatus.COMPLETE:
                self.logger.error(f"Failed to execute entry order for signal: {signal}")
                return None
                    
            # Create position
            position = Position(
                position_id=order.order_id,
                signal=signal,
                entry_order=order
            )
            
            # Add to portfolio with commission
            commission_rate = self._get_commission_rate()
            self.portfolio.add_position(position, commission=commission_rate)
            
            # Log commission cost
            if hasattr(order, 'commission'):
                self.logger.debug(f"Paid {order.commission:.2f} in commission for entry order {order.order_id}")
            
            # Place bracket orders (SL and target)
            if signal.stop_loss > 0 or signal.take_profit > 0:
                self._place_bracket_orders(
                    main_order=order,
                    stop_loss=signal.stop_loss,
                    target_price=signal.take_profit,
                    signal=signal
                )
            
            # Update trade tracking
            self.positions[position.position_id] = position
            self.last_trade_time = datetime.now(self.ist_tz)
            self.daily_trades_count += 1
            self.last_signal_time = datetime.now(self.ist_tz)
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {str(e)}", exc_info=True)
            return None
        
    def cancel_all_pending_orders(self) -> None:
        """Cancel all pending orders."""
        self.logger.info("Canceling all pending orders...")
        
        if self.data_provider.mode == 'live' and self.data_provider.breeze:
            try:
                # Get all open orders
                orders = self.data_provider.breeze.get_order_list()
                
                if orders and 'Success' in orders:
                    for order in orders['Success']:
                        if order['status'] in ['open', 'pending']:
                            self.data_provider.breeze.cancel_order(
                                order_id=order['order_id'],
                                stock_code=order['stock_code'],
                                exchange_code=order['exchange_code']
                            )
                            self.logger.info(f"Canceled order: {order['order_id']}")
                            
            except Exception as e:
                self.logger.error(f"Error canceling orders: {str(e)}")
