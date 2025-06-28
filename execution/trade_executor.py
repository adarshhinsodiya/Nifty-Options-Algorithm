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
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.logger = self._setup_logger()
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.last_trade_time = None
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the trade executor."""
        logger = logging.getLogger(__name__)
        return logger
    
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
        tag: str = ''
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
                
                # Place order via Breeze API
                response = self.data_provider.breeze.place_order(
                    stock_code=symbol,
                    exchange_code=exchange_code,
                    product=product,
                    action='buy' if quantity > 0 else 'sell',
                    order_type=order_type.value.lower(),
                    stoploss=stop_loss,
                    quantity=abs(quantity),
                    price=price,
                    validity=validity,
                    disclosed_quantity=0,
                    expiry_date='',
                    right='',
                    strike_price=0,
                    trader_id='',
                    tag=tag
                )
                
                if not response or 'Success' not in response:
                    self.logger.error(f"Order placement failed: {response}")
                    order.status = OrderStatus.REJECTED
                    return None
                
                # Update order with exchange details
                order.exchange_order_id = response['Success']['order_id']
                order.status = OrderStatus.OPEN
                
                # For bracket orders, place SL and target orders
                if stop_loss > 0 or target_price > 0:
                    self._place_bracket_orders(order, stop_loss, target_price)
            
            # In backtest mode, simulate order execution
            else:
                # Simulate immediate fill at the requested price
                order.status = OrderStatus.COMPLETE
                order.filled_quantity = quantity
                order.average_price = price
                order.filled_timestamp = datetime.now(self.ist_tz)
            
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
        target_price: float
    ) -> None:
        """Place bracket orders (SL and target) for the main order."""
        try:
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
                    sl_response = self.data_provider.breeze.place_order(
                        stock_code=main_order.symbol,
                        exchange_code='NFO',
                        product='options',
                        action='sell' if main_order.quantity > 0 else 'buy',
                        order_type='stoploss',
                        stoploss=stop_loss,
                        quantity=abs(main_order.quantity),
                        price=0,
                        validity='day',
                        disclosed_quantity=0,
                        expiry_date='',
                        right='',
                        strike_price=0,
                        trader_id='',
                        tag=f"SL_{main_order.order_id}"
                    )
                    
                    if sl_response and 'Success' in sl_response:
                        sl_order.exchange_order_id = sl_response['Success']['order_id']
                        sl_order.status = OrderStatus.OPEN
                
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
                    tgt_response = self.data_provider.breeze.place_order(
                        stock_code=main_order.symbol,
                        exchange_code='NFO',
                        product='options',
                        action='sell' if main_order.quantity > 0 else 'buy',
                        order_type='limit',
                        stoploss=0,
                        quantity=abs(main_order.quantity),
                        price=target_price,
                        validity='day',
                        disclosed_quantity=0,
                        expiry_date='',
                        right='',
                        strike_price=0,
                        trader_id='',
                        tag=f"TGT_{main_order.order_id}"
                    )
                    
                    if tgt_response and 'Success' in tgt_response:
                        tgt_order.exchange_order_id = tgt_response['Success']['order_id']
                        tgt_order.status = OrderStatus.OPEN
                
                self.orders[tgt_order_id] = tgt_order
                
        except Exception as e:
            self.logger.error(f"Error placing bracket orders: {str(e)}")
    
    def execute_signal(self, signal: TradeSignal) -> Optional[Position]:
        """Execute a trading signal.
        
        Args:
            signal: Trading signal to execute
            
        Returns:
            Position object if successful, None otherwise
        """
        try:
            # Check if we already have an open position for this signal
            for position in self.portfolio.get_open_positions():
                if (
                    position.signal.signal_type == signal.signal_type and
                    position.signal.option_type == signal.option_type and
                    position.signal.strike == signal.strike
                ):
                    self.logger.info(f"Position already open for signal: {signal}")
                    return None
            
            # Get option instrument details
            option = self._get_option_instrument(
                symbol=signal.symbol,
                strike=signal.strike,
                option_type=signal.option_type,
                expiry_date=signal.expiry_date
            )
            
            if not option:
                self.logger.error(f"Failed to get option instrument for {signal}")
                return None
            
            # Calculate position size based on risk
            risk_amount = self.portfolio.current_cash * (self.trading_config.risk_per_trade / 100)
            risk_per_contract = abs(signal.entry_price - signal.stop_loss) * option['lot_size']
            
            if risk_per_contract <= 0:
                self.logger.error(f"Invalid risk per contract: {risk_per_contract}")
                return None
            
            # Calculate number of lots to trade
            num_lots = int(risk_amount / risk_per_contract)
            num_lots = max(1, min(num_lots, 10))  # Limit to 10 lots max
            
            # Place entry order
            quantity = num_lots * option['lot_size']
            order = self.place_order(
                symbol=option['stock_code'],
                order_type=OrderType.MARKET,
                quantity=quantity,
                product='options',
                exchange_code='NFO',
                tag=f"ENTRY_{signal.signal_type.value}_{datetime.now(self.ist_tz).strftime('%H%M%S')}"
            )
            
            if not order or order.status != OrderStatus.COMPLETE:
                self.logger.error(f"Failed to place entry order for {signal}")
                return None
            
            # Create position
            position = Position(
                position_id=f"POS_{order.order_id}",
                signal=signal,
                entry_order=order,
                status=PositionStatus.OPEN
            )
            
            # Add position to portfolio
            self.portfolio.add_position(position)
            self.positions[position.position_id] = position
            
            self.logger.info(
                f"Position opened: {position.position_id} | {signal.signal_type.value} | "
                f"{signal.option_type.upper()} {signal.strike} | "
                f"Entry: {order.average_price} | SL: {signal.stop_loss} | TP: {signal.take_profit}"
            )
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error executing signal {signal}: {str(e)}")
            return None
    
    def manage_positions(self) -> None:
        """Monitor and manage open positions (check SL/TP)."""
        try:
            current_time = datetime.now(self.ist_tz)
            
            for position in self.portfolio.get_open_positions():
                # Skip if position is already closed
                if position.status != PositionStatus.OPEN:
                    continue
                
                # Get current price
                current_price = self.data_provider.get_latest_price(
                    position.entry_order.symbol, 'NFO'
                )
                
                if current_price <= 0:
                    self.logger.warning(f"Invalid price for {position.entry_order.symbol}")
                    continue
                
                signal = position.signal
                exit_reason = None
                exit_price = 0.0
                
                # Check stop loss
                if (
                    (signal.signal_type == TradeSignalType.LONG and current_price <= signal.stop_loss) or
                    (signal.signal_type == TradeSignalType.SHORT and current_price >= signal.stop_loss)
                ):
                    exit_reason = "STOPPED_OUT"
                    exit_price = signal.stop_loss
                
                # Check take profit
                elif (
                    (signal.signal_type == TradeSignalType.LONG and current_price >= signal.take_profit) or
                    (signal.signal_type == TradeSignalType.SHORT and current_price <= signal.take_profit)
                ):
                    exit_reason = "TAKE_PROFIT"
                    exit_price = signal.take_profit
                
                # Check end of day (EOD)
                elif current_time.time() >= time(15, 15):  # 3:15 PM IST
                    exit_reason = "EOD_EXIT"
                    exit_price = current_price
                
                # Exit position if any condition is met
                if exit_reason:
                    self._exit_position(position, exit_price, exit_reason)
                    
        except Exception as e:
            self.logger.error(f"Error managing positions: {str(e)}")
    
    def _exit_position(
        self, 
        position: Position, 
        exit_price: float,
        reason: str
    ) -> None:
        """Exit a position with the given reason."""
        try:
            # Place exit order
            exit_order = self.place_order(
                symbol=position.entry_order.symbol,
                order_type=OrderType.MARKET,
                quantity=-position.entry_order.quantity,  # Opposite of entry
                product='options',
                exchange_code='NFO',
                tag=f"EXIT_{reason}_{position.position_id}"
            )
            
            if not exit_order or exit_order.status != OrderStatus.COMPLETE:
                self.logger.error(f"Failed to place exit order for {position.position_id}")
                return
            
            # Close the position
            self.portfolio.close_position(position, exit_order, reason)
            
            # Calculate P&L
            entry_price = position.entry_order.average_price
            exit_price = exit_order.average_price
            
            if position.signal.signal_type == TradeSignalType.LONG:
                pnl = (exit_price - entry_price) * position.entry_order.quantity
            else:  # SHORT
                pnl = (entry_price - exit_price) * position.entry_order.quantity
            
            pnl_pct = (pnl / (entry_price * position.entry_order.quantity)) * 100
            
            self.logger.info(
                f"Position closed: {position.position_id} | {position.signal.signal_type.value} | "
                f"Entry: {entry_price:.2f} | Exit: {exit_price:.2f} | "
                f"P&L: {pnl:.2f} ({pnl_pct:.2f}%) | Reason: {reason}"
            )
            
        except Exception as e:
            self.logger.error(f"Error exiting position {position.position_id}: {str(e)}")
    
    def square_off_all_positions(self) -> None:
        """Square off all open positions at market price."""
        self.logger.info("Squaring off all open positions...")
        
        for position in self.portfolio.get_open_positions():
            self._exit_position(position, 0, "MANUAL_EXIT")
    
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
