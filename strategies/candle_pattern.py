"""
Candle pattern strategy implementation.
"""
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

from .base_strategy import BaseStrategy, SignalDirection
from core.models import TradeSignal, TradeSignalType
from core.config_manager import ConfigManager

class CandlePatternStrategy(BaseStrategy):
    """Implements the candle pattern trading strategy with WebSocket-based real-time data."""
    
    def __init__(self, config: ConfigManager, data_provider=None):
        """Initialize the strategy with configuration and data provider.
        
        Args:
            config: Configuration manager instance
            data_provider: Optional data provider for real-time data
        """
        super().__init__(config)
        self.options_config = config.get_options_config()
        self.data_provider = data_provider
        self.market_data = None
        self._last_analysis_time = None
        self._last_signal = None
        self._candle_cache = {}  # Cache for 1-minute candles by symbol
        self._tick_cache = {}  # Cache for latest ticks by symbol
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR) for volatility measurement."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR using SMA of True Range
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_option_strike(self, spot_price: float, signal_type: str) -> int:
        """Calculate the ITM+2 strike price based on signal type."""
        strike_step = self.trading_config.strike_step
        strike_offset = self.options_config.strike_offset
        
        # Round to nearest ATM strike first
        atm_strike = round(spot_price / strike_step) * strike_step
        
        if signal_type == TradeSignalType.LONG.value:
            # For LONG (Call option), ITM+2 means 2 strikes below ATM
            strike = atm_strike - (strike_offset * strike_step)
        else:  # SHORT
            # For SHORT (Put option), ITM+2 means 2 strikes above ATM
            strike = atm_strike + (strike_offset * strike_step)
        
        # Ensure strike is a multiple of strike_step (sanity check)
        strike = int(round(strike / strike_step) * strike_step)
        
        self.logger.debug(
            f"Strike selection - Spot: {spot_price:.2f}, "
            f"Signal: {signal_type}, "
            f"ATM: {atm_strike}, "
            f"Offset: {strike_offset}, "
            f"Final Strike: {strike}"
        )
        
        return strike
    
    def _analyze_candle_pattern(self, df: pd.DataFrame) -> Tuple[SignalDirection, Optional[TradeSignal]]:
        """
        Analyze candle patterns for trading signals.
        
        Args:
            df: DataFrame with OHLCV data (last row is current candle)
            
        Returns:
            Tuple of (SignalDirection, Optional[TradeSignal])
        """
        if len(df) < 2:
            self.logger.debug("Not enough candles for analysis")
            return SignalDirection.NONE, None

        try:
            current = df.iloc[-1]  # Current candle
            prev = df.iloc[-2]     # Previous candle
            
            # Debug log candle data
            self.logger.debug(f"Analyzing candles - Current: O:{current['open']} H:{current['high']} L:{current['low']} C:{current['close']}")
            self.logger.debug(f"Previous: O:{prev['open']} H:{prev['high']} L:{prev['low']} C:{prev['close']}")
            
            # Check for valid price data
            if any(pd.isna(x) for x in [current['open'], current['high'], current['low'], current['close'],
                                       prev['open'], prev['high'], prev['low'], prev['close']]):
                self.logger.warning("Missing or invalid price data in candles")
                return SignalDirection.NONE, None
                
        except (IndexError, KeyError) as e:
            self.logger.error(f"Error accessing candle data: {str(e)}")
            return SignalDirection.NONE, None

        # Validate data
        for candle in [current, prev]:
            if any(pd.isna(x) for x in [candle['open'], candle['close'], candle['high'], candle['low']]):
                self.logger.warning("Missing candle data - skipping")
                return SignalDirection.NONE, None
                
        # Calculate candle metrics
        prev_body = abs(prev['close'] - prev['open'])
        prev_range = prev['high'] - prev['low']
        if prev_range == 0:
            self.logger.warning("Prev candle range is zero - skipping")
            return SignalDirection.NONE, None

        # Calculate wicks
        if prev['close'] < prev['open']:  # Bearish candle
            prev_top_wick = prev['high'] - prev['open']
            prev_bottom_wick = prev['close'] - prev['low']
        else:  # Bullish or neutral candle
            prev_top_wick = prev['high'] - prev['close']
            prev_bottom_wick = prev['open'] - prev['low']
            
        # Debug log wick calculations
        self.logger.debug(f"Prev body: {prev_body}, range: {prev_range}, top_wick: {prev_top_wick}, bottom_wick: {prev_bottom_wick}")
        
        signal_direction = SignalDirection.NONE
        entry_price = current['close']  # Use close price for entry
        
        # LONG signal conditions (Bullish Engulfing with confirmation)
        if (
            prev['close'] < prev['open'] and  # Previous candle is bearish
            (prev_top_wick > prev_body) and    # Upper wick is larger than body
            (prev_top_wick > prev_bottom_wick) and  # Upper wick is larger than lower wick
            (current['low'] < prev['low']) and       # Current candle makes a lower low
            (current['close'] > prev['open']) and    # Current candle closes above previous open
            (current['close'] > current['open'])     # Current candle is bullish
        ):
            signal_direction = SignalDirection.LONG
            self.logger.info("LONG signal detected")

        # SHORT signal conditions (Bearish Engulfing with confirmation)
        elif (
            prev['close'] > prev['open'] and  # Previous candle is bullish
            (prev_bottom_wick > prev_body) and  # Lower wick is larger than body
            (prev_bottom_wick > prev_top_wick) and  # Lower wick is larger than upper wick
            (current['high'] > prev['high']) and    # Current candle makes a higher high
            (current['close'] < prev['open']) and   # Current candle closes below previous open
            (current['close'] < current['open'])    # Current candle is bearish
        ):
            signal_direction = SignalDirection.SHORT
            self.logger.info("SHORT signal detected")

        if signal_direction != SignalDirection.NONE:
            # Calculate stop loss and take profit based on ATR
            atr = self.calculate_atr(df.iloc[-20:], period=self.signal_config.atr_period).iloc[-1]
            
            if signal_direction == SignalDirection.LONG:
                stop_loss = min(prev['low'], current['low']) - (atr * self.signal_config.atr_multiplier)
                take_profit = entry_price + ((entry_price - stop_loss) * 1.5)  # 1.5:1 risk:reward
                option_type = "ce"  # Call option for LONG
                signal_type = TradeSignalType.LONG
            else:  # SHORT
                stop_loss = max(prev['high'], current['high']) + (atr * self.signal_config.atr_multiplier)
                take_profit = entry_price - ((stop_loss - entry_price) * 1.5)  # 1.5:1 risk:reward
                option_type = "pe"  # Put option for SHORT
                signal_type = TradeSignalType.SHORT
            
            # Get strike price for options
            strike = self.get_option_strike(entry_price, signal_type.value)
            
            # Calculate expiry date (next Thursday)
            today = datetime.now(self.ist_tz)
            days_until_thursday = (3 - today.weekday()) % 7  # 3 = Thursday
            if days_until_thursday == 0:  # If today is Thursday, use next Thursday
                days_until_thursday = 7
            expiry_date = (today + timedelta(days=days_until_thursday)).replace(
                hour=15, minute=30, second=0, microsecond=0, tzinfo=self.ist_tz
            )
            
            # Create trade signal
            signal = TradeSignal(
                signal_type=signal_type,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                symbol=self.trading_config.symbol,
                strike=strike,
                option_type=option_type,
                expiry_date=expiry_date,
                entry_time=datetime.now(self.ist_tz),
                strategy=self.__class__.__name__
            )
            
            return signal_direction, signal
            
        return SignalDirection.NONE, None
    
    def check_entry_conditions(self, data: pd.DataFrame = None) -> bool:
        """Check if entry conditions are met for a new position.
        
        Args:
            data: Optional DataFrame with OHLCV data. If None, uses real-time data.
            
        Returns:
            bool: True if entry conditions are met, False otherwise
        """
        # Check if we already have an open position
        if self.current_position is not None:
            return False
            
        # Check cooldown period
        current_time = datetime.now(self.ist_tz)
        if self.last_signal_time is not None:
            time_since_last_signal = current_time - self.last_signal_time
            min_seconds_between_signals = self.signal_config.get('min_seconds_between_signals', 300)
            if time_since_last_signal.total_seconds() < min_seconds_between_signals:
                return False
        
        # If data is provided, use it for analysis
        if data is not None and len(data) >= 2:
            signal_direction, _ = self._analyze_candle_pattern(data)
            return signal_direction != SignalDirection.NONE
            
        # Otherwise, use real-time data if available
        if self.data_provider:
            # Get the latest candles for analysis
            symbol = self.trading_config.symbol
            candles = self.data_provider.get_latest_candles(symbol, num_candles=20)
            
            if len(candles) >= 2:
                # Convert to DataFrame for analysis
                df = pd.DataFrame(candles)
                df.set_index('timestamp', inplace=True)
                
                # Analyze the latest candles
                signal_direction, _ = self._analyze_candle_pattern(df)
                return signal_direction != SignalDirection.NONE
                
        return False
    
    def check_exit_conditions(self, data: pd.DataFrame = None, position: TradeSignal = None) -> bool:
        """Check if exit conditions are met for an existing position.
        
        Args:
            data: Optional DataFrame with OHLCV data. If None, uses real-time data.
            position: The position to check exit conditions for
            
        Returns:
            bool: True if exit conditions are met, False otherwise
        """
        if position is None and self.current_position is not None:
            position = self.current_position
            
        if position is None:
            return False
            
        # Get current price from real-time data or provided data
        current_price = None
        
        if data is not None and len(data) > 0:
            current_price = data.iloc[-1]['close']
        elif self.data_provider:
            # Try to get the latest price from the data provider
            price_data = self.data_provider.get_latest_price(position.symbol)
            if price_data:
                current_price = price_data['price']
                
        if current_price is None:
            self.logger.warning("Could not determine current price for exit check")
            return False
            
        # Check stop loss
        if position.signal_type == TradeSignalType.LONG and current_price <= position.stop_loss:
            self.logger.info(f"Stop loss hit for {position.signal_type} position at {current_price}")
            return True
            
        if position.signal_type == TradeSignalType.SHORT and current_price >= position.stop_loss:
            self.logger.info(f"Stop loss hit for {position.signal_type} position at {current_price}")
            return True
            
        # Check take profit
        if position.signal_type == TradeSignalType.LONG and current_price >= position.take_profit:
            self.logger.info(f"Take profit hit for {position.signal_type} position at {current_price}")
            return True
            
        if position.signal_type == TradeSignalType.SHORT and current_price <= position.take_profit:
            self.logger.info(f"Take profit hit for {position.signal_type} position at {current_price}")
            return True
            
        # Check time-based exit (EOD)
        if self.trading_config.exit_at_eod and not self._is_market_hours():
            self.logger.info(f"Market closing - exiting {position.signal_type} position")
            return True
            
        return False
    
    def generate_entry_signal(self, data: pd.DataFrame = None) -> Optional[TradeSignal]:
        """Generate a trade signal for entry.
        
        This is called only after check_entry_conditions() returns True.
        
        Args:
            data: Optional DataFrame containing OHLCV data. If None, uses real-time data.
            
        Returns:
            TradeSignal: The generated trade signal or None if no signal
        """
        # If data is provided, use it for signal generation
        if data is not None and len(data) >= 2:
            signal_direction, signal = self._analyze_candle_pattern(data)
            if signal_direction != SignalDirection.NONE and signal is not None:
                self.last_signal_time = datetime.now(self.ist_tz)
                return signal
                
        # Otherwise, use real-time data if available
        if self.data_provider:
            # Get the latest candles for analysis
            symbol = self.trading_config.symbol
            candles = self.data_provider.get_latest_candles(symbol, num_candles=20)
            
            if len(candles) >= 2:
                # Convert to DataFrame for analysis
                df = pd.DataFrame(candles)
                df.set_index('timestamp', inplace=True)
                
                # Generate signal from the latest data
                signal_direction, signal = self._analyze_candle_pattern(df)
                if signal_direction != SignalDirection.NONE and signal is not None:
                    self.last_signal_time = datetime.now(self.ist_tz)
                    return signal
                    
        return None
    
    def generate_exit_signal(self, data: pd.DataFrame, position: TradeSignal) -> Optional[TradeSignal]:
        """Generate a trade signal for exit.
        
        This is called only after check_exit_conditions() returns True.
        
        Args:
            data: DataFrame containing OHLCV data
            position: The current position to exit
            
        Returns:
            TradeSignal: The generated exit signal or None if no signal
        """
        # Create an exit signal based on the current position
        exit_signal = TradeSignal(
            signal_type=TradeSignalType.EXIT,
            entry_price=data.iloc[-1]['close'],
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            symbol=position.symbol,
            strike=position.strike,
            option_type=position.option_type,
            expiry_date=position.expiry_date,
            entry_time=datetime.now(self.ist_tz),
            strategy=self.__class__.__name__,
            parent_signal_id=position.signal_id
        )
        return exit_signal
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        now = datetime.now(self.ist_tz)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
            
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open.time() <= now.time() <= market_close.time()
