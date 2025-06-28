"""
Candle pattern strategy implementation.
"""
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

from core.models import TradeSignal, TradeSignalType
from core.config_manager import ConfigManager

class CandlePatternStrategy:
    """Implements the candle pattern trading strategy."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the strategy with configuration."""
        self.config = config
        self.trading_config = config.get_trading_config()
        self.options_config = config.get_options_config()
        self.signal_config = config.get_signal_config()
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Set up logger for the strategy."""
        import logging
        logger = logging.getLogger(__name__)
        return logger
    
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
        """Calculate the ITM+2 strike price based on signal type.
        
        Args:
            spot_price: Current spot price of the underlying
            signal_type: Type of signal (LONG or SHORT)
            
        Returns:
            int: Strike price that is ITM+2 based on the signal type, properly rounded
                to the nearest strike step.
        """
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
    
    def analyze_candle_pattern(self, df: pd.DataFrame, index: int) -> Tuple[Optional[str], Optional[TradeSignal]]:
        """
        Analyze 2-candle pattern (previous + current) for trading signals.
        
        Args:
            df: DataFrame with OHLCV data
            index: Current index in the DataFrame
            
        Returns:
            Tuple of (signal_type, TradeSignal) or (None, None) if no signal
        """
        if index < 0:
            index = len(df) + index
        if index >= len(df) or index < 1:
            self.logger.debug("Not enough candles for analysis")
            return None, None

        try:
            current = df.iloc[index]
            prev = df.iloc[index - 1]
            
            # Debug log candle data
            self.logger.debug(f"Current candle: O:{current['open']} H:{current['high']} L:{current['low']} C:{current['close']}")
            self.logger.debug(f"Previous candle: O:{prev['open']} H:{prev['high']} L:{prev['low']} C:{prev['close']}")
            
        except IndexError:
            self.logger.error("Invalid index for current or previous candle")
            return None, None

        # Validate data
        for candle in [current, prev]:
            if any(pd.isna(x) for x in [candle['open'], candle['close'], candle['high'], candle['low']]):
                self.logger.warning("Missing candle data - skipping")
                return None, None
                
        # Calculate candle metrics
        prev_body = abs(prev['close'] - prev['open'])
        prev_range = prev['high'] - prev['low']
        if prev_range == 0:
            self.logger.warning("Prev candle range is zero - skipping")
            return None, None

        # Calculate wicks
        if prev['close'] < prev['open']:  # Bearish candle
            prev_top_wick = prev['high'] - prev['open']
            prev_bottom_wick = prev['close'] - prev['low']
        else:  # Bullish or neutral candle
            prev_top_wick = prev['high'] - prev['close']
            prev_bottom_wick = prev['open'] - prev['low']
            
        # Debug log wick calculations
        self.logger.debug(f"Prev body: {prev_body}, range: {prev_range}, top_wick: {prev_top_wick}, bottom_wick: {prev_bottom_wick}")
        
        signal_type = None
        entry_price = current['open']
        
        # LONG signal conditions (Bullish Engulfing with confirmation)
        if (
            prev['close'] < prev['open'] and  # Previous candle is bearish
            (prev_top_wick > prev_body) and    # Upper wick is larger than body
            (prev_top_wick > prev_bottom_wick) and  # Upper wick is larger than lower wick
            (current['low'] < prev['low']) and       # Current candle makes a lower low
            (current['close'] > prev['open']) and    # Current candle closes above previous open
            (current['close'] > current['open'])     # Current candle is bullish
        ):
            signal_type = TradeSignalType.LONG
            self.logger.info("LONG signal generated")

        # SHORT signal conditions (Bearish Engulfing with confirmation)
        elif (
            prev['close'] > prev['open'] and  # Previous candle is bullish
            (prev_bottom_wick > prev_body) and  # Lower wick is larger than body
            (prev_bottom_wick > prev_top_wick) and  # Lower wick is larger than upper wick
            (current['high'] > prev['high']) and    # Current candle makes a higher high
            (current['close'] < prev['open']) and   # Current candle closes below previous open
            (current['close'] < current['open'])    # Current candle is bearish
        ):
            signal_type = TradeSignalType.SHORT
            self.logger.info("SHORT signal generated")

        if signal_type:
            # Calculate stop loss and take profit based on ATR
            atr = self.calculate_atr(df.iloc[max(0, index-20):index+1], period=self.signal_config.atr_period).iloc[-1]
            
            if signal_type == TradeSignalType.LONG:
                stop_loss = min(prev['low'], current['low']) - (atr * self.signal_config.atr_multiplier)
                take_profit = entry_price + ((entry_price - stop_loss) * 1.5)  # 1.5:1 risk:reward
                option_type = "ce"  # Call option for LONG
            else:  # SHORT
                stop_loss = max(prev['high'], current['high']) + (atr * self.signal_config.atr_multiplier)
                take_profit = entry_price - ((stop_loss - entry_price) * 1.5)  # 1.5:1 risk:reward
                option_type = "pe"  # Put option for SHORT
            
            # Get strike price for options
            strike = self.get_option_strike(entry_price, signal_type.value)
            
            # Calculate expiry date (next Thursday)
            today = datetime.now(self.ist_tz)
            days_until_thursday = (3 - today.weekday()) % 7  # 3 = Thursday
            if days_until_thursday == 0:  # If today is Thursday, use next Thursday
                days_until_thursday = 7
            expiry_date = (today + timedelta(days=days_until_thursday)).replace(hour=15, minute=30, second=0, microsecond=0)
            
            return signal_type.value, TradeSignal(
                signal_type=signal_type,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strike=strike,
                option_type=option_type,
                timestamp=datetime.now(self.ist_tz),
                spot_price=entry_price,
                expiry_date=expiry_date
            )

        return None, None
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for the entire DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signal column added
        """
        if df.empty:
            return df
            
        # Add technical indicators
        df['rsi'] = self.calculate_rsi(df, period=self.signal_config.rsi_period)
        df['atr'] = self.calculate_atr(df, period=self.signal_config.atr_period)
        
        # Initialize signal column
        df['signal'] = None
        
        # Generate signals
        for i in range(1, len(df)):
            signal_type, signal = self.analyze_candle_pattern(df, i)
            if signal_type and signal:
                df.at[df.index[i], 'signal'] = signal_type
                # Store the full signal object for later use
                df.at[df.index[i], '_signal_obj'] = signal
        
        return df
