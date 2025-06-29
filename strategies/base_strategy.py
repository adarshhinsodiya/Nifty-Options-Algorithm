"""
Base strategy interface that all trading strategies should implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from datetime import datetime
from enum import Enum

from core.models import TradeSignal, TradeSignalType
from core.config_manager import ConfigManager


class SignalDirection(Enum):
    """Direction of the trading signal."""
    NONE = 0
    LONG = 1
    SHORT = 2


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the strategy with configuration.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.trading_config = config.get_trading_config()
        self.signal_config = config.get_signal_config()
        self.ist_tz = config.get_timezone()
        self.logger = self._setup_logger()
        
        # Track the current position state
        self.current_position: Optional[TradeSignal] = None
        self.last_signal_time: Optional[datetime] = None
    
    def _setup_logger(self):
        """Set up logger for the strategy."""
        import logging
        return logging.getLogger(self.__class__.__name__)
    
    def update_market_data(self, data: pd.DataFrame) -> None:
        """Update the strategy with the latest market data.
        
        Args:
            data: DataFrame containing OHLCV data
        """
        self.market_data = data
    
    @abstractmethod
    def check_entry_conditions(self, data: pd.DataFrame) -> bool:
        """Check if entry conditions are met for a new position.
        
        This method should ONLY check if entry conditions are met and return a boolean.
        Signal generation, logging, and trade execution should be handled by separate functions.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            bool: True if entry conditions are met, False otherwise
        """
        pass
    
    @abstractmethod
    def generate_entry_signal(self, data: pd.DataFrame) -> Optional[TradeSignal]:
        """Generate a trade signal for entry.
        
        This method should be called only after check_entry_conditions() returns True.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            TradeSignal: The generated trade signal or None if no signal
        """
        pass
    
    @abstractmethod
    def check_exit_conditions(self, data: pd.DataFrame, position: TradeSignal) -> bool:
        """Check if exit conditions are met for an existing position.
        
        This method should ONLY check if exit conditions are met and return a boolean.
        Signal generation, logging, and trade execution should be handled by separate functions.
        
        Args:
            data: DataFrame containing OHLCV data
            position: The current position to check exit conditions for
            
        Returns:
            bool: True if exit conditions are met, False otherwise
        """
        pass
        
    @abstractmethod
    def generate_exit_signal(self, data: pd.DataFrame, position: TradeSignal) -> Optional[TradeSignal]:
        """Generate a trade signal for exit.
        
        This method should be called only after check_exit_conditions() returns True.
        
        Args:
            data: DataFrame containing OHLCV data
            position: The current position to exit
            
        Returns:
            TradeSignal: The generated exit signal or None if no signal
        """
        pass
    
    def set_position(self, position: Optional[TradeSignal]) -> None:
        """Update the current position held by the strategy.
        
        Args:
            position: The current position or None if flat
        """
        self.current_position = position
        if position is not None:
            self.last_signal_time = position.entry_time
    
    def get_position(self) -> Optional[TradeSignal]:
        """Get the current position.
        
        Returns:
            Optional[TradeSignal]: Current position or None if flat
        """
        return self.current_position
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk parameters.
        
        Args:
            entry_price: Entry price of the trade
            stop_loss: Stop loss price
            
        Returns:
            int: Number of contracts/shares to trade
        """
        risk_per_trade = self.trading_config.risk_per_trade
        account_size = self.trading_config.account_size
        risk_amount = account_size * (risk_per_trade / 100.0)
        
        # Calculate position size based on risk per trade and stop loss
        risk_per_contract = abs(entry_price - stop_loss)
        if risk_per_contract == 0:
            return 0
            
        position_size = int(risk_amount / risk_per_contract)
        return max(1, position_size)  # Always trade at least 1 contract
