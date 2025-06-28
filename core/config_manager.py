"""
Configuration manager for the trading system.
Handles loading and validating configuration from INI files.
"""
import os
import configparser
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TradingConfig:
    """Trading configuration parameters."""
    symbol: str
    strike_step: int
    risk_per_trade: float
    max_open_positions: int
    max_daily_trades: int

@dataclass
class BacktestConfig:
    """Backtesting configuration parameters."""
    start_date: str
    end_date: str
    initial_capital: float
    commission: float
    slippage: float

@dataclass
class OptionsConfig:
    """Options trading configuration."""
    expiry_days: int
    option_type: str
    strike_offset: int
    lot_size: int

@dataclass
class SignalConfig:
    """Signal generation configuration."""
    rsi_period: int
    rsi_overbought: int
    rsi_oversold: int
    atr_period: int
    atr_multiplier: float

class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str = "config/config.ini"):
        """Initialize configuration manager with path to config file."""
        self.config_path = Path(config_path)
        self.config = configparser.ConfigParser()
        self._load_config()
        
    def _load_config(self) -> None:
        """Load configuration from INI file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        self.config.read(self.config_path)
        
        # Validate required sections
        required_sections = ['general', 'api', 'trading', 'backtest', 'options', 'signals']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section in config: {section}")
    
    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration."""
        return TradingConfig(
            symbol=self.config.get('trading', 'symbol'),
            strike_step=self.config.getint('trading', 'strike_step'),
            risk_per_trade=self.config.getfloat('trading', 'risk_per_trade'),
            max_open_positions=self.config.getint('trading', 'max_open_positions'),
            max_daily_trades=self.config.getint('trading', 'max_daily_trades')
        )
    
    def get_backtest_config(self) -> BacktestConfig:
        """Get backtesting configuration."""
        return BacktestConfig(
            start_date=self.config.get('backtest', 'start_date'),
            end_date=self.config.get('backtest', 'end_date'),
            initial_capital=self.config.getfloat('backtest', 'initial_capital'),
            commission=self.config.getfloat('backtest', 'commission'),
            slippage=self.config.getfloat('backtest', 'slippage')
        )
    
    def get_options_config(self) -> OptionsConfig:
        """Get options trading configuration."""
        return OptionsConfig(
            expiry_days=self.config.getint('options', 'expiry_days'),
            option_type=self.config.get('options', 'option_type'),
            strike_offset=self.config.getint('options', 'strike_offset'),
            lot_size=self.config.getint('options', 'lot_size')
        )
    
    def get_signal_config(self) -> SignalConfig:
        """Get signal generation configuration."""
        return SignalConfig(
            rsi_period=self.config.getint('signals', 'rsi_period'),
            rsi_overbought=self.config.getint('signals', 'rsi_overbought'),
            rsi_oversold=self.config.getint('signals', 'rsi_oversold'),
            atr_period=self.config.getint('signals', 'atr_period'),
            atr_multiplier=self.config.getfloat('signals', 'atr_multiplier')
        )
    
    def get_api_credentials(self) -> Dict[str, str]:
        """Get API credentials."""
        return {
            'api_key': self.config.get('api', 'api_key'),
            'api_secret': self.config.get('api', 'api_secret'),
            'session_token': self.config.get('api', 'session_token'),
            'api_url': self.config.get('api', 'api_url')
        }
    
    def get_logging_config(self) -> Dict[str, str]:
        """Get logging configuration."""
        return {
            'log_level': self.config.get('general', 'log_level'),
            'log_file': self.config.get('general', 'log_file'),
            'log_rotation': self.config.get('general', 'log_rotation'),
            'log_retention': self.config.get('general', 'log_retention')
        }
