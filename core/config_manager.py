"""
Configuration manager for the trading system.
Handles loading configuration from INI files and environment variables.
"""
import os
import configparser
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size: float
    max_portfolio_risk: float
    max_daily_loss_pct: float

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
        required_sections = ['general', 'api', 'trading', 'backtest', 'options', 'signals', 'risk']
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
        """Get API credentials from environment variables.
        
        Returns:
            Dict containing API credentials with default values if not set.
            
        Raises:
            ValueError: If required API credentials are missing from environment.
        """
        credentials = {
            'api_key': os.getenv('BREEZE_API_KEY'),
            'api_secret': os.getenv('BREEZE_API_SECRET'),
            'session_token': os.getenv('BREEZE_SESSION_TOKEN'),
            'api_url': os.getenv('BREEZE_API_URL', 'https://api.icicidirect.com/breezeapi/api/v2/')
        }
        
        # Check for required credentials
        if not credentials['api_key'] or not credentials['api_secret']:
            raise ValueError(
                "Missing required API credentials. Please set BREEZE_API_KEY and BREEZE_API_SECRET "
                "in your .env file or environment variables.")
                
        return credentials
    
    def get_logging_config(self) -> Dict[str, str]:
        """Get logging configuration."""
        return {
            'log_level': self.config.get('general', 'log_level'),
            'log_file': self.config.get('general', 'log_file'),
            'log_rotation': self.config.get('general', 'log_rotation'),
            'log_retention': self.config.get('general', 'log_retention')
        }
    
    def get_risk_config(self) -> RiskConfig:
        """Get risk management configuration."""
        return RiskConfig(
            max_position_size=self.config.getfloat('risk', 'max_position_size'),
            max_portfolio_risk=self.config.getfloat('risk', 'max_portfolio_risk'),
            max_daily_loss_pct=self.config.getfloat('risk', 'max_daily_loss_pct')
        )
        
    def get_timezone(self):
        """Get the timezone from configuration.
        
        Returns:
            str: Timezone string (default: 'Asia/Kolkata')
        """
        return self.config.get('general', 'timezone', fallback='Asia/Kolkata')
