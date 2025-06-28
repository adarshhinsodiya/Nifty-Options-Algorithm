"""
Data provider module for fetching and managing market data.
Supports both real-time and historical data retrieval.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional, Tuple, Union
import logging
import json
from pathlib import Path

from core.config_manager import ConfigManager
from core.models import TradeSignal, TradeSignalType

class DataProvider:
    """Handles data retrieval from various sources (live/historical)."""
    
    def __init__(self, config: ConfigManager, mode: str = 'backtest'):
        """Initialize data provider with configuration and mode.
        
        Args:
            config: Configuration manager instance
            mode: Operation mode ('backtest' or 'live')
        """
        self.config = config
        self.mode = mode.lower()
        self.trading_config = config.get_trading_config()
        self.options_config = config.get_options_config()
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.logger = self._setup_logger()
        self.breeze = None
        
        if self.mode == 'live':
            self._initialize_breeze_api()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the data provider."""
        logger = logging.getLogger(__name__)
        return logger
    
    def _initialize_breeze_api(self) -> None:
        """Initialize ICICI Direct Breeze API connection."""
        try:
            from breeze_connect import BreezeConnect
            
            api_creds = self.config.get_api_credentials()
            self.breeze = BreezeConnect(api_key=api_creds['api_key'])
            
            # Generate session token if not provided
            if not api_creds.get('session_token'):
                # This requires user interaction in a real scenario
                self.logger.warning("Session token not provided. Generate one using generate_session()")
            else:
                self.breeze.generate_session(
                    api_secret=api_creds['api_secret'],
                    session_token=api_creds['session_token']
                )
                self.breeze.ws_connect()  # Connect to websocket for live data
                
        except ImportError:
            self.logger.error("breeze-connect package not installed. Install with: pip install breeze-connect")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Breeze API: {str(e)}")
            raise
    
    def get_historical_data(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        interval: str = '1minute',
        exchange_code: str = 'NSE',
        product_type: str = 'cash'
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.
        
        Args:
            symbol: Trading symbol (e.g., 'NIFTY', 'RELIANCE')
            from_date: Start date
            to_date: End date
            interval: Data interval ('1minute', '5minute', '1day', etc.)
            exchange_code: Exchange code (NSE/NFO)
            product_type: Product type ('cash' for equity, 'futures', 'options')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if self.mode == 'backtest':
                # For backtesting, try to load from file first
                file_path = f"data/historical/{symbol}_{interval}.csv"
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, parse_dates=['date'])
                    df.set_index('date', inplace=True)
                    return df
                
                # If file doesn't exist, fetch from API
                if not self.breeze:
                    self.logger.error("Breeze API not initialized")
                    return pd.DataFrame()
                
                data = self.breeze.get_historical_data_v2(
                    from_date=from_date.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                    to_date=to_date.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                    stock_code=symbol,
                    exchange_code=exchange_code,
                    product_type=product_type,
                    interval=interval
                )
                
                if not data or 'Success' not in data:
                    self.logger.error(f"Failed to fetch historical data: {data}")
                    return pd.DataFrame()
                
                df = pd.DataFrame(data['Success'])
                
                # Convert and rename columns
                df['date'] = pd.to_datetime(df['datetime'])
                df.rename(columns={
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                }, inplace=True)
                
                # Set index and sort
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                # Save to file for future use
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                df.to_csv(file_path)
                
                return df
                
            else:  # Live mode
                if not self.breeze:
                    self.logger.error("Breeze API not initialized")
                    return pd.DataFrame()
                
                data = self.breeze.get_historical_data_v2(
                    from_date=from_date.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                    to_date=to_date.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                    stock_code=symbol,
                    exchange_code=exchange_code,
                    product_type=product_type,
                    interval=interval
                )
                
                if not data or 'Success' not in data:
                    self.logger.error(f"Failed to fetch historical data: {data}")
                    return pd.DataFrame()
                
                df = pd.DataFrame(data['Success'])
                
                # Convert and rename columns
                df['date'] = pd.to_datetime(df['datetime'])
                df.rename(columns={
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                }, inplace=True)
                
                # Set index and sort
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                return df
                
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    def get_option_chain(
        self,
        expiry_date: datetime,
        symbol: str = 'NIFTY',
        exchange_code: str = 'NFO'
    ) -> pd.DataFrame:
        """Fetch option chain for the given expiry date.
        
        Args:
            expiry_date: Expiry date of the options
            symbol: Underlying symbol (default: 'NIFTY')
            exchange_code: Exchange code (default: 'NFO')
            
        Returns:
            DataFrame with option chain data
        """
        try:
            if not self.breeze:
                self.logger.error("Breeze API not initialized")
                return pd.DataFrame()
            
            # Format expiry date as 'dd-MMM-YYYY'
            expiry_str = expiry_date.strftime('%d-%b-%Y').upper()
            
            # Get option chain
            chain = self.breeze.get_option_chain_quotes(
                stock_code=symbol,
                exchange_code=exchange_code,
                product_type="options",
                expiry_date=expiry_str
            )
            
            if not chain or 'Success' not in chain:
                self.logger.error(f"Failed to fetch option chain: {chain}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(chain['Success'])
            
            # Extract strike price and option type from symbol
            df['strike'] = df['strike_price'].astype(float)
            df['type'] = df['option_type'].str.upper()
            
            # Filter for standard expiry (last Thursday of the month)
            df = df[df['expiry_date'] == expiry_str]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching option chain: {str(e)}")
            return pd.DataFrame()
    
    def get_instrument_quotes(
        self,
        exchange_code: str,
        stock_code: str,
        product_type: str = 'cash',
        expiry_date: str = '',
        strike_price: str = '',
        right: str = '',
        get_exchange_quotes: bool = False,
        get_market_depth: bool = False
    ) -> Dict:
        """Get quotes for a specific instrument."""
        try:
            if not self.breeze:
                self.logger.error("Breeze API not initialized")
                return {}
                
            quotes = self.breeze.get_quotes(
                stock_code=stock_code,
                exchange_code=exchange_code,
                product_type=product_type,
                expiry_date=expiry_date,
                strike_price=strike_price,
                right=right,
                get_exchange_quotes=get_exchange_quotes,
                get_market_depth=get_market_depth
            )
            
            return quotes.get('Success', {}) if quotes else {}
            
        except Exception as e:
            self.logger.error(f"Error getting instrument quotes: {str(e)}")
            return {}
    
    def get_option_instrument(
        self,
        symbol: str,
        strike: int,
        option_type: str,
        expiry_date: datetime
    ) -> Dict:
        """Get instrument details for an option contract."""
        try:
            # Get option chain for the expiry date
            chain = self.get_option_chain(expiry_date, symbol)
            if chain.empty:
                return {}
            
            # Find matching option
            option_type = option_type.upper()
            matching = chain[
                (chain['strike'] == strike) & 
                (chain['type'] == option_type)
            ]
            
            if matching.empty:
                self.logger.warning(f"No matching option found: {symbol} {strike} {option_type} {expiry_date}")
                return {}
            
            # Get the first match (should be only one)
            option = matching.iloc[0]
            
            return {
                'exchange_code': 'NFO',
                'stock_code': option['trading_symbol'],
                'strike': strike,
                'option_type': option_type,
                'expiry_date': expiry_date.strftime('%Y-%m-%d'),
                'lot_size': int(option['lot_size']),
                'tick_size': float(option['tick_size'])
            }
            
        except Exception as e:
            self.logger.error(f"Error getting option instrument: {str(e)}")
            return {}
    
    def subscribe_feeds(self, instruments: List[Dict], callback) -> None:
        """Subscribe to real-time market data feeds.
        
        Args:
            instruments: List of instrument dictionaries with exchange_code and stock_code
            callback: Function to call when data is received
        """
        if not self.breeze:
            self.logger.error("Breeze API not initialized")
            return
            
        try:
            # Unsubscribe from all previous subscriptions
            self.breeze.unsubscribe_feeds()
            
            # Subscribe to new instruments
            for inst in instruments:
                self.breeze.subscribe_feeds(
                    stock_code=inst['stock_code'],
                    exchange_code=inst['exchange_code'],
                    product_type=inst.get('product_type', 'cash'),
                    expiry_date=inst.get('expiry_date', ''),
                    strike_price=inst.get('strike_price', ''),
                    right=inst.get('right', ''),
                    get_exchange_quotes=inst.get('get_exchange_quotes', True),
                    get_market_depth=inst.get('get_market_depth', False)
                )
            
            # Set the callback function
            self.breeze.on_ticks = callback
            
        except Exception as e:
            self.logger.error(f"Error subscribing to feeds: {str(e)}")
    
    def get_latest_price(self, symbol: str, exchange_code: str = 'NSE') -> float:
        """Get the latest price for a symbol."""
        try:
            if not self.breeze:
                self.logger.error("Breeze API not initialized")
                return 0.0
                
            quotes = self.breeze.get_quotes(
                stock_code=symbol,
                exchange_code=exchange_code,
                product_type='cash',
                expiry_date='',
                right='',
                strike_price='0',
                get_exchange_quotes=True,
                get_market_depth=False
            )
            
            if not quotes or 'Success' not in quotes:
                self.logger.error(f"Failed to get latest price for {symbol}")
                return 0.0
                
            return float(quotes['Success']['ltp'])
            
        except Exception as e:
            self.logger.error(f"Error getting latest price: {str(e)}")
            return 0.0
