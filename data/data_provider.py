"""
Data provider module for fetching and managing market data.
Supports both real-time and historical data retrieval.
"""
import os
import time
import random
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, TypeVar, Type, Set
import logging
import json
from pathlib import Path
from functools import wraps
from breeze_connect import BreezeConnect
from core.config_manager import ConfigManager
from core.models import TradeSignal, TradeSignalType
from core.websocket.websocket_handler import WebSocketHandler

# Type variable for generic function return type
T = TypeVar('T')

# Custom exception for session expiry
class SessionExpiredError(Exception):
    """Raised when the Breeze API session has expired."""
    pass

# Custom exception for API errors
class APIError(Exception):
    """Raised when there's an error with the Breeze API."""
    pass

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
        
        # Breeze API related attributes
        self.breeze = None
        self._session_token = None
        self._session_expiry = None
        self._last_api_call = None
        self._max_retries = 3
        self._initial_backoff = 1  # seconds
        self._max_backoff = 30  # seconds
        
        # WebSocket related attributes
        self._websocket_handler = None
        self._active_subscriptions = set()  # Track active subscriptions
        self._tick_store = {}  # Store latest ticks by symbol
        self._candle_store = {}  # Store 1-minute candles by symbol
        self._tick_lock = threading.Lock()  # Thread lock for tick store access
        
        # Initialize Breeze API for both live and backtest modes
        try:
            self._initialize_breeze_api()
            if self.mode == 'live':
                self._initialize_websocket()
        except Exception as e:
            self.logger.error(f"Failed to initialize Breeze API: {str(e)}")
            if self.mode == 'live':
                raise  # In live mode, fail fast if API initialization fails
            # In backtest mode, we'll try to continue but historical data will fail
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the data provider."""
        logger = logging.getLogger(__name__)
        return logger
    
    def _initialize_breeze_api(self) -> None:
        """Initialize the Breeze API client."""
        try:
            self.logger.info("Initializing Breeze API...")
            
            # Get API credentials
            credentials = self.config.get_api_credentials()
            
            # Log credentials (masking sensitive info)
            creds_to_log = credentials.copy()
            if 'api_key' in creds_to_log:
                creds_to_log['api_key'] = creds_to_log['api_key'][:4] + '...' + creds_to_log['api_key'][-4:]
            if 'api_secret' in creds_to_log:
                creds_to_log['api_secret'] = '***' + creds_to_log['api_secret'][-4:]
            if 'session_token' in creds_to_log and creds_to_log['session_token']:
                creds_to_log['session_token'] = creds_to_log['session_token'][:4] + '...' + creds_to_log['session_token'][-4:]
                
            self.logger.info(f"Using credentials: {creds_to_log}")
            
            # Verify required credentials
            if not credentials.get('api_key'):
                raise ValueError("Missing Breeze API key")
            if not credentials.get('api_secret'):
                raise ValueError("Missing Breeze API secret")
            
            # Initialize Breeze API client
            self.logger.info("Creating BreezeConnect instance...")
            self.breeze = BreezeConnect(api_key=credentials['api_key'])
            
            # Set the session token if available
            if credentials.get('session_token'):
                self.logger.info("Generating session with provided token...")
                try:
                    session_response = self.breeze.generate_session(
                        api_secret=credentials['api_secret'],
                        session_token=credentials['session_token']
                    )
                    self.logger.info(f"Session response: {session_response}")
                    
                    # Test the session with a simple API call
                    self.logger.info("Testing session with get_customer_details...")
                    customer_details = self.breeze.get_customer_details()
                    self.logger.info(f"Customer details: {customer_details}")
                    
                    if customer_details is None:
                        self.logger.error("Failed to get customer details. Session may be invalid.")
                    
                except Exception as session_error:
                    self.logger.error(f"Error generating session: {str(session_error)}", exc_info=True)
                    raise ValueError("Invalid session token or API secret") from session_error
            else:
                self.logger.warning("No session token provided. You'll need to generate one.")
                if self.mode == 'live':
                    self.logger.warning("Live trading requires a valid session token.")
                
                # In backtest mode, we can try to generate a new session
                if self.mode == 'backtest' and credentials.get('api_secret'):
                    self.logger.info("Attempting to generate new session for backtesting...")
                    try:
                        session_response = self.breeze.generate_session(api_secret=credentials['api_secret'])
                        self.logger.info(f"New session generated: {session_response}")
                        
                        # Test the new session
                        customer_details = self.breeze.get_customer_details()
                        self.logger.info(f"Customer details with new session: {customer_details}")
                        
                    except Exception as auth_error:
                        self.logger.error(f"Failed to generate new session: {str(auth_error)}")
                        self.logger.warning("Continuing with limited functionality")
        
        except ImportError as e:
            error_msg = "breeze-connect package not installed. Install with: pip install breeze-connect"
            self.logger.error(error_msg)
            if self.mode == 'live':
                raise ImportError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to initialize Breeze API: {str(e)}"
            self.logger.error(error_msg)
            if self.mode == 'live':
                raise APIError(error_msg) from e
            raise
    
    def _initialize_websocket(self) -> None:
        """Initialize the WebSocket handler for real-time data."""
        if not self.breeze:
            raise APIError("Breeze API not initialized")
            
        try:
            self._websocket_handler = WebSocketHandler(self.breeze, self.logger)
            # Register callback for tick data
            self._websocket_handler.register_callback('tick', self._process_tick_data)
            self.logger.info("WebSocket handler initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize WebSocket handler: {str(e)}")
            raise
            
    def _process_tick_data(self, tick_data: Dict[str, Any]) -> None:
        """Process incoming tick data and update tick store.
        
        Args:
            tick_data: Dictionary containing tick data
        """
        try:
            if not tick_data or 'symbol' not in tick_data:
                return
                
            symbol = tick_data['symbol']
            tick_time = tick_data.get('datetime', datetime.now(self.ist_tz))
            
            if not isinstance(tick_time, datetime):
                tick_time = datetime.fromisoformat(tick_time) if isinstance(tick_time, str) else datetime.now(self.ist_tz)
                
            # Update tick store with latest price
            with self._tick_lock:
                self._tick_store[symbol] = {
                    'symbol': symbol,
                    'price': tick_data.get('last', tick_data.get('ltp', 0)),
                    'volume': tick_data.get('volume', 0),
                    'timestamp': tick_time,
                    'bid': tick_data.get('bid', 0),
                    'ask': tick_data.get('ask', 0)
                }
                
                # Update 1-minute candles
                self._update_candle_store(symbol, tick_data, tick_time)
                
        except Exception as e:
            self.logger.error(f"Error processing tick data: {str(e)}", exc_info=True)
    
    def _update_candle_store(self, symbol: str, tick_data: Dict[str, Any], tick_time: datetime) -> None:
        """Update 1-minute candle store with new tick data.
        
        Args:
            symbol: Trading symbol
            tick_data: Tick data
            tick_time: Timestamp of the tick
        """
        try:
            current_minute = tick_time.replace(second=0, microsecond=0)
            price = tick_data.get('last', tick_data.get('ltp', 0))
            
            with self._tick_lock:
                if symbol not in self._candle_store:
                    self._candle_store[symbol] = {}
                    
                if current_minute not in self._candle_store[symbol]:
                    # Create new candle
                    self._candle_store[symbol][current_minute] = {
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': tick_data.get('volume', 0),
                        'count': 1
                    }
                else:
                    # Update existing candle
                    candle = self._candle_store[symbol][current_minute]
                    candle['high'] = max(candle['high'], price)
                    candle['low'] = min(candle['low'], price)
                    candle['close'] = price
                    candle['volume'] += tick_data.get('volume', 0)
                    candle['count'] += 1
                    
                # Clean up old candles (keep last 1000)
                if len(self._candle_store[symbol]) > 1000:
                    oldest_minute = min(self._candle_store[symbol].keys())
                    del self._candle_store[symbol][oldest_minute]
                    
        except Exception as e:
            self.logger.error(f"Error updating candle store: {str(e)}", exc_info=True)
    
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest price data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest price data or None if not available
        """
        with self._tick_lock:
            return self._tick_store.get(symbol, None)
    
    def get_latest_candles(self, symbol: str, num_candles: int = 20) -> List[Dict[str, Any]]:
        """Get the latest N candles for a symbol.
        
        Args:
            symbol: Trading symbol
            num_candles: Number of candles to return
            
        Returns:
            List of candle data dictionaries
        """
        with self._tick_lock:
            if symbol not in self._candle_store or not self._candle_store[symbol]:
                return []
                
            # Get the most recent candles
            sorted_minutes = sorted(self._candle_store[symbol].keys(), reverse=True)
            latest_minutes = sorted_minutes[:num_candles]
            
            # Return candles in chronological order
            return [
                {'timestamp': minute, **self._candle_store[symbol][minute]} 
                for minute in sorted(latest_minutes)
            ]
    
    def subscribe_realtime_data(
        self, 
        symbols: List[str], 
        callback: Callable[[Dict[str, Any]], None],
        exchange_code: str = 'NSE',
        product_type: str = 'cash',
        expiry_date: str = '',
        strike_price: str = '',
        right: str = ''
    ) -> None:
        """
        Subscribe to real-time market data for the given symbols.
        
        Args:
            symbols: List of stock/option symbols to subscribe to
            callback: Function to call when data is received
            exchange_code: Exchange code (NSE/NFO)
            product_type: Product type ('cash', 'futures', 'options')
            expiry_date: Expiry date for derivatives (format: 'DD-MM-YYYY')
            strike_price: Strike price for options
            right: 'call' or 'put' for options
        """
        if not self._websocket_handler:
            self.logger.error("WebSocket handler not initialized")
            return
            
        try:
            # Format symbols for WebSocket subscription
            formatted_symbols = []
            for symbol in symbols:
                # Create a unique key for this subscription
                sub_key = f"{exchange_code}:{symbol}:{product_type}"
                if product_type == 'options':
                    sub_key += f":{expiry_date}:{strike_price}:{right}"
                
                # Add to active subscriptions if not already subscribed
                if sub_key not in self._active_subscriptions:
                    formatted_symbols.append({
                        'exchange_code': exchange_code,
                        'stock_code': symbol,
                        'product_type': product_type,
                        'expiry_date': expiry_date,
                        'strike_price': strike_price,
                        'right': right
                    })
                    self._active_subscriptions.add(sub_key)
            
            if not formatted_symbols:
                self.logger.debug("No new symbols to subscribe to")
                return
                
            # Subscribe to symbols via WebSocket
            self._websocket_handler.subscribe(formatted_symbols, callback)
            self.logger.info(f"Subscribed to {len(formatted_symbols)} symbols for real-time updates")
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to real-time data: {str(e)}")
            raise
    
    def unsubscribe_realtime_data(self, symbols: List[str], callback: Callable = None) -> None:
        """
        Unsubscribe from real-time market data for the given symbols.
        
        Args:
            symbols: List of stock/option symbols to unsubscribe from
            callback: Specific callback to remove (if None, removes all callbacks for the symbol)
        """
        if not self._websocket_handler:
            return
            
        try:
            # Format symbols for WebSocket unsubscription
            formatted_symbols = []
            for symbol in symbols:
                # Find all matching subscription keys
                matching_keys = [k for k in self._active_subscriptions if k.startswith(f":{symbol}:")]
                for key in matching_keys:
                    # Parse the key to get subscription details
                    parts = key.split(':')
                    if len(parts) >= 3:
                        exchange_code = parts[0]
                        stock_code = parts[1]
                        product_type = parts[2]
                        
                        formatted_symbols.append({
                            'exchange_code': exchange_code,
                            'stock_code': stock_code,
                            'product_type': product_type
                        })
                        
                        # Remove from active subscriptions
                        self._active_subscriptions.discard(key)
            
            if not formatted_symbols:
                self.logger.debug("No active subscriptions found for the given symbols")
                return
                
            # Unsubscribe from symbols via WebSocket
            self._websocket_handler.unsubscribe(formatted_symbols, callback)
            self.logger.info(f"Unsubscribed from {len(formatted_symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from real-time data: {str(e)}")
            raise
    
    def start_websocket(self) -> None:
        """Start the WebSocket connection for real-time data."""
        if not self._websocket_handler:
            self.logger.error("WebSocket handler not initialized")
            return
            
        try:
            if not self._websocket_handler.is_connected:
                self._websocket_handler.connect()
                self.logger.info("WebSocket connection started")
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket: {str(e)}")
            raise
    
    def stop_websocket(self) -> None:
        """Stop the WebSocket connection."""
        if not self._websocket_handler:
            return
            
        try:
            if self._websocket_handler.is_connected:
                self._websocket_handler.disconnect()
                self.logger.info("WebSocket connection stopped")
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket: {str(e)}")
    
    def _is_session_valid(self) -> bool:
        """Check if the current session is valid and not expired."""
        if not self._session_token or not self._session_expiry:
            return False
            
        # Consider session expired if within 5 minutes of expiry
        buffer = timedelta(minutes=5)
        return datetime.now(pytz.utc) < (self._session_expiry - buffer)
    
    def _renew_session(self) -> None:
        """Session renewal is disabled. Using initial session token only."""
        self.logger.warning("Session renewal is disabled. Using initial session token.")
        return
    
    
    def _ensure_valid_session(self) -> None:
        """Check if the Breeze API session is valid."""
        if not self._session_token:
            raise SessionExpiredError("No session token available")
            
        # Just log a warning if session appears expired but continue anyway
        if not self._is_session_valid():
            self.logger.warning("Session appears expired but continuing with current token")
    
    def _api_call_with_retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute an API call with retry logic and session management.
        
        Args:
            func: The API function to call
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the API call
            
        Raises:
            APIError: If the API call fails after all retries
            SessionExpiredError: If session renewal fails
        """
        last_exception = None
        
        for attempt in range(self._max_retries):
            try:
                # Ensure we have a valid session before each attempt
                self._ensure_valid_session()
                
                # Make the API call
                result = func(*args, **kwargs)
                self._last_api_call = datetime.now(pytz.utc)
                
                # Check for API errors in the response
                if isinstance(result, dict) and 'Status' in result and result['Status'] != 200:
                    error_msg = result.get('error', 'Unknown API error')
                    
                    # If session expired, try to renew and retry
                    if 'session expired' in str(error_msg).lower():
                        self.logger.warning("Session expired, attempting to renew...")
                        self._renew_session()
                        continue
                        
                    raise APIError(f"API error: {error_msg}")
                
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                
                # Calculate backoff with jitter
                backoff = min(
                    self._initial_backoff * (2 ** attempt) + random.uniform(0, 1),
                    self._max_backoff
                )
                
                # Don't sleep on the last attempt
                if attempt < self._max_retries - 1:
                    self.logger.info(f"Retrying in {backoff:.2f} seconds...")
                    time.sleep(backoff)
        
        # If we get here, all retries failed
        self.logger.error(f"API call failed after {self._max_retries} attempts")
        if isinstance(last_exception, APIError):
            raise last_exception
        raise APIError(f"API call failed: {str(last_exception)}") from last_exception
    
    def _get_historical_data_impl(self, symbol: str, from_date: datetime, to_date: datetime,
                               interval: str = '1minute', exchange_code: str = 'NSE',
                               product_type: str = 'cash') -> Dict:
        """Implementation of historical data fetch with retry logic."""
        try:
            # Format dates as required by Breeze API
            from_str = from_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            to_str = to_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            
            self.logger.info(f"Fetching historical data for {symbol} from {from_str} to {to_str}")
            self.logger.info(f"Exchange: {exchange_code}, Product: {product_type}, Interval: {interval}")
            
            # Log the exact request being made
            request_data = {
                'interval': interval,
                'from_date': from_str,
                'to_date': to_str,
                'stock_code': symbol,
                'exchange_code': exchange_code,
                'product_type': product_type
            }
            self.logger.info(f"API Request: {request_data}")
            
            try:
                # Make the API call
                result = self.breeze.get_historical_data(**request_data)
                
                if result is None:
                    self.logger.error("API returned None response. This usually indicates an authentication issue.")
                    # Try to get more detailed error information
                    try:
                        # Try a simple API call to check authentication
                        test = self.breeze.get_customer_details()
                        self.logger.info(f"Test API call response: {test}")
                        
                        # Check if we can get any data with a simpler request
                        test_data = self.breeze.get_historical_data(
                            interval='1day',
                            from_date='2024-01-01T00:00:00.000Z',
                            to_date='2024-01-02T00:00:00.000Z',
                            stock_code='NIFTY',
                            exchange_code='NSE',
                            product_type='cash'
                        )
                        self.logger.info(f"Test data response: {test_data}")
                        
                    except Exception as auth_error:
                        self.logger.error(f"Authentication check failed: {str(auth_error)}", exc_info=True)
                
                self.logger.info(f"Raw API response type: {type(result)}")
                self.logger.info(f"Raw API response: {result}")
                return result if result is not None else {}
                
            except Exception as e:
                self.logger.error(f"Error in Breeze API call: {str(e)}", exc_info=True)
                raise
            
        except Exception as e:
            self.logger.error(f"Error in _get_historical_data_impl: {str(e)}", exc_info=True)
            raise  # Re-raise the exception to be handled by the caller
            
        
    def get_historical_data(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        interval: str = '1minute',
        exchange_code: str = 'NSE',
        product_type: str = 'cash'
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data with retry and session management.
        
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
        if not self.breeze:
            self.logger.error("Breeze API not initialized. Check your API credentials in the .env file")
            self.logger.info("Please ensure you have set BREEZE_API_KEY and BREEZE_API_SECRET in your .env file")
            return pd.DataFrame()
            
        try:
            # First try with NSE/cash (index data)
            try:
                self.logger.info("Attempting to fetch NIFTY index data (NSE/cash)")
                result = self._api_call_with_retry(
                    self._get_historical_data_impl,
                    symbol='NIFTY',
                    from_date=from_date,
                    to_date=to_date,
                    interval=interval,
                    exchange_code='NSE',
                    product_type='cash'
                )
                
                if not result or not isinstance(result, dict) or not result.get('Success', False):
                    error_msg = result.get('Error', 'Unknown error') if isinstance(result, dict) else 'Invalid response format'
                    self.logger.warning(f"Failed to fetch NIFTY index data: {error_msg}")
                    raise APIError(f"Failed to fetch NIFTY index data: {error_msg}")
                    
            except Exception as e:
                self.logger.warning(f"Error fetching NIFTY index data: {str(e)}")
                
                # Fall back to NFO/options if NSE/cash fails
                try:
                    self.logger.info("Falling back to NIFTY options data (NFO/options)")
                    result = self._api_call_with_retry(
                        self._get_historical_data_impl,
                        symbol='NIFTY',
                        from_date=from_date,
                        to_date=to_date,
                        interval=interval,
                        exchange_code='NFO',
                        product_type='options'
                    )
                    
                    if not result or not isinstance(result, dict) or not result.get('Success', False):
                        error_msg = result.get('Error', 'Unknown error') if isinstance(result, dict) else 'Invalid response format'
                        self.logger.error(f"Failed to fetch NIFTY options data: {error_msg}")
                        raise APIError(f"Failed to fetch NIFTY options data: {error_msg}")
                        
                except Exception as options_error:
                    self.logger.error(f"Error fetching NIFTY options data: {str(options_error)}")
                    raise APIError(f"All data fetch attempts failed. Last error: {str(options_error)}")
            
            self.logger.info(f"API call completed. Response type: {type(result)}")
            if result is None:
                self.logger.error("API returned None response")
            elif isinstance(result, dict):
                self.logger.info(f"API response keys: {result.keys()}")
            else:
                self.logger.info(f"API response length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            
            if not result or 'Success' not in result or not result['Success']:
                error_msg = result.get('Error', 'Unknown error') if isinstance(result, dict) else 'Invalid response format'
                self.logger.error(f"Failed to fetch historical data: {error_msg}")
                return pd.DataFrame()
                
            # Convert to DataFrame and standardize column names
            df = pd.DataFrame(result['Success'])
            if df.empty:
                self.logger.warning(f"No data returned for {symbol} from {from_date} to {to_date}")
                return df
                
            # Convert string dates to datetime
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
            
            self.logger.info(f"Successfully fetched {len(df)} candles for {symbol} from {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}", exc_info=True)
            return pd.DataFrame()
            
    def _get_option_chain_impl(
        self,
        expiry_date: datetime,
        symbol: str = 'NIFTY',
        exchange_code: str = 'NFO'
    ) -> Dict:
        """Internal implementation of get_option_chain without retry logic."""
        if not self.breeze:
            raise APIError("Breeze API not initialized")
            
        # Format expiry date as 'dd-MMM-YYYY'
        expiry_str = expiry_date.strftime('%d-%b-%Y').upper()
        
        # Get option chain from Breeze API
        chain = self.breeze.get_option_chain_quotes(
            exchange_code=exchange_code,
            stock_code=symbol,
            product_type='options',
            expiry_date=expiry_str
        )
        
        if not chain or 'Success' not in chain:
            raise APIError(f"Failed to fetch option chain for {symbol} {expiry_str}")
        
        return chain

    def get_option_chain(
        self,
        expiry_date: datetime,
        symbol: str = 'NIFTY',
        exchange_code: str = 'NFO'
    ) -> pd.DataFrame:
        """Fetch option chain for the given expiry date with retry and session management.
        
        Args:
            expiry_date: Expiry date of the options
            symbol: Underlying symbol (default: 'NIFTY')
            exchange_code: Exchange code (default: 'NFO')
            
        Returns:
            DataFrame with option chain data
        """
        try:
            # Call with retry and session management
            chain = self._api_call_with_retry(
                self._get_option_chain_impl,
                expiry_date=expiry_date,
                symbol=symbol,
                exchange_code=exchange_code
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(chain['Success'])
            
            # Extract strike price and option type from symbol
            df['strike'] = df['strike_price'].astype(float)
            df['type'] = df['option_type'].str.upper()
            
            # Filter for standard expiry (last Thursday of the month)
            expiry_str = expiry_date.strftime('%d-%b-%Y').upper()
            df = df[df['expiry_date'] == expiry_str]
            
            return df
            
        except APIError as e:
            self.logger.error(f"API error in get_option_chain: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error in get_option_chain: {str(e)}")
        
        return pd.DataFrame()
    
    def _get_instrument_quotes_impl(
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
        """Internal implementation of get_instrument_quotes without retry logic."""
        if not self.breeze:
            raise APIError("Breeze API not initialized")
            
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
        
        if not quotes or 'Success' not in quotes:
            raise APIError(f"Failed to get quotes for {stock_code}")
            
        return quotes

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
        """Get quotes for a specific instrument with retry and session management."""
        try:
            # Call with retry and session management
            quotes = self._api_call_with_retry(
                self._get_instrument_quotes_impl,
                exchange_code=exchange_code,
                stock_code=stock_code,
                product_type=product_type,
                expiry_date=expiry_date,
                strike_price=strike_price,
                right=right,
                get_exchange_quotes=get_exchange_quotes,
                get_market_depth=get_market_depth
            )
            
            return quotes.get('Success', {})
            
        except APIError as e:
            self.logger.error(f"API error in get_instrument_quotes: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error in get_instrument_quotes: {str(e)}")
        
        return {}
    
    def get_option_instrument(
        self,
        symbol: str,
        strike: int,
        option_type: str,
        expiry_date: datetime
    ) -> Dict:
        """Get instrument details for an option contract with retry and session management."""
        try:
            # Get option chain for the expiry date (this already has retry logic)
            chain = self.get_option_chain(expiry_date, symbol)
            if chain.empty:
                self.logger.warning(f"No option chain found for {symbol} {expiry_date}")
                return {}
            
            # Find matching option
            option_type = option_type.upper()
            matching = chain[
                (chain['strike'] == strike) & 
                (chain['type'] == option_type)
            ]
            
            if matching.empty:
                self.logger.warning(f"No matching option found: {symbol} {strike} {option_type}")
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
            self.logger.error(f"Error in get_option_instrument: {str(e)}")
            return {}
    
    def _subscribe_feeds_impl(self, instruments: List[Dict]) -> None:
        """Internal implementation of subscribe_feeds without retry logic."""
        if not self.breeze:
            raise APIError("Breeze API not initialized")
            
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

    def subscribe_feeds(self, instruments: List[Dict], callback: Callable) -> None:
        """Subscribe to real-time market data feeds with retry and session management.
        
        Args:
            instruments: List of instrument dictionaries with exchange_code and stock_code
            callback: Function to call when data is received
        """
        try:
            # Call with retry and session management
            self._api_call_with_retry(
                self._subscribe_feeds_impl,
                instruments=instruments
            )
            
            # Set the callback function
            self.breeze.on_ticks = callback
            
        except APIError as e:
            self.logger.error(f"API error in subscribe_feeds: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error in subscribe_feeds: {str(e)}")
    
    def _get_latest_price_impl(self, symbol: str, exchange_code: str = 'NSE') -> Dict:
        """Internal implementation of get_latest_price without retry logic."""
        if not self.breeze:
            raise APIError("Breeze API not initialized")
            
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
            raise APIError(f"Failed to get latest price for {symbol}")
            
        return quotes

    def get_latest_price(self, symbol: str, exchange_code: str = 'NSE') -> float:
        """Get the latest price for a symbol with retry and session management."""
        try:
            # Call with retry and session management
            quotes = self._api_call_with_retry(
                self._get_latest_price_impl,
                symbol=symbol,
                exchange_code=exchange_code
            )
            
            return float(quotes['Success']['ltp'])
            
        except APIError as e:
            self.logger.error(f"API error in get_latest_price: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error in get_latest_price: {str(e)}")
        
        return 0.0
