"""
Live Data Manager for real-time market data distribution.

This module provides a centralized service for managing real-time market data
subscriptions and distributing updates to multiple consumers.
"""
import logging
import time
import threading
from queue import Queue, Empty
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
from datetime import datetime, timedelta
import pytz

from core.websocket.websocket_handler import WebSocketHandler

# Type aliases
DataCallback = Callable[[Dict[str, Any]], None]
Symbol = str

class LiveDataManager:
    """
    Manages real-time market data subscriptions and distribution.
    
    This class acts as a central hub for real-time market data, handling:
    - WebSocket connection management
    - Symbol subscription management
    - Data distribution to multiple consumers
    - Caching of recent market data
    - Automatic reconnection and error recovery
    """
    
    def __init__(self, websocket_handler: WebSocketHandler, logger: Optional[logging.Logger] = None):
        """
        Initialize the LiveDataManager.
        
        Args:
            websocket_handler: Initialized WebSocketHandler instance
            logger: Optional logger instance
        """
        self.ws = websocket_handler
        self.logger = logger or logging.getLogger(__name__)
        
        # Data structures
        self._tick_store: Dict[Symbol, Dict[str, Any]] = {}
        self._candle_store: Dict[Tuple[Symbol, str], List[Dict[str, Any]]] = {}
        self._subscribers: Dict[Symbol, Set[DataCallback]] = {}
        self._candle_subscribers: Dict[Tuple[Symbol, str], Set[DataCallback]] = {}
        self._lock = threading.RLock()
        
        # Candle aggregation settings
        self.candle_intervals = ['1m', '5m', '15m', '1h']
        self.max_candles = 1000  # Max candles to keep in memory per symbol/interval
        
        # Register WebSocket callbacks
        self.ws.register_callback('tick', self._on_tick_update)
        self.ws.register_callback('error', self._on_error)
        self.ws.register_callback('connected', self._on_ws_connected)
        self.ws.register_callback('disconnected', self._on_ws_disconnected)
        
        # Start background tasks
        self._stop_event = threading.Event()
        self._candle_aggregator_thread = threading.Thread(
            target=self._candle_aggregation_loop,
            name="CandleAggregator",
            daemon=True
        )
        self._candle_aggregator_thread.start()
    
    def start(self) -> None:
        """Start the LiveDataManager and underlying WebSocket connection."""
        if not self.ws.is_connected:
            self.ws.connect()
    
    def stop(self) -> None:
        """Stop the LiveDataManager and clean up resources."""
        self._stop_event.set()
        if self.ws.is_connected:
            self.ws.disconnect()
    
    def subscribe_ticks(self, symbols: List[str], callback: DataCallback) -> None:
        """
        Subscribe to tick updates for the given symbols.
        
        Args:
            symbols: List of symbols to subscribe to (e.g., ['NIFTY', 'RELIANCE'])
            callback: Function to call when tick data is received
        """
        if not symbols:
            return
            
        with self._lock:
            for symbol in symbols:
                symbol = symbol.upper()
                if symbol not in self._subscribers:
                    self._subscribers[symbol] = set()
                    # Subscribe to WebSocket if not already subscribed
                    if symbol not in self.ws.subscribed_symbols:
                        self.ws.subscribe([symbol])
                
                self._subscribers[symbol].add(callback)
                self.logger.debug(f"Subscribed to ticks for {symbol}")
    
    def unsubscribe_ticks(self, symbols: List[str], callback: DataCallback) -> None:
        """
        Unsubscribe from tick updates for the given symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
            callback: Callback function to remove
        """
        if not symbols:
            return
            
        with self._lock:
            for symbol in symbols:
                symbol = symbol.upper()
                if symbol in self._subscribers and callback in self._subscribers[symbol]:
                    self._subscribers[symbol].remove(callback)
                    self.logger.debug(f"Unsubscribed from ticks for {symbol}")
                    
                    # If no more subscribers, consider unsubscribing from WebSocket
                    if not self._subscribers[symbol]:
                        del self._subscribers[symbol]
                        # Note: We don't immediately unsubscribe from WebSocket as other
                        # components might be using the same symbol
    
    def get_latest_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest tick data for a symbol.
        
        Args:
            symbol: Symbol to get tick data for
            
        Returns:
            Latest tick data as a dictionary, or None if not available
        """
        return self._tick_store.get(symbol.upper())
    
    def get_historical_ticks(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical tick data for a symbol.
        
        Note: This is a placeholder. In a production system, you would typically
        fetch this from a time-series database.
        
        Args:
            symbol: Symbol to get tick data for
            limit: Maximum number of ticks to return
            
        Returns:
            List of historical ticks (most recent first)
        """
        # In a real implementation, this would query a time-series database
        return []
    
    def get_candles(self, symbol: str, interval: str = '1m', limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical candle data for a symbol and interval.
        
        Args:
            symbol: Symbol to get candle data for
            interval: Candle interval (e.g., '1m', '5m', '15m', '1h')
            limit: Maximum number of candles to return
            
        Returns:
            List of historical candles (most recent first)
        """
        key = (symbol.upper(), interval)
        with self._lock:
            candles = self._candle_store.get(key, [])
            return candles[-limit:] if limit else candles
    
    def _on_tick_update(self, tick: Dict[str, Any]) -> None:
        """Handle incoming tick data from WebSocket."""
        symbol = tick.get('symbol')
        if not symbol:
            return
            
        symbol = symbol.upper()
        timestamp = tick.get('timestamp', datetime.utcnow().isoformat())
        
        # Update tick store
        with self._lock:
            self._tick_store[symbol] = {
                'symbol': symbol,
                'timestamp': timestamp,
                'last_price': tick.get('last_price'),
                'volume': tick.get('volume'),
                'bid': tick.get('bid'),
                'ask': tick.get('ask'),
                'bid_qty': tick.get('bid_qty'),
                'ask_qty': tick.get('ask_qty'),
                'open_interest': tick.get('open_interest')
            }
            
            # Notify subscribers
            callbacks = self._subscribers.get(symbol, set()).copy()
        
        # Call callbacks without holding the lock
        for callback in callbacks:
            try:
                callback(self._tick_store[symbol])
            except Exception as e:
                self.logger.error(f"Error in tick callback for {symbol}: {str(e)}", exc_info=True)
    
    def _on_ws_connected(self, data: Dict[str, Any]) -> None:
        """Handle WebSocket connected event."""
        self.logger.info("WebSocket connected, resubscribing to symbols...")
        
        # Resubscribe to all symbols
        with self._lock:
            symbols = list(self._subscribers.keys())
            if symbols:
                self.ws.subscribe(symbols)
    
    def _on_ws_disconnected(self, data: Dict[str, Any]) -> None:
        """Handle WebSocket disconnected event."""
        reason = data.get('reason', 'unknown')
        self.logger.warning(f"WebSocket disconnected: {reason}")
    
    def _on_error(self, error: Dict[str, Any]) -> None:
        """Handle WebSocket error event."""
        error_msg = error.get('message', 'Unknown error')
        error_type = error.get('type', 'unknown')
        self.logger.error(f"WebSocket error ({error_type}): {error_msg}")
    
    def _candle_aggregation_loop(self) -> None:
        """Background thread for aggregating ticks into candles."""
        while not self._stop_event.is_set():
            try:
                # Get current time and align to minute boundary
                now = datetime.utcnow()
                next_minute = (now.replace(second=0, microsecond=0) + 
                             timedelta(minutes=1))
                sleep_time = (next_minute - now).total_seconds()
                
                if sleep_time > 0:
                    self._stop_event.wait(sleep_time)
                
                # Aggregate ticks into candles
                self._aggregate_ticks_to_candles()
                
            except Exception as e:
                self.logger.error(f"Error in candle aggregation loop: {str(e)}", exc_info=True)
                time.sleep(1)  # Prevent tight loop on errors
    
    def _aggregate_ticks_to_candles(self) -> None:
        """Aggregate ticks into candles for all subscribed symbols and intervals."""
        with self._lock:
            current_time = datetime.utcnow()
            
            for symbol in list(self._subscribers.keys()):
                tick = self._tick_store.get(symbol)
                if not tick:
                    continue
                    
                for interval in self.candle_intervals:
                    self._update_candle(symbol, interval, tick, current_time)
    
    def _update_candle(self, symbol: str, interval: str, tick: Dict[str, Any], 
                      timestamp: datetime) -> None:
        """
        Update candle data for a symbol and interval based on tick data.
        
        Args:
            symbol: Symbol to update
            interval: Candle interval (e.g., '1m', '5m')
            tick: Latest tick data
            timestamp: Current timestamp
        """
        key = (symbol, interval)
        
        # Parse interval (e.g., '5m' -> 5 minutes)
        try:
            minutes = int(interval.rstrip('mh'))
            if 'h' in interval:
                minutes *= 60  # Convert hours to minutes
        except (ValueError, AttributeError):
            self.logger.warning(f"Invalid interval format: {interval}")
            return
        
        # Calculate candle start time
        if interval.endswith('m'):
            # Align to minute boundary
            timestamp = timestamp.replace(second=0, microsecond=0)
            minutes_past_hour = timestamp.minute % minutes
            candle_start = timestamp - timedelta(minutes=minutes_past_hour)
        else:
            # For hourly candles, align to hour boundary
            candle_start = timestamp.replace(minute=0, second=0, microsecond=0)
        
        # Get or create candle for this interval
        with self._lock:
            if key not in self._candle_store:
                self._candle_store[key] = []
            
            candles = self._candle_store[key]
            current_price = tick.get('last_price')
            
            if not candles or candles[-1]['timestamp'] < candle_start.timestamp() * 1000:
                # Create new candle
                new_candle = {
                    'symbol': symbol,
                    'interval': interval,
                    'timestamp': int(candle_start.timestamp() * 1000),
                    'open': current_price,
                    'high': current_price,
                    'low': current_price,
                    'close': current_price,
                    'volume': tick.get('volume', 0),
                    'open_interest': tick.get('open_interest', 0)
                }
                candles.append(new_candle)
                
                # Trim old candles
                if len(candles) > self.max_candles:
                    self._candle_store[key] = candles[-self.max_candles:]
                
                # Notify candle subscribers
                self._notify_candle_subscribers(symbol, interval, new_candle)
            else:
                # Update current candle
                current_candle = candles[-1]
                current_candle['high'] = max(current_candle['high'], current_price)
                current_candle['low'] = min(current_candle['low'], current_price)
                current_candle['close'] = current_price
                current_candle['volume'] += tick.get('volume', 0)
                current_candle['open_interest'] = tick.get('open_interest', current_candle.get('open_interest', 0))
                
                # Update subscribers with the latest candle
                self._notify_candle_subscribers(symbol, interval, current_candle)
    
    def _notify_candle_subscribers(self, symbol: str, interval: str, candle: Dict[str, Any]) -> None:
        """Notify all subscribers of a candle update."""
        key = (symbol.upper(), interval)
        with self._lock:
            callbacks = self._candle_subscribers.get(key, set()).copy()
        
        for callback in callbacks:
            try:
                callback(candle)
            except Exception as e:
                self.logger.error(f"Error in candle callback for {symbol} {interval}: {str(e)}", exc_info=True)
    
    def subscribe_candles(self, symbol: str, interval: str, callback: DataCallback) -> None:
        """
        Subscribe to candle updates for a symbol and interval.
        
        Args:
            symbol: Symbol to subscribe to
            interval: Candle interval (e.g., '1m', '5m')
            callback: Function to call when new candle data is available
        """
        symbol = symbol.upper()
        key = (symbol, interval)
        
        with self._lock:
            if key not in self._candle_subscribers:
                self._candle_subscribers[key] = set()
            
            self._candle_subscribers[key].add(callback)
            self.logger.debug(f"Subscribed to {symbol} {interval} candles")
            
            # Send the most recent candle if available
            candles = self._candle_store.get(key, [])
            if candles:
                try:
                    callback(candles[-1])
                except Exception as e:
                    self.logger.error(f"Error in initial candle callback: {str(e)}", exc_info=True)
    
    def unsubscribe_candles(self, symbol: str, interval: str, callback: DataCallback) -> None:
        """
        Unsubscribe from candle updates for a symbol and interval.
        
        Args:
            symbol: Symbol to unsubscribe from
            interval: Candle interval (e.g., '1m', '5m')
            callback: Callback function to remove
        """
        key = (symbol.upper(), interval)
        
        with self._lock:
            if key in self._candle_subscribers and callback in self._candle_subscribers[key]:
                self._candle_subscribers[key].remove(callback)
                self.logger.debug(f"Unsubscribed from {symbol} {interval} candles")
