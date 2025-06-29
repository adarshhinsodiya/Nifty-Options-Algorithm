"""
WebSocket handler for real-time market data streaming.

This module provides WebSocket-based real-time data streaming functionality
for the NIFTY Options Trading System, specifically for live trading mode.
"""
import json
import logging
import time
import threading
from queue import Queue, Empty
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime
import pytz

# Type alias for callback functions
DataCallback = Callable[[Dict[str, Any]], None]

class WebSocketHandler:
    """
    Handles WebSocket connections for real-time market data streaming.
    
    This class manages WebSocket connections to the ICICI Direct Breeze API
    for receiving real-time market data, including ticks and OHLC updates.
    """
    
    def __init__(self, breeze_client, logger: Optional[logging.Logger] = None):
        """
        Initialize the WebSocket handler.
        
        Args:
            breeze_client: Instance of BreezeConnect client
            logger: Logger instance (optional)
        """
        self.breeze = breeze_client
        self.logger = logger or logging.getLogger(__name__)
        self.symbol_callbacks: Dict[str, List[DataCallback]] = {}
        self.event_callbacks: Dict[str, List[DataCallback]] = {}
        self.subscribed_symbols: Set[str] = set()
        self.is_connected = False
        self._message_queue = Queue()
        self._processing_thread = None
        self._stop_event = threading.Event()
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5  # seconds
        self._lock = threading.Lock()
        
        # Register default callbacks
        self._register_default_callbacks()
        
    def register_callback(self, event_type: str, callback: DataCallback) -> None:
        """Register a callback for a specific event type.
        
        Args:
            event_type: Type of event to register for (e.g., 'tick', 'candle', 'error')
            callback: Callback function to register
        """
        with self._lock:
            if event_type not in self.event_callbacks:
                self.event_callbacks[event_type] = []
            if callback not in self.event_callbacks[event_type]:
                self.event_callbacks[event_type].append(callback)
                self.logger.debug(f"Registered callback for event type: {event_type}")
                
    def unregister_callback(self, event_type: str, callback: DataCallback) -> None:
        """Unregister a callback for a specific event type.
        
        Args:
            event_type: Type of event to unregister from
            callback: Callback function to unregister
        """
        with self._lock:
            if event_type in self.event_callbacks and callback in self.event_callbacks[event_type]:
                self.event_callbacks[event_type].remove(callback)
                self.logger.debug(f"Unregistered callback for event type: {event_type}")
                
    def _trigger_event(self, event_type: str, data: Any) -> None:
        """Trigger all callbacks for a specific event type.
        
        Args:
            event_type: Type of event to trigger
            data: Data to pass to the callbacks
        """
        with self._lock:
            callbacks = self.event_callbacks.get(event_type, []).copy()
            
        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback: {str(e)}", exc_info=True)
    
    def _register_default_callbacks(self) -> None:
        """Register default WebSocket event callbacks."""
        if hasattr(self.breeze, 'on_ticks'):
            self.breeze.on_ticks = self._on_ticks
        if hasattr(self.breeze, 'on_connect'):
            self.breeze.on_connect = self._on_connect
        if hasattr(self.breeze, 'on_error'):
            self.breeze.on_error = self._on_error
        if hasattr(self.breeze, 'on_close'):
            self.breeze.on_close = self._on_close
    
    def connect(self) -> bool:
        """
        Establish WebSocket connection.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        if self.is_connected:
            self.logger.warning("WebSocket already connected")
            return True
            
        try:
            self.logger.info("Connecting to WebSocket...")
            self.breeze.ws_connect()
            
            # Start message processing thread
            self._stop_event.clear()
            self._processing_thread = threading.Thread(
                target=self._process_messages,
                daemon=True,
                name="WebSocketMessageProcessor"
            )
            self._processing_thread.start()
            
            self.is_connected = True
            self._reconnect_attempts = 0
            self.logger.info("WebSocket connected successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket: {str(e)}")
            self._schedule_reconnect()
            return False
    
    def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if not self.is_connected:
            return
            
        self.logger.info("Disconnecting WebSocket...")
        self._stop_event.set()
        
        if hasattr(self.breeze, 'ws_disconnect'):
            try:
                self.breeze.ws_disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting WebSocket: {str(e)}")
        
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)
            
        self.is_connected = False
        self.logger.info("WebSocket disconnected")
    
    def subscribe(self, symbols: List[Union[str, Dict[str, Any]]], callback: Optional[DataCallback] = None) -> None:
        """
        Subscribe to real-time updates for the given symbols.
        
        Args:
            symbols: List of stock/option symbols or symbol configs to subscribe to
            callback: Optional function to call when data is received for these symbols
        """
        if not symbols:
            return
            
        # Convert symbols to a consistent format
        formatted_symbols = []
        for symbol in symbols:
            if isinstance(symbol, dict):
                # Already in config format
                stock_code = symbol.get('stock_code')
                if not stock_code:
                    continue
                formatted_symbols.append((stock_code, symbol))
            else:
                # Simple symbol string
                formatted_symbols.append((symbol, {'stock_code': symbol}))
        
        if not formatted_symbols:
            return
            
        # Add symbols to subscription list
        new_symbols = []
        with self._lock:
            for symbol, symbol_config in formatted_symbols:
                if symbol not in self.subscribed_symbols:
                    new_symbols.append(symbol_config)
                    self.subscribed_symbols.add(symbol)
                    
                # Register callback if provided
                if callback is not None:
                    if symbol not in self.symbol_callbacks:
                        self.symbol_callbacks[symbol] = []
                    if callback not in self.symbol_callbacks[symbol]:
                        self.symbol_callbacks[symbol].append(callback)
        
        if not new_symbols:
            return
        
        # Subscribe to symbols via WebSocket
        if self.is_connected:
            try:
                self.breeze.subscribe_feeds(
                    stock_token=list(new_symbols),
                    on_ticks=self._on_ticks,
                    on_connect=self._on_connect,
                    on_error=self._on_error
                )
                self.logger.info(f"Subscribed to {len(new_symbols)} symbols")
            except Exception as e:
                self.logger.error(f"Failed to subscribe to symbols: {str(e)}")
    
    def unsubscribe(self, symbols: List[str], callback: Optional[DataCallback] = None) -> None:
        """
        Unsubscribe from real-time updates for the given symbols.
        
        Args:
            symbols: List of stock/option symbols to unsubscribe from
            callback: Specific callback to remove (if None, removes all callbacks for the symbol)
        """
        if not symbols:
            return
            
        for symbol in symbols:
            if symbol not in self.callbacks:
                continue
                
            if callback is None:
                # Remove all callbacks for this symbol
                del self.callbacks[symbol]
                self.subscribed_symbols.discard(symbol)
            else:
                # Remove specific callback
                if callback in self.callbacks[symbol]:
                    self.callbacks[symbol].remove(callback)
                    if not self.callbacks[symbol]:
                        del self.callbacks[symbol]
                        self.subscribed_symbols.discard(symbol)
        
        # Unsubscribe from symbols via WebSocket if needed
        if self.is_connected and hasattr(self.breeze, 'unsubscribe'):
            try:
                self.breeze.unsubscribe(
                    stock_token=symbols,
                    on_ticks=self._on_ticks
                )
                self.logger.info(f"Unsubscribed from {len(symbols)} symbols")
            except Exception as e:
                self.logger.error(f"Failed to unsubscribe from symbols: {str(e)}")
    
    def _process_messages(self) -> None:
        """Process messages from the WebSocket in a separate thread."""
        while not self._stop_event.is_set():
            try:
                # Process all available messages with a small timeout
                while True:
                    try:
                        message = self._message_queue.get(timeout=0.1)
                        self._handle_message(message)
                    except Empty:
                        break
                        
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in WebSocket message processing: {str(e)}", exc_info=True)
                time.sleep(1)  # Prevent tight loop on errors
    
    def _handle_message(self, message: Dict[str, Any]) -> None:
        """
        Handle incoming WebSocket message.
        
        Args:
            message: The WebSocket message data
        """
        try:
            # Extract symbol from message
            symbol = message.get('symbol')
            if not symbol or symbol not in self.callbacks:
                return
            
            # Call all registered callbacks for this symbol
            for callback in self.callbacks.get(symbol, []):
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Error in WebSocket callback: {str(e)}", exc_info=True)
                    
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {str(e)}", exc_info=True)
    
    def _on_ticks(self, ticks: Dict[str, Any]) -> None:
        """Handle incoming tick data from WebSocket."""
        try:
            # Trigger tick event for general subscribers
            self._trigger_event('tick', ticks)
            
            # Forward to message queue for processing
            self._message_queue.put(ticks)
            
            # Extract symbol and forward to symbol-specific callbacks
            symbol = ticks.get('symbol')
            if symbol:
                with self._lock:
                    callbacks = self.symbol_callbacks.get(symbol, []).copy()
                
                for callback in callbacks:
                    try:
                        callback(ticks)
                    except Exception as e:
                        self.logger.error(f"Error in symbol callback for {symbol}: {str(e)}", exc_info=True)
                        
        except Exception as e:
            self.logger.error(f"Error processing tick: {str(e)}", exc_info=True)
    
    def _on_connect(self) -> None:
        """Handle WebSocket connection established event."""
        self.logger.info("WebSocket connection established")
        self.is_connected = True
        self._reconnect_attempts = 0
        
        # Resubscribe to symbols after reconnection
        if self.subscribed_symbols:
            self.logger.info(f"Resubscribing to {len(self.subscribed_symbols)} symbols...")
            self.subscribe(list(self.subscribed_symbols), None)
    
    def _on_error(self, error: Exception) -> None:
        """Handle WebSocket error event."""
        self.logger.error(f"WebSocket error: {str(error)}")
        self.is_connected = False
        self._schedule_reconnect()
    
    def _on_close(self) -> None:
        """Handle WebSocket connection closed event."""
        self.logger.warning("WebSocket connection closed")
        self.is_connected = False
        self._schedule_reconnect()
    
    def _schedule_reconnect(self) -> None:
        """Schedule a reconnection attempt."""
        if self._stop_event.is_set():
            return
            
        self._reconnect_attempts += 1
        
        if self._reconnect_attempts > self._max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached. Giving up.")
            return
        
        delay = min(self._reconnect_delay * (2 ** (self._reconnect_attempts - 1)), 300)  # Max 5 minutes
        self.logger.info(f"Scheduling reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} in {delay} seconds...")
        
        def _reconnect():
            time.sleep(delay)
            if not self._stop_event.is_set():
                self.connect()
        
        threading.Thread(target=_reconnect, daemon=True).start()
    
    def __del__(self):
        """Clean up resources on object deletion."""
        self.disconnect()
