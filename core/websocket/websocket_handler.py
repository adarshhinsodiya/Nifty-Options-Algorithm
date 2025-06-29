"""
WebSocket handler for real-time market data streaming.

This module provides WebSocket-based real-time data streaming functionality
for the NIFTY Options Trading System, specifically for live trading mode.
"""
import json
import logging
import random
import time
import threading
from queue import Queue, Empty
from typing import Dict, List, Optional, Callable, Any, Set, Union
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
        Initialize the WebSocket handler with reconnection and error handling.
        
        Args:
            breeze_client: Instance of BreezeConnect client
            logger: Logger instance (optional)
        """
        self.breeze = breeze_client
        self.logger = logger or logging.getLogger(__name__)
        self.symbol_callbacks: Dict[str, List[DataCallback]] = {}
        self.event_callbacks: Dict[str, List[DataCallback]] = {}
        self.subscribed_symbols: Set[str] = set()
        self.pending_subscriptions: List[Dict[str, Any]] = []
        self.is_connected = False
        self._message_queue = Queue()
        self._processing_thread = None
        self._stop_event = threading.Event()
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10  # Increased max reconnection attempts
        self._reconnect_delay = 5  # Initial delay in seconds
        self._max_reconnect_delay = 60  # Max 1 minute between reconnection attempts
        self._lock = threading.Lock()
        self._last_error = None
        self._last_activity = None
        self._connection_timeout = 30  # Consider connection dead after 30s of inactivity
        self._heartbeat_interval = 10  # Send heartbeat every 10s
        self._last_heartbeat = time.time()
        self._tick_count = 0  # Initialize tick counter
        self._shutdown_event = threading.Event()  # For graceful shutdown
        
        # Register default callbacks
        self._register_default_callbacks()
        
        # Start the connection monitor thread
        self._connection_monitor_thread = threading.Thread(
            target=self._connection_monitor,
            name="WebSocketConnectionMonitor",
            daemon=True
        )
        self._connection_monitor_thread.start()
        
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
        Establish WebSocket connection with retry logic.
        
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
            self._last_activity = time.time()
            self._last_heartbeat = time.time()
            self.logger.info("WebSocket connected successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect to WebSocket: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._last_error = error_msg
            self._attempt_reconnect()
            return False
    
    def disconnect(self) -> None:
        """Disconnect WebSocket connection gracefully."""
        if not self.is_connected:
            return
            
        self.logger.info("Disconnecting WebSocket...")
        self._stop_event.set()
        
        try:
            self.breeze.ws_disconnect()
            self.is_connected = False
            self.logger.info("WebSocket disconnected gracefully")
        except Exception as e:
            self.logger.error(f"Error disconnecting WebSocket: {str(e)}")
        finally:
            self._stop_event.clear()
            self._reconnect_attempts = 0
        
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)
            
        self.is_connected = False
        self.logger.info("WebSocket disconnected")
    
    def subscribe(self, symbols: List[Union[str, Dict[str, Any]]], callback: Optional[DataCallback] = None) -> None:
        """
        Subscribe to real-time updates for the given symbols with robust error handling.
        
        Args:
            symbols: List of stock/option symbols or symbol configs to subscribe to
            callback: Optional function to call when data is received for these symbols
            
        Raises:
            ValueError: If symbols parameter is invalid
            RuntimeError: If subscription fails after retries
        """
        if not symbols:
            self.logger.warning("No symbols provided for subscription")
            return
            
        if not isinstance(symbols, (list, tuple)):
            raise ValueError("Symbols must be a list or tuple")
            
        # Convert symbols to a consistent format and validate
        formatted_symbols = []
        for symbol in symbols:
            try:
                if isinstance(symbol, dict):
                    # Already in config format
                    stock_code = symbol.get('stock_code')
                    if not stock_code:
                        self.logger.warning(f"Skipping symbol with missing stock_code: {symbol}")
                        continue
                    formatted_symbols.append((stock_code, symbol))
                else:
                    # Simple symbol string
                    if not isinstance(symbol, str):
                        self.logger.warning(f"Skipping invalid symbol (not a string): {symbol}")
                        continue
                    formatted_symbols.append((symbol, {'stock_code': symbol}))
            except Exception as e:
                self.logger.error(f"Error processing symbol {symbol}: {str(e)}")
                continue
        
        if not formatted_symbols:
            self.logger.warning("No valid symbols to subscribe to")
            return
            
        # Add symbols to subscription list
        new_subscriptions = []
        with self._lock:
            for symbol, symbol_config in formatted_symbols:
                if symbol not in self.subscribed_symbols:
                    new_subscriptions.append(symbol_config)
                    self.subscribed_symbols.add(symbol)
                    self.logger.debug(f"Added {symbol} to subscription list")
                    
                # Register callback if provided
                if callback is not None:
                    if symbol not in self.symbol_callbacks:
                        self.symbol_callbacks[symbol] = []
                    if callback not in self.symbol_callbacks[symbol]:
                        self.symbol_callbacks[symbol].append(callback)
                        self.logger.debug(f"Registered callback for {symbol}")
        
        if not new_subscriptions:
            self.logger.debug("No new symbols to subscribe to")
            return
            
        # Store pending subscriptions for reconnection
        self.pending_subscriptions.extend(new_subscriptions)
        
        # Subscribe to symbols via WebSocket if connected
        if self.is_connected:
            self._do_subscribe(new_subscriptions)
    
    def _do_subscribe(self, subscriptions: List[Dict[str, Any]]) -> None:
        """
        Perform the actual subscription with retry logic.
        
        Args:
            subscriptions: List of subscription dictionaries
        """
        if not subscriptions:
            return
            
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Subscribing to {len(subscriptions)} symbols (attempt {attempt + 1}/{max_retries})")
                self.breeze.subscribe_feeds(subscriptions)
                self.logger.info(f"Successfully subscribed to {len(subscriptions)} symbols")
                return  # Success
                
            except Exception as e:
                error_msg = f"Failed to subscribe to symbols (attempt {attempt + 1}/{max_retries}): {str(e)}"
                self.logger.error(error_msg)
                
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    # If all retries fail, raise the last exception
                    raise RuntimeError(f"Failed to subscribe after {max_retries} attempts: {str(e)}")
    
    def _safe_callback(self, callback: Callable, data: Any, symbol: str) -> None:
        """
        Safely execute a callback with error handling.
        
        Args:
            callback: The callback function to execute
            data: The data to pass to the callback
            symbol: The symbol this callback is for (for logging)
        """
        try:
            callback(data)
        except Exception as e:
            self.logger.error(f"Error in callback for {symbol}: {str(e)}", exc_info=True)
            self._trigger_event('callback_error', {
                'symbol': symbol,
                'error': str(e),
                'type': type(e).__name__,
                'timestamp': datetime.now().isoformat()
            })
    
    def _check_connection_health(self) -> None:
        """Check the health of the WebSocket connection and attempt recovery if needed."""
        if not self.is_connected:
            self.logger.warning("Connection health check: Not connected")
            self._attempt_reconnect()
            return
            
        current_time = time.time()
        
        # Check for heartbeat timeout
        if current_time - self._last_heartbeat > self._heartbeat_interval * 2:
            self.logger.warning("Heartbeat timeout, reconnecting...")
            self.disconnect()
            self._attempt_reconnect()
            return
            
        # Check for general inactivity
        if current_time - self._last_activity > self._connection_timeout:
            self.logger.warning("Connection inactive for too long, reconnecting...")
            self.disconnect()
            self._attempt_reconnect()
            return
    
    def _send_heartbeat(self) -> None:
        """Send a heartbeat message to keep the connection alive."""
        try:
            self._last_heartbeat = time.time()
            # Some WebSocket APIs support explicit heartbeats
            if hasattr(self.breeze, 'ws_heartbeat'):
                self.breeze.ws_heartbeat()
        except Exception as e:
            self.logger.warning(f"Error sending heartbeat: {str(e)}")
    
    def _handle_heartbeat(self) -> None:
        """Handle incoming heartbeat message."""
        self._last_heartbeat = time.time()
        self._last_activity = time.time()
        self.logger.debug("Heartbeat received")
    
    def _connection_monitor(self) -> None:
        """Background thread to monitor connection health and attempt recovery."""
        self.logger.info("Starting WebSocket connection monitor")
        
        while not self._stop_event.is_set():
            try:
                # Check connection health periodically
                self._check_connection_health()
                
                # Send heartbeat if needed
                current_time = time.time()
                if self.is_connected and current_time - self._last_heartbeat > self._heartbeat_interval:
                    self._send_heartbeat()
                
                # Small sleep to prevent high CPU usage
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in connection monitor: {str(e)}", exc_info=True)
                time.sleep(5)  # Avoid tight loop on errors
        
        self.logger.info("WebSocket connection monitor stopped")
        
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
        Handle incoming WebSocket message and dispatch to callbacks.
        
        Args:
            message: The WebSocket message data
        """
        try:
            if 'tick' in message:
                self._on_ticks(message['tick'])
            elif 'connected' in message:
                self._on_connect()
            elif 'disconnected' in message:
                self._on_disconnect()
            elif 'error' in message:
                self._on_error(message['error'])
            elif 'heartbeat' in message:
                self._handle_heartbeat()
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}", exc_info=True)
            self._trigger_event('error', {
                'error': 'message_processing_error',
                'message': str(e),
                'original_message': str(message)[:500]  # Truncate if too large
            })
    
    def _on_ticks(self, ticks: Dict[str, Any]) -> None:
        """Handle incoming tick data from WebSocket with validation and error handling."""
        try:
            self._last_activity = time.time()
            
            # Validate tick data
            if not ticks or 'symbol' not in ticks:
                self.logger.warning("Received invalid tick data (missing symbol)")
                return
                
            symbol = ticks.get('symbol')
            
            # Log tick for debugging (but not too frequently to avoid log spam)
            if self._tick_count % 100 == 0:  # Log every 100th tick
                self.logger.debug(f"Tick received for {symbol}: {ticks.get('last_price', 'N/A')}")
            
            # Update last activity timestamp
            self._last_activity = time.time()
            
            # Trigger tick event for general subscribers
            self._trigger_event('tick', ticks)
            
            # Forward to message queue for processing
            try:
                self._message_queue.put(ticks, timeout=1.0)
            except queue.Full:
                self.logger.warning("Message queue full, dropping tick")
            
            # Forward to symbol-specific callbacks
            with self._lock:
                callbacks = self.symbol_callbacks.get(symbol, []).copy()
            
            # Process callbacks in separate threads to avoid blocking
            if callbacks:
                for callback in callbacks:
                    try:
                        # Use thread pool for parallel callback execution
                        threading.Thread(
                            target=self._safe_callback,
                            args=(callback, ticks, symbol),
                            daemon=True
                        ).start()
                    except Exception as e:
                        self.logger.error(f"Error dispatching callback for {symbol}: {str(e)}", exc_info=True)
            
            # Periodically check connection health
            self._tick_count += 1
            if self._tick_count % 1000 == 0:  # Every 1000 ticks
                self._check_connection_health()
                        
        except Exception as e:
            self.logger.error(f"Critical error processing tick: {str(e)}", exc_info=True)
            self._trigger_event('error', {
                'error': 'tick_processing_error',
                'message': str(e),
                'tick': ticks,
                'timestamp': datetime.now().isoformat()
            })
    
    def _on_connect(self) -> None:
        """Handle WebSocket connection established event."""
        self.is_connected = True
        self._reconnect_attempts = 0
        self._last_activity = time.time()
        self._last_heartbeat = time.time()
        self.logger.info("WebSocket connection established")
        
        # Trigger connected event
        self._trigger_event('connected', {})
        
        # Resubscribe to symbols on reconnect
        self._resubscribe()
    
    def _resubscribe(self) -> None:
        """Resubscribe to all pending subscriptions."""
        if not self.pending_subscriptions:
            return
            
        self.logger.info(f"Resubscribing to {len(self.pending_subscriptions)} symbols")
        
        # Group subscriptions by batch to avoid overwhelming the WebSocket
        batch_size = 10  # Adjust based on broker limits
        for i in range(0, len(self.pending_subscriptions), batch_size):
            batch = self.pending_subscriptions[i:i + batch_size]
            try:
                self.breeze.subscribe_feeds(batch)
                self.logger.debug(f"Resubscribed to batch {i//batch_size + 1}")
                time.sleep(0.5)  # Small delay between batches
            except Exception as e:
                self.logger.error(f"Error resubscribing batch {i//batch_size + 1}: {str(e)}")
    
    def _on_error(self, error: Exception) -> None:
        """Handle WebSocket error event."""
        self.logger.error(f"WebSocket error: {str(error)}")
        self.is_connected = False
        self._schedule_reconnect()
    
    def _on_disconnect(self) -> None:
        """Handle WebSocket disconnection with reconnection logic."""
        if self.is_connected:  # Only log if we were actually connected
            self.logger.warning("WebSocket disconnected")
            self.is_connected = False
            
            # Trigger disconnected event
            self._trigger_event('disconnected', {'reason': 'connection_lost'})
            
            # Only attempt reconnect if not explicitly stopped
            if not self._stop_event.is_set():
                self._attempt_reconnect()
    
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
    
    def _attempt_reconnect(self) -> None:
        """Attempt to reconnect to WebSocket with exponential backoff and jitter."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached. Manual intervention required.")
            self._trigger_event('error', {
                'error': 'max_reconnection_attempts_reached',
                'message': 'Maximum number of reconnection attempts reached',
                'timestamp': datetime.now().isoformat()
            })
            return
            
        self._reconnect_attempts += 1
        
        # Exponential backoff with jitter
        base_delay = min(self._reconnect_delay * (2 ** (self._reconnect_attempts - 1)), self._max_reconnect_delay)
        jitter = random.uniform(0, 1) * 2 - 1  # Random value between -1 and 1
        delay = max(1, base_delay + jitter)  # Ensure at least 1 second
        
        self.logger.info(f"Attempting to reconnect in {delay:.1f} seconds (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})")
        
        def reconnect():
            time.sleep(delay)
            if not self.is_connected and not self._stop_event.is_set():
                try:
                    self.connect()
                except Exception as e:
                    self.logger.error(f"Error during reconnection attempt: {str(e)}")
                    self._attempt_reconnect()  # Continue reconnection attempts
        
        # Start reconnection in a separate thread
        threading.Thread(target=reconnect, daemon=True, name=f"WebSocketReconnect-{self._reconnect_attempts}").start()
    
    def close(self) -> None:
        """Clean up resources and stop all threads."""
        self.logger.info("Closing WebSocket handler...")
        self._shutdown_event.set()
        self.disconnect()
        
        # Wait for threads to complete
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)
            
        if self._connection_monitor_thread and self._connection_monitor_thread.is_alive():
            self._connection_monitor_thread.join(timeout=5.0)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure resources are cleaned up."""
        self.close()
    
    def __del__(self):
        """Fallback cleanup on garbage collection."""
        if not self._shutdown_event.is_set():
            self.close()
