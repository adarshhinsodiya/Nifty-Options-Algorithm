"""
Signal monitoring and processing for real-time trading.

This module handles the generation and processing of trading signals
in both live and backtest modes, with robust error handling and retry logic.
"""
import logging
import time
import json
import queue
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
import pytz
from pathlib import Path

from core.models import TradeSignal, TradeSignalType
from core.config_manager import ConfigManager
from core.exceptions import MaxRetriesExceededError

class SignalMonitor:
    """Monitors for and processes trading signals in real-time."""
    
    def __init__(self, config: ConfigManager, trade_executor, data_provider=None):
        """Initialize the signal monitor.
        
        Args:
            config: Configuration manager instance
            trade_executor: Trade executor for executing signals
            data_provider: Optional data provider for real-time data
        """
        self.config = config
        self.trade_executor = trade_executor
        self.data_provider = data_provider
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.check_interval = 60  # seconds
        self.signals_dir = Path("signals")
        self.signals_dir.mkdir(exist_ok=True)
        
        # Signal queue for thread-safe processing
        self.signal_queue = queue.Queue()
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        # Last signal times to prevent duplicate processing
        self.last_signal_times: Dict[str, datetime] = {}
        self.signal_cooldown = timedelta(minutes=1)  # Minimum time between signals for same symbol
    
    def start(self):
        """Start the signal monitoring loop."""
        if self.running:
            self.logger.warning("Signal monitor is already running")
            return
            
        self.running = True
        self.logger.info("Starting signal monitor")
        
        while self.running:
            try:
                self._check_signals()
                time.sleep(self.check_interval)
            except KeyboardInterrupt:
                self.logger.info("Signal monitor stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in signal monitor: {str(e)}", exc_info=True)
                time.sleep(5)  # Prevent tight error loop
    
    def stop(self):
        """Stop the signal monitoring loop."""
        self.running = False
        self.logger.info("Stopping signal monitor")
    
    def _check_signals(self):
        """Check for new trading signals and process them."""
        try:
            self.logger.debug("Checking for new signals...")
            
            # Process any signals in the queue first
            self._process_queued_signals()
            
            # Get new signals from strategy if data provider is available
            if self.data_provider and hasattr(self.trade_executor.strategy, 'generate_signals'):
                try:
                    # Get latest market data
                    symbol = self.config.get('trading', 'symbol', fallback='NIFTY')
                    interval = self.config.get('trading', 'interval', fallback='1m')
                    
                    # Get historical data for signal generation
                    df = self.data_provider.get_historical_data(
                        symbol=symbol,
                        interval=interval,
                        duration='1d'  # Last day of data
                    )
                    
                    if df is not None and not df.empty:
                        # Generate signals from strategy
                        signals = self.trade_executor.strategy.generate_signals(df)
                        
                        # Process each signal
                        for signal in signals:
                            if signal and self._should_process_signal(signal):
                                self.process_signal(signal)
                                
                except Exception as e:
                    self.logger.error(f"Error generating signals: {str(e)}", exc_info=True)
                    
        except Exception as e:
            self.logger.error(f"Error in signal check loop: {str(e)}", exc_info=True)
    
    def _should_process_signal(self, signal: TradeSignal) -> bool:
        """Determine if a signal should be processed."""
        # Check cooldown period
        last_signal_time = self.last_signal_times.get(signal.symbol)
        if last_signal_time and (datetime.now(self.ist_tz) - last_signal_time) < self.signal_cooldown:
            self.logger.debug(f"Skipping signal for {signal.symbol} - in cooldown period")
            return False
            
        # Update last signal time
        self.last_signal_times[signal.symbol] = datetime.now(self.ist_tz)
        return True
    
    def _process_queued_signals(self):
        """Process any signals in the queue."""
        try:
            while True:
                try:
                    # Non-blocking get from queue
                    signal = self.signal_queue.get_nowait()
                    if signal and self._should_process_signal(signal):
                        self.process_signal(signal)
                except queue.Empty:
                    break
        except Exception as e:
            self.logger.error(f"Error processing queued signals: {str(e)}", exc_info=True)
    
    def queue_signal(self, signal: TradeSignal) -> bool:
        """Add a signal to the processing queue.
        
        Args:
            signal: TradeSignal to add to the queue
            
        Returns:
            bool: True if signal was queued, False otherwise
        """
        try:
            self.signal_queue.put(signal)
            return True
        except Exception as e:
            self.logger.error(f"Failed to queue signal: {str(e)}", exc_info=True)
            return False
    
    def process_signal(self, signal: TradeSignal, max_retries: int = None) -> bool:
        """Process a single trading signal with retry logic.
        
        Args:
            signal: TradeSignal to process
            max_retries: Maximum number of retry attempts (default: self.max_retries)
            
        Returns:
            bool: True if signal was processed successfully, False otherwise
            
        Raises:
            MaxRetriesExceededError: If max retries are exceeded
        """
        if max_retries is None:
            max_retries = self.max_retries
            
        attempt = 0
        last_exception = None
        
        while attempt <= max_retries:
            try:
                # Log the signal
                self._log_signal(signal)
                
                # Execute the signal
                result = self.trade_executor.execute_signal(signal)
                
                if result:
                    self.logger.info(f"Successfully processed signal: {signal}")
                    return True
                else:
                    self.logger.warning(f"Signal execution returned False: {signal}")
                    
            except Exception as e:
                last_exception = e
                self.logger.error(
                    f"Error processing signal (attempt {attempt + 1}/{max_retries + 1}): {str(e)}",
                    exc_info=attempt == max_retries  # Only log full traceback on final attempt
                )
                
                # Exponential backoff
                if attempt < max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                
            attempt += 1
        
        # If we get here, all retries failed
        error_msg = f"Failed to process signal after {max_retries + 1} attempts: {signal}"
        if last_exception:
            error_msg += f" (Last error: {str(last_exception)})"
            
        self.logger.error(error_msg)
        raise MaxRetriesExceededError(error_msg) from last_exception
    
    def _log_signal(self, signal: TradeSignal):
        """Log a trading signal to a JSON file and console.
        
        Args:
            signal: TradeSignal to log
            
        Returns:
            dict: The logged signal data if successful, None otherwise
        """
        try:
            # Create signal data dictionary
            signal_data = {
                "timestamp": datetime.now(self.ist_tz).isoformat(),
                "symbol": signal.symbol,
                "signal_type": signal.signal_type.value,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "target_price": signal.target_price,
                "quantity": signal.quantity,
                "expiry_date": signal.expiry_date.isoformat() if signal.expiry_date else None,
                "option_type": signal.option_type,
                "strike": signal.strike,
                "lot_size": signal.lot_size,
                "reason": signal.reason,
                "metadata": signal.metadata or {},
                "status": "pending"
            }
            
            # Generate filename with timestamp and unique ID
            timestamp = datetime.now(self.ist_tz).strftime("%Y%m%d_%H%M%S")
            signal_id = signal.signal_id if hasattr(signal, 'signal_id') else id(signal)
            filename = f"signal_{timestamp}_{signal.symbol}_{signal.signal_type.value}_{signal_id}.json"
            filepath = self.signals_dir / filename
            
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file with atomic write
            temp_path = filepath.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(signal_data, f, indent=2, default=str)
            temp_path.replace(filepath)  # Atomic rename
            
            # Log to console
            self.logger.info(
                f"Signal {signal_id}: {signal.signal_type} {signal.symbol} "
                f"@ {signal.entry_price} (SL: {signal.stop_loss}, TP: {signal.target_price}) "
                f"- {signal.reason or 'No reason provided'}"
            )
            
            return signal_data
            
        except Exception as e:
            self.logger.error(f"Error logging signal: {str(e)}", exc_info=True)
            return None
    
    def get_recent_signals(self, hours: int = 24, symbol: str = None, 
                         signal_type: str = None) -> List[Dict[str, Any]]:
        """Get recent signals from the log with filtering options.
        
        Args:
            hours: Number of hours to look back (default: 24)
            symbol: Optional symbol to filter by
            signal_type: Optional signal type to filter by (e.g., 'BUY', 'SELL')
            
        Returns:
            List of signal dictionaries, most recent first
        """
        try:
            signals = []
            cutoff_time = datetime.now(self.ist_tz) - timedelta(hours=hours)
            
            # Get all signal files, sorted by modification time (newest first)
            for file in sorted(self.signals_dir.glob("signal_*.json"), 
                             key=lambda x: x.stat().st_mtime, 
                             reverse=True):
                try:
                    # Check file modification time
                    file_time = datetime.fromtimestamp(file.stat().st_mtime, tz=self.ist_tz)
                    if file_time < cutoff_time:
                        continue
                        
                    # Load signal data
                    with open(file, 'r') as f:
                        signal_data = json.load(f)
                    
                    # Apply filters
                    if symbol and signal_data.get('symbol') != symbol:
                        continue
                        
                    if signal_type and signal_data.get('signal_type') != signal_type:
                        continue
                    
                    signals.append(signal_data)
                    
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON in signal file: {file}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error reading signal file {file}: {str(e)}")
                    continue
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error getting recent signals: {str(e)}", exc_info=True)
            return []
    
    def get_signal_status(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific signal by ID.
        
        Args:
            signal_id: The ID of the signal to look up
            
        Returns:
            Dictionary with signal status, or None if not found
        """
        try:
            # Look for signal file with matching ID
            for file in self.signals_dir.glob(f"*_{signal_id}.json"):
                try:
                    with open(file, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    continue
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting signal status for {signal_id}: {str(e)}")
            return None
