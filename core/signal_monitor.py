"""
Signal monitoring and processing for real-time trading.
"""
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pytz
from pathlib import Path

from core.models import TradeSignal, TradeSignalType
from core.config_manager import ConfigManager

class SignalMonitor:
    """Monitors for and processes trading signals in real-time."""
    
    def __init__(self, config: ConfigManager, trade_executor):
        """Initialize the signal monitor.
        
        Args:
            config: Configuration manager instance
            trade_executor: Trade executor for executing signals
        """
        self.config = config
        self.trade_executor = trade_executor
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.check_interval = 60  # seconds
        self.signals_dir = Path("signals")
        self.signals_dir.mkdir(exist_ok=True)
    
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
            # In a real implementation, this would check a signal queue or API
            # For now, we'll just log that we're checking for signals
            self.logger.debug("Checking for new signals...")
            
            # Example: Get signals from strategy (to be implemented)
            # signals = self._get_signals_from_strategy()
            # for signal in signals:
            #     self._process_signal(signal)
                
        except Exception as e:
            self.logger.error(f"Error checking signals: {str(e)}", exc_info=True)
    
    def process_signal(self, signal: TradeSignal):
        """Process a single trading signal.
        
        Args:
            signal: TradeSignal to process
        """
        try:
            # Log the signal
            self._log_signal(signal)
            
            # Execute the signal
            self.trade_executor.execute_signal(signal)
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {str(e)}", exc_info=True)
    
    def _log_signal(self, signal: TradeSignal):
        """Log a trading signal to a JSON file.
        
        Args:
            signal: TradeSignal to log
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
                "metadata": signal.metadata or {}
            }
            
            # Generate filename with timestamp
            timestamp = datetime.now(self.ist_tz).strftime("%Y%m%d_%H%M%S")
            filename = f"signal_{timestamp}_{signal.symbol}_{signal.signal_type.value}.json"
            filepath = self.signals_dir / filename
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(signal_data, f, indent=2)
            
            # Log to console
            self.logger.info(f"Signal logged: {signal.signal_type} {signal.symbol} "
                          f"@ {signal.entry_price} (SL: {signal.stop_loss}, "
                          f"TP: {signal.target_price})")
            
        except Exception as e:
            self.logger.error(f"Error logging signal: {str(e)}", exc_info=True)
    
    def get_recent_signals(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent signals from the log.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of signal dictionaries
        """
        try:
            signals = []
            cutoff_time = datetime.now(self.ist_tz) - timedelta(hours=hours)
            
            for file in sorted(self.signals_dir.glob("signal_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
                try:
                    # Check file modification time
                    file_time = datetime.fromtimestamp(file.stat().st_mtime, tz=self.ist_tz)
                    if file_time < cutoff_time:
                        continue
                        
                    # Load signal data
                    with open(file, 'r') as f:
                        signal_data = json.load(f)
                    signals.append(signal_data)
                    
                except Exception as e:
                    self.logger.error(f"Error reading signal file {file}: {str(e)}")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error getting recent signals: {str(e)}", exc_info=True)
            return []
