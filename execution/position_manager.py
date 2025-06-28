"""
Position manager for monitoring and managing open positions, including SL/TP.
"""
import logging
import time
from datetime import datetime, timedelta
import json
import pytz
from typing import Dict, List, Optional, Tuple

from core.models import Position, PositionStatus, Order, OrderStatus, OrderType
from core.config_manager import ConfigManager

class PositionManager:
    """Manages open positions and monitors for SL/TP conditions."""
    
    def __init__(self, config: ConfigManager, data_provider, trade_executor):
        """Initialize the position manager.
        
        Args:
            config: Configuration manager instance
            data_provider: Data provider for market data
            trade_executor: Trade executor for closing positions
        """
        self.config = config
        self.data_provider = data_provider
        self.trade_executor = trade_executor
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.check_interval = 5  # seconds
        
        # SL/TP monitoring settings
        self.sl_buffer_pct = 0.001  # 0.1% buffer for SL/TP triggers
        self.check_after_minutes = 1  # Start checking after 1 minute of position open
    
    def start(self):
        """Start the position monitoring loop."""
        if self.running:
            self.logger.warning("Position manager is already running")
            return
            
        self.running = True
        self.logger.info("Starting position manager")
        
        while self.running:
            try:
                self._check_positions()
                time.sleep(self.check_interval)
            except KeyboardInterrupt:
                self.logger.info("Position manager stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in position manager: {str(e)}", exc_info=True)
                time.sleep(5)  # Prevent tight error loop
    
    def stop(self):
        """Stop the position monitoring loop."""
        self.running = False
        self.logger.info("Stopping position manager")
    
    def _check_positions(self):
        """Check all open positions for SL/TP conditions."""
        try:
            # Get all open positions
            open_positions = self.trade_executor.portfolio.get_open_positions()
            
            for position in open_positions:
                # Skip if position was just opened
                if (datetime.now(self.ist_tz) - position.entry_time).total_seconds() < 60:
                    continue
                    
                self._check_position(position)
                
        except Exception as e:
            self.logger.error(f"Error checking positions: {str(e)}", exc_info=True)
    
    def _check_position(self, position: Position):
        """Check a single position for SL/TP conditions.
        
        Args:
            position: Position to check
        """
        try:
            symbol = position.symbol
            current_price = self.data_provider.get_latest_price(symbol, 'NFO')
            
            if not current_price or current_price <= 0:
                self.logger.warning(f"Invalid price for {symbol}: {current_price}")
                return
            
            # Log position status
            self._log_position_status(position, current_price)
            
            # Check SL/TP conditions
            if position.stop_loss and current_price <= position.stop_loss * (1 - self.sl_buffer_pct):
                self._handle_sl_hit(position, current_price)
            elif position.target_price and current_price >= position.target_price * (1 + self.sl_buffer_pct):
                self._handle_tp_hit(position, current_price)
                
        except Exception as e:
            self.logger.error(f"Error checking position {position.position_id}: {str(e)}", exc_info=True)
    
    def _handle_sl_hit(self, position: Position, current_price: float):
        """Handle stop loss hit."""
        self.logger.warning(
            f"SL hit for {position.symbol} | "
            f"Entry: {position.entry_price} | "
            f"Current: {current_price} | "
            f"SL: {position.stop_loss}"
        )
        self._close_position(position, "SL")
    
    def _handle_tp_hit(self, position: Position, current_price: float):
        """Handle target price hit."""
        self.logger.info(
            f"TP hit for {position.symbol} | "
            f"Entry: {position.entry_price} | "
            f"Current: {current_price} | "
            f"TP: {position.target_price}"
        )
        self._close_position(position, "TP")
    
    def _close_position(self, position: Position, reason: str):
        """Close a position with the given reason."""
        try:
            self.logger.info(f"Closing position {position.position_id} ({reason})...")
            
            # Create exit signal
            from core.models import TradeSignal, TradeSignalType
            signal = TradeSignal(
                symbol=position.symbol,
                signal_type=TradeSignalType.SELL if position.quantity > 0 else TradeSignalType.BUY,
                entry_price=position.entry_price,
                stop_loss=position.stop_loss,
                target_price=position.target_price,
                timestamp=datetime.now(self.ist_tz),
                expiry_date=position.expiry_date,
                option_type=position.option_type,
                strike=position.strike,
                lot_size=abs(position.quantity),
                reason=reason
            )
            
            # Close the position
            self.trade_executor.execute_signal(signal)
            
        except Exception as e:
            self.logger.error(f"Error closing position {position.position_id}: {str(e)}", exc_info=True)
    
    def _log_position_status(self, position: Position, current_price: float):
        """Log position status in JSON format."""
        try:
            status = {
                "timestamp": datetime.now(self.ist_tz).isoformat(),
                "position_id": position.position_id,
                "symbol": position.symbol,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "current_price": current_price,
                "stop_loss": position.stop_loss,
                "target_price": position.target_price,
                "unrealized_pnl": position.calculate_pnl(current_price),
                "status": position.status.value,
                "age_minutes": (datetime.now(self.ist_tz) - position.entry_time).total_seconds() / 60
            }
            
            self.logger.info(f"POSITION_UPDATE: {json.dumps(status, indent=2, default=str)}")
            
        except Exception as e:
            self.logger.error(f"Error logging position status: {str(e)}", exc_info=True)
