"""
Test script for WebSocket integration and LiveDataManager.

This script tests the WebSocket connection, subscription management,
and data distribution functionality of the LiveDataManager.
"""
import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config
from core.broker.breeze import BreezeConnectWrapper
from core.websocket.websocket_handler import WebSocketHandler
from core.data.live_data_manager import LiveDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('websocket_test.log')
    ]
)
logger = logging.getLogger(__name__)

class WebSocketTester:
    """Test class for WebSocket and LiveDataManager functionality."""
    
    def __init__(self):
        """Initialize the tester with configuration and dependencies."""
        self.config = load_config()
        self.breeze = None
        self.ws_handler = None
        self.data_manager = None
        self.test_symbols = ['NIFTY']  # Only subscribing to NIFTY
        self.running = False
        
    def setup(self) -> bool:
        """Set up test environment and connections."""
        try:
            # Initialize Breeze API client
            self.breeze = BreezeConnectWrapper(
                api_key=self.config['breeze']['api_key'],
                api_secret=self.config['breeze']['api_secret']
            )
            
            # Get session token (assuming we have credentials in config)
            session_file = self.config.get('breeze', {}).get('session_file', 'breeze_session.json')
            if os.path.exists(session_file):
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    self.breeze.set_session_token(session_data['session_token'])
            else:
                logger.warning("No session file found. Please log in first.")
                return False
            
            # Initialize WebSocket handler
            self.ws_handler = WebSocketHandler(self.breeze, logger)
            
            # Initialize LiveDataManager
            self.data_manager = LiveDataManager(self.ws_handler, logger)
            
            # Register test callbacks
            self._register_callbacks()
            
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}", exc_info=True)
            return False
    
    def _register_callbacks(self) -> None:
        """Register test callbacks for tick and candle data."""
        if not self.data_manager:
            return
        
        # Register tick callback
        def on_tick(tick: Dict[str, Any]) -> None:
            logger.info(f"TICK: {tick['symbol']} - {tick['last_price']} @ {tick['timestamp']}")
        
        # Register candle callback
        def on_candle(candle: Dict[str, Any]) -> None:
            logger.info(
                f"CANDLE {candle['symbol']} {candle['interval']}: "
                f"O:{candle['open']} H:{candle['high']} L:{candle['low']} C:{candle['close']} "
                f"@ {datetime.fromtimestamp(candle['timestamp']/1000).isoformat()}"
            )
        
        # Subscribe to test symbols
        for symbol in self.test_symbols:
            self.data_manager.subscribe_ticks([symbol], on_tick)
            self.data_manager.subscribe_candles(symbol, '1m', on_candle)
    
    def run(self, duration: int = 60) -> None:
        """Run the WebSocket test for the specified duration."""
        if not self.data_manager:
            logger.error("Data manager not initialized. Call setup() first.")
            return
        
        try:
            logger.info("Starting WebSocket test...")
            self.running = True
            
            # Start data manager
            self.data_manager.start()
            
            # Subscribe to test symbols
            for symbol in self.test_symbols:
                logger.info(f"Subscribed to {symbol} data")
            
            # Run for specified duration
            logger.info(f"Running test for {duration} seconds...")
            start_time = time.time()
            
            while self.running and (time.time() - start_time) < duration:
                try:
                    # Print connection status periodically
                    if int(time.time() - start_time) % 10 == 0:
                        status = "connected" if self.ws_handler.is_connected else "disconnected"
                        logger.info(f"Status: {status} | Running time: {int(time.time() - start_time)}s")
                    
                    time.sleep(1)
                    
                except KeyboardInterrupt:
                    logger.info("Test interrupted by user")
                    self.running = False
                except Exception as e:
                    logger.error(f"Error in test loop: {str(e)}", exc_info=True)
                    time.sleep(1)
        
        except Exception as e:
            logger.error(f"Test failed: {str(e)}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources and connections."""
        self.running = False
        
        if self.data_manager:
            logger.info("Stopping LiveDataManager...")
            self.data_manager.stop()
        
        if self.ws_handler:
            logger.info("Disconnecting WebSocket...")
            self.ws_handler.disconnect()
        
        logger.info("Test completed")

if __name__ == "__main__":
    tester = WebSocketTester()
    if tester.setup():
        try:
            # Run test for 5 minutes by default
            tester.run(duration=300)
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            tester.cleanup()
    else:
        logger.error("Failed to set up WebSocket tester")
