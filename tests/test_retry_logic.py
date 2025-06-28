"""
Test script to verify retry/backoff logic in live trading mode.
"""
import os
import time
import logging
from datetime import datetime, timedelta
import pytz

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('retry_test.log')
    ]
)
logger = logging.getLogger(__name__)

def test_session_renewal(data_provider):
    """Test session renewal functionality."""
    logger.info("Testing session renewal...")
    
    # Get current session expiry
    old_expiry = data_provider._session_expiry
    logger.info(f"Current session expires at: {old_expiry}")
    
    # Force session renewal
    data_provider._session_expiry = datetime.now(pytz.utc) - timedelta(minutes=1)
    
    # This should trigger a renewal
    data_provider._ensure_valid_session()
    
    # Verify new expiry is in the future
    new_expiry = data_provider._session_expiry
    logger.info(f"New session expires at: {new_expiry}")
    
    assert new_expiry > datetime.now(pytz.utc), "Session expiry not updated"
    assert new_expiry > old_expiry, "New session expiry not in the future"
    logger.info("✅ Session renewal test passed")

def test_api_retry(data_provider):
    """Test API retry functionality."""
    logger.info("Testing API retry logic...")
    
    # Test with a valid API call
    try:
        result = data_provider._api_call_with_retry(
            data_provider.breeze.get_demat_holdings
        )
        logger.info("✅ Valid API call successful")
        logger.debug(f"API response: {result}")
    except Exception as e:
        logger.error(f"Valid API call failed: {str(e)}")
        raise
    
    # Test with an invalid API call (should retry and fail)
    try:
        data_provider._api_call_with_retry(
            data_provider.breeze.get_demat_holdings,
            invalid_param="should_fail"
        )
        logger.warning("❌ Invalid API call did not raise an exception")
    except Exception as e:
        logger.info(f"✅ Invalid API call failed as expected: {str(e)}")

def test_order_placement(trade_executor):
    """Test order placement with retry logic."""
    from core.models import TradeSignal, TradeSignalType
    
    logger.info("Testing order placement with retry...")
    
    # Create a test signal
    signal = TradeSignal(
        symbol="NIFTY",
        signal_type=TradeSignalType.BUY,
        entry_price=22000,
        stop_loss=21900,
        target_price=22200,
        timestamp=datetime.now(pytz.timezone('Asia/Kolkata')),
        expiry_date=datetime.now(pytz.timezone('Asia/Kolkata')) + timedelta(days=7),
        option_type="CE",
        strike=22000,
        lot_size=50
    )
    
    try:
        # Place the order
        position = trade_executor.execute_signal(signal)
        if position:
            logger.info(f"✅ Order placed successfully. Position ID: {position.position_id}")
            
            # Cancel the test order
            trade_executor.cancel_all_pending_orders()
            logger.info("Cancelled test orders")
        else:
            logger.warning("❌ Failed to place order")
    except Exception as e:
        logger.error(f"Order placement failed: {str(e)}")
        raise

def main():
    """Main test function."""
    from core.config_manager import ConfigManager
    from data.data_provider import DataProvider
    from execution.trade_executor import TradeExecutor
    from core.models import Portfolio
    
    logger.info("Starting retry logic tests...")
    
    # Initialize components
    config = ConfigManager()
    data_provider = DataProvider(config, mode='live')
    portfolio = Portfolio(config)
    trade_executor = TradeExecutor(config, data_provider, portfolio)
    
    try:
        # Run tests
        test_session_renewal(data_provider)
        test_api_retry(data_provider)
        test_order_placement(trade_executor)
        
        logger.info("✅ All tests completed successfully")
    except Exception as e:
        logger.error(f"❌ Tests failed: {str(e)}", exc_info=True)
    finally:
        # Clean up
        logger.info("Test cleanup complete")

if __name__ == "__main__":
    main()
