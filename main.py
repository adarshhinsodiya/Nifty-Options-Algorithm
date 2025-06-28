"""
Nifty Options Trading System

This script serves as the main entry point for the trading system.
It can be run in either live trading or backtesting mode.
"""
import argparse
import logging
import sys
import os
from pathlib import Path

from core.orchestrator import TradingOrchestrator

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Nifty Options Trading System')
    
    # Add arguments
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['live', 'backtest'], 
        default='backtest',
        help='Operation mode: live trading or backtesting (default: backtest)'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.ini',
        help='Path to configuration file (default: config/config.ini)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for backtesting (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for backtesting (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        help='Initial capital for backtesting'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()

def setup_logging(log_level: str = 'INFO') -> None:
    """Set up logging configuration."""
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'trading_system.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main entry point for the trading system."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting Nifty Options Trading System in {args.mode.upper()} mode")
        
        # Initialize and run the trading orchestrator
        orchestrator = TradingOrchestrator(
            config_path=args.config,
            mode=args.mode
        )
        
        # Override config with command line arguments if provided
        if args.mode == 'backtest':
            if args.start_date:
                orchestrator.config.config.set('backtest', 'start_date', args.start_date)
            if args.end_date:
                orchestrator.config.config.set('backtest', 'end_date', args.end_date)
            if args.initial_capital:
                orchestrator.config.config.set('backtest', 'initial_capital', str(args.initial_capital))
        
        # Run the trading system
        orchestrator.run()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
        if 'orchestrator' in locals():
            orchestrator.shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logging.shutdown()

if __name__ == "__main__":
    main()
