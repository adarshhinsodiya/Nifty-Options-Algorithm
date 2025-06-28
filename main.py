"""
Nifty Options Trading System

This script serves as the main entry point for the trading system.
It can be run in either live trading or backtesting mode.
"""
import argparse
import logging
import sys
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.orchestrator import TradingOrchestrator
from execution.position_manager import PositionManager
from core.signal_monitor import SignalMonitor
from core.reporting import PortfolioReporter

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
    
    # Backtesting arguments
    backtest_group = parser.add_argument_group('Backtesting Options')
    backtest_group.add_argument(
        '--start-date',
        type=str,
        help='Start date for backtesting (YYYY-MM-DD)'
    )
    
    backtest_group.add_argument(
        '--end-date',
        type=str,
        help='End date for backtesting (YYYY-MM-DD)'
    )
    
    backtest_group.add_argument(
        '--initial-capital',
        type=float,
        help='Initial capital for backtesting'
    )
    
    # Live trading arguments
    live_group = parser.add_argument_group('Live Trading Options')
    live_group.add_argument(
        '--check-interval',
        type=int,
        default=60,
        help='Interval in seconds between market data checks (default: 60)'
    )
    
    # General options
    general_group = parser.add_argument_group('General Options')
    general_group.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    general_group.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate a performance report and exit (for backtest mode)'
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

class TradingSystem:
    """Main trading system class that manages all components."""
    
    def __init__(self, args):
        """Initialize the trading system."""
        self.args = args
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.orchestrator = None
        self.position_manager = None
        self.signal_monitor = None
        self.portfolio_reporter = None
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
    
    def initialize(self):
        """Initialize all components."""
        try:
            self.logger.info(f"Initializing trading system in {self.args.mode.upper()} mode")
            
            # Initialize orchestrator
            self.orchestrator = TradingOrchestrator(
                config_path=self.args.config,
                mode=self.args.mode
            )
            
            # Override config with command line arguments if provided
            if self.args.mode == 'backtest':
                if self.args.start_date:
                    self.orchestrator.config.config.set('backtest', 'start_date', self.args.start_date)
                if self.args.end_date:
                    self.orchestrator.config.config.set('backtest', 'end_date', self.args.end_date)
                if self.args.initial_capital:
                    self.orchestrator.config.config.set('backtest', 'initial_capital', str(self.args.initial_capital))
            
            # Initialize additional components for live trading
            if self.args.mode == 'live':
                # Initialize position manager
                self.position_manager = PositionManager(
                    config=self.orchestrator.config,
                    data_provider=self.orchestrator.data_provider,
                    trade_executor=self.orchestrator.trade_executor
                )
                
                # Initialize signal monitor
                self.signal_monitor = SignalMonitor(
                    config=self.orchestrator.config,
                    trade_executor=self.orchestrator.trade_executor
                )
                
                # Initialize portfolio reporter
                self.portfolio_reporter = PortfolioReporter(
                    config=self.orchestrator.config,
                    portfolio=self.orchestrator.portfolio
                )
                
                # Register signal monitor with orchestrator
                self.orchestrator.signal_monitor = self.signal_monitor
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading system: {str(e)}", exc_info=True)
            return False
    
    def run(self):
        """Run the trading system."""
        if not self.initialize():
            self.logger.error("Failed to initialize trading system")
            return False
        
        self.running = True
        
        try:
            if self.args.mode == 'backtest':
                self._run_backtest()
            else:  # live mode
                self._run_live()
                
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested...")
        except Exception as e:
            self.logger.error(f"Fatal error: {str(e)}", exc_info=True)
            return False
        finally:
            self.shutdown()
        
        return True
    
    def _run_backtest(self):
        """Run backtesting mode."""
        self.logger.info("Starting backtest...")
        
        # Run the backtest
        self.orchestrator.run()
        
        # Generate report if requested
        if self.args.generate_report:
            self._generate_report()
    
    def _run_live(self):
        """Run live trading mode."""
        self.logger.info("Starting live trading...")
        
        # Start position manager in a separate thread
        import threading
        position_thread = threading.Thread(target=self.position_manager.start, daemon=True)
        position_thread.start()
        
        # Start signal monitor in a separate thread
        signal_thread = threading.Thread(target=self.signal_monitor.start, daemon=True)
        signal_thread.start()
        
        # Main trading loop
        while self.running:
            try:
                # Generate and process signals
                signals = self.orchestrator.generate_signals()
                for signal in signals:
                    self.signal_monitor.process_signal(signal)
                
                # Log portfolio status periodically
                self._log_portfolio_status()
                
                # Sleep until next check
                time.sleep(self.args.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in main trading loop: {str(e)}", exc_info=True)
                time.sleep(5)  # Prevent tight error loop
    
    def _log_portfolio_status(self):
        """Log portfolio status."""
        if not self.portfolio_reporter:
            return
            
        try:
            # Log open positions summary
            open_positions = self.portfolio_reporter.get_open_positions_summary()
            if open_positions and 'positions' in open_positions:
                self.logger.info(f"Open positions: {len(open_positions['positions'])}")
                for pos in open_positions['positions']:
                    self.logger.info(
                        f"  {pos['symbol']}: {pos['quantity']} @ {pos['entry_price']} "
                        f"(P&L: {pos.get('unrealized_pnl', 0):.2f})"
                    )
            
            # Log daily P&L
            daily_report = self.portfolio_reporter.generate_daily_report()
            if daily_report and 'total_pnl' in daily_report:
                self.logger.info(
                    f"Daily P&L: {daily_report['total_pnl']:.2f} "
                    f"(Trades: {daily_report.get('total_trades', 0)}, "
                    f"Win Rate: {daily_report.get('win_rate', 0):.1f}%)"
                )
                
        except Exception as e:
            self.logger.error(f"Error logging portfolio status: {str(e)}", exc_info=True)
    
    def _generate_report(self):
        """Generate performance report."""
        if not self.portfolio_reporter or self.args.mode != 'backtest':
            return
            
        try:
            self.logger.info("Generating performance report...")
            report = self.portfolio_reporter.generate_performance_report(days=30)
            
            if report and 'total_trades' in report and report['total_trades'] > 0:
                self.logger.info("\n" + "="*50)
                self.logger.info("PERFORMANCE REPORT")
                self.logger.info("="*50)
                self.logger.info(f"Period: {report.get('start_date')} to {report.get('end_date')}")
                self.logger.info(f"Total Trades: {report.get('total_trades')}")
                self.logger.info(f"Winning Trades: {report.get('winning_trades')} ({report.get('win_rate', 0):.1f}%)")
                self.logger.info(f"Total P&L: {report.get('total_pnl', 0):.2f}")
                self.logger.info(f"Average Win: {report.get('avg_win', 0):.2f}")
                self.logger.info(f"Average Loss: {report.get('avg_loss', 0):.2f}")
                self.logger.info(f"Profit Factor: {report.get('profit_factor', 0):.2f}")
                self.logger.info("="*50 + "\n")
            else:
                self.logger.warning("No trades found for the reporting period")
                
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}", exc_info=True)
    
    def shutdown(self):
        """Shut down the trading system gracefully."""
        if not self.running:
            return
            
        self.logger.info("Shutting down trading system...")
        self.running = False
        
        try:
            # Shutdown position manager
            if self.position_manager:
                self.position_manager.stop()
            
            # Shutdown signal monitor
            if self.signal_monitor:
                self.signal_monitor.stop()
            
            # Shutdown orchestrator
            if self.orchestrator:
                self.orchestrator.shutdown()
                
            # Generate final report if in backtest mode
            if self.args.mode == 'backtest' and not self.args.generate_report:
                self._generate_report()
                
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}", exc_info=True)
        finally:
            logging.shutdown()


def main():
    """Main entry point for the trading system."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Create and run the trading system
    trading_system = TradingSystem(args)
    sys.exit(0 if trading_system.run() else 1)

if __name__ == "__main__":
    main()
