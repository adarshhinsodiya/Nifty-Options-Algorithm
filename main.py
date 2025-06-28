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
        """Run live trading mode with enhanced monitoring and error handling."""
        self.logger.info("Starting live trading with enhanced monitoring...")
        
        # Start position manager in a separate thread
        import threading
        position_thread = threading.Thread(
            target=self.position_manager.start, 
            name="PositionManagerThread",
            daemon=True
        )
        position_thread.start()
        
        # Start signal monitor in a separate thread
        signal_thread = threading.Thread(
            target=self.signal_monitor.start,
            name="SignalMonitorThread",
            daemon=True
        )
        signal_thread.start()
        
        # Start a thread for periodic health checks
        health_check_interval = 300  # 5 minutes
        last_health_check = time.time()
        
        # Main trading loop
        while self.running:
            try:
                current_time = time.time()
                
                # Log system status periodically
                if current_time - last_health_check >= health_check_interval:
                    self._log_system_status()
                    last_health_check = current_time
                
                # Log portfolio status periodically
                self._log_portfolio_status()
                
                # Check thread status
                if not position_thread.is_alive():
                    self.logger.error("Position manager thread died, attempting to restart...")
                    position_thread = threading.Thread(
                        target=self.position_manager.start,
                        name="PositionManagerThread-Restarted",
                        daemon=True
                    )
                    position_thread.start()
                
                if not signal_thread.is_alive():
                    self.logger.error("Signal monitor thread died, attempting to restart...")
                    signal_thread = threading.Thread(
                        target=self.signal_monitor.start,
                        name="SignalMonitorThread-Restarted",
                        daemon=True
                    )
                    signal_thread.start()
                
                # Sleep for a short duration to prevent high CPU usage
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in main trading loop: {str(e)}", exc_info=True)
                time.sleep(5)  # Prevent tight error loop
    
    def _log_system_status(self):
        """Log system status including signal processing metrics."""
        try:
            if not hasattr(self, 'orchestrator') or not self.orchestrator:
                return
                
            # Get signal metrics from orchestrator
            signal_metrics = self.orchestrator.get_signal_metrics()
            
            # Log signal processing status
            self.logger.info("\n" + "="*50)
            self.logger.info("SYSTEM STATUS")
            self.logger.info("="*50)
            self.logger.info(f"Uptime: {signal_metrics.get('uptime_minutes', 0):.1f} minutes")
            self.logger.info(f"Signals Processed: {signal_metrics.get('signals_processed', 0)}")
            self.logger.info(f"Signals Failed: {signal_metrics.get('signals_failed', 0)}")
            self.logger.info(f"Signal Queue Size: {signal_metrics.get('queue_size', 0)}")
            
            # Log last error if any
            if signal_metrics.get('last_error'):
                self.logger.warning(f"Last Error: {signal_metrics.get('last_error')}")
                
            self.logger.info("="*50 + "\n")
            
            # Log thread status
            self.logger.debug("Active threads:")
            for thread in threading.enumerate():
                self.logger.debug(f"  {thread.name}: {'Alive' if thread.is_alive() else 'Dead'}")
                
        except Exception as e:
            self.logger.error(f"Error logging system status: {str(e)}", exc_info=True)
    
    def _log_portfolio_status(self):
        """Log portfolio status with enhanced metrics and error handling."""
        if not hasattr(self, 'portfolio_reporter') or not self.portfolio_reporter:
            return
            
        try:
            # Log open positions summary
            open_positions = self.portfolio_reporter.get_open_positions_summary()
            if open_positions and 'positions' in open_positions and open_positions['positions']:
                self.logger.info("\n" + "-"*40)
                self.logger.info("OPEN POSITIONS")
                self.logger.info("-"*40)
                for pos in open_positions['positions']:
                    pnl_pct = (pos.get('unrealized_pnl', 0) / (pos.get('entry_price', 1) * abs(pos.get('quantity', 1)))) * 100
                    self.logger.info(
                        f"{pos.get('symbol', 'N/A'):<10} | "
                        f"Qty: {pos.get('quantity', 0):<5} | "
                        f"Entry: {pos.get('entry_price', 0):<8.2f} | "
                        f"LTP: {pos.get('current_price', pos.get('entry_price', 0)):<8.2f} | "
                        f"P&L: {pos.get('unrealized_pnl', 0):<8.2f} ({pnl_pct:.2f}%)"
                    )
                self.logger.info("-"*40)
            
            # Log daily P&L
            daily_report = self.portfolio_reporter.generate_daily_report()
            if daily_report and 'total_pnl' in daily_report:
                self.logger.info("\n" + "-"*40)
                self.logger.info("DAILY PERFORMANCE")
                self.logger.info("-"*40)
                self.logger.info(f"Trades: {daily_report.get('total_trades', 0)}")
                self.logger.info(f"Win Rate: {daily_report.get('win_rate', 0):.1f}%")
                self.logger.info(f"Total P&L: {daily_report.get('total_pnl', 0):.2f}")
                self.logger.info(f"Average Win: {daily_report.get('avg_win', 0):.2f}")
                self.logger.info(f"Average Loss: {daily_report.get('avg_loss', 0):.2f}")
                self.logger.info(f"Profit Factor: {daily_report.get('profit_factor', 0):.2f}")
                self.logger.info("-"*40 + "\n")
                
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
        """Shut down the trading system gracefully with proper cleanup."""
        if not self.running:
            return
            
        self.logger.info("Initiating graceful shutdown...")
        self.running = False
        
        try:
            # Log final status before shutting down
            try:
                self._log_system_status()
                self._log_portfolio_status()
            except Exception as e:
                self.logger.error(f"Error logging final status: {str(e)}", exc_info=True)
            
            # Shutdown components in reverse order of initialization
            components = [
                ("Position Manager", self.position_manager, 'stop'),
                ("Signal Monitor", self.signal_monitor, 'stop'),
                ("Orchestrator", self.orchestrator, 'shutdown')
            ]
            
            for name, component, method_name in components:
                if component:
                    try:
                        self.logger.info(f"Shutting down {name}...")
                        if hasattr(component, method_name):
                            getattr(component, method_name)()
                            self.logger.info(f"{name} shutdown complete")
                        else:
                            self.logger.warning(f"{name} does not have {method_name} method")
                    except Exception as e:
                        self.logger.error(f"Error shutting down {name}: {str(e)}", exc_info=True)
            
            # Generate final report if in backtest mode
            if self.args.mode == 'backtest' and not self.args.generate_report:
                self._generate_report()
                
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}", exc_info=True)
        finally:
            try:
                # Ensure all logging is flushed
                logging.shutdown()
                self.logger.info("Trading system shutdown complete")
            except Exception as e:
                print(f"Error during logging shutdown: {str(e)}")  # Use print as logging might be down


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
