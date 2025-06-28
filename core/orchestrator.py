"""
Main orchestrator for the trading system.
Handles both live trading and backtesting modes.
"""
import logging
import time
import json
import os
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import pytz
from pathlib import Path

from core.config_manager import ConfigManager
from core.models import Portfolio, Position, TradeSignal, TradeSignalType
from data.data_provider import DataProvider
from execution.trade_executor import TradeExecutor
from strategies.candle_pattern import CandlePatternStrategy

class TradingOrchestrator:
    """Orchestrates the trading system in both live and backtest modes."""
    
    def __init__(self, config_path: str = "config/config.ini", mode: str = 'backtest'):
        """Initialize the trading orchestrator.
        
        Args:
            config_path: Path to the configuration file
            mode: Operation mode ('backtest' or 'live')
        """
        self.mode = mode.lower()
        self.config = ConfigManager(config_path)
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_provider = DataProvider(self.config, mode=self.mode)
        self.portfolio = Portfolio(
            initial_capital=self.config.get_backtest_config().initial_capital,
            current_cash=self.config.get_backtest_config().initial_capital
        )
        self.trade_executor = TradeExecutor(self.config, self.data_provider, self.portfolio)
        self.strategy = CandlePatternStrategy(self.config)
        
        # State tracking
        self.running = False
        self.last_candle_time = None
        self.current_date = None
        
        # Tick data aggregation
        self.current_minute = None
        self.current_candle = None
        self.ticks_buffer = []
    
    def _setup_logging(self) -> None:
        """Configure logging for the application."""
        log_config = self.config.get_logging_config()
        log_level = getattr(logging, log_config['log_level'].upper(), logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_config['log_file'])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config['log_file']),
                logging.StreamHandler()
            ]
        )
        
        # Set up log rotation
        from logging.handlers import TimedRotatingFileHandler
        file_handler = TimedRotatingFileHandler(
            log_config['log_file'],
            when='d',
            interval=1,
            backupCount=7,
            encoding='utf-8'
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(logging.StreamHandler())
    
    def run(self) -> None:
        """Run the trading system in the specified mode."""
        self.logger.info(f"Starting trading system in {self.mode.upper()} mode")
        
        try:
            if self.mode == 'backtest':
                self._run_backtest()
            else:
                self._run_live()
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down gracefully...")
        except Exception as e:
            self.logger.error(f"Error in trading system: {str(e)}", exc_info=True)
        finally:
            self.shutdown()
    
    def _run_backtest(self) -> None:
        """Run backtesting mode."""
        self.logger.info("Starting backtest...")
        
        # Get backtest configuration
        backtest_config = self.config.get_backtest_config()
        start_date = datetime.strptime(backtest_config.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(backtest_config.end_date, '%Y-%m-%d').date()
        
        # Get historical data
        df = self._load_historical_data(start_date, end_date)
        if df.empty:
            self.logger.error("No historical data available for backtest")
            return
        
        # Generate signals
        df = self.strategy.generate_signals(df)
        
        # Process each candle
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Update current date for EOD checks
            self.current_date = current.name.date()
            
            # Process any open positions
            self.trade_executor.manage_positions()
            
            # Check for new signals
            if pd.notna(current['signal']):
                # Get the fully enriched signal from the DataFrame
                if '_signal_obj' in current and current['_signal_obj'] is not None:
                    signal = current['_signal_obj']
                    self.logger.debug(f"Executing signal from DataFrame: {signal}")
                else:
                    # Fallback to generating the signal if not found in DataFrame
                    self.logger.warning("Signal object not found in DataFrame, falling back to analysis")
                    signal_type, signal = self.strategy.analyze_candle_pattern(df, i)
                    if not signal:
                        self.logger.warning("Strategy returned no signal despite signal flag")
                        continue
                
                # Execute the signal
                self.trade_executor.execute_signal(signal)
            
            # Sleep to simulate real-time (optional)
            time.sleep(0.01)
        
        # Close any remaining positions at the end of backtest
        self.trade_executor.square_off_all_positions()
        
        # Generate performance report
        self._generate_performance_report()
    
    def _run_live(self) -> None:
        """Run live trading mode."""
        self.logger.info("Starting live trading...")
        
        # Initialize real-time data subscription
        self._setup_realtime_data()
        
        # Initialize tick data structures
        self.current_minute = None
        self.current_candle = None
        self.ticks_buffer = []
        
        # Main trading loop
        self.running = True
        while self.running:
            try:
                current_time = datetime.now(self.ist_tz)
                
                # Check if market is open
                if not self._is_market_hours():
                    if self._is_market_closed():
                        # Process any remaining ticks before closing
                        if self.current_candle:
                            self._process_completed_candle()
                        # Square off all positions at EOD
                        self.trade_executor.square_off_all_positions()
                        self.logger.info("Market closed. Shutting down...")
                        break
                    
                    # Wait for market to open
                    time.sleep(60)
                    continue
                
                # Process any ticks that have been received
                if self.ticks_buffer:
                    # Process ticks in batches to avoid blocking
                    ticks_to_process = self.ticks_buffer.copy()
                    self.ticks_buffer = []
                    self._process_ticks(ticks_to_process)
                
                # Sleep to avoid excessive CPU usage
                time.sleep(0.1)  # 100ms sleep for better tick processing
                
            except Exception as e:
                self.logger.error(f"Error in live trading loop: {str(e)}", exc_info=True)
                time.sleep(5)  # Wait before retrying
    
    def _load_historical_data(
        self, 
        start_date: datetime.date, 
        end_date: datetime.date
    ) -> pd.DataFrame:
        """Load historical data for backtesting."""
        self.logger.info(f"Loading historical data from {start_date} to {end_date}")
        
        # Get data from data provider
        df = self.data_provider.get_historical_data(
            symbol=self.config.get_trading_config().symbol,
            from_date=start_date,
            to_date=end_date,
            interval='1minute',
            exchange_code='NSE',
            product_type='cash'
        )
        
        if df.empty:
            self.logger.warning("No historical data returned from data provider")
            return pd.DataFrame()
        
        # Ensure index is timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize(self.ist_tz)
        
        self.logger.info(f"Loaded {len(df)} candles of historical data")
        return df
    
    def _process_ticks(self, ticks: List[Dict[str, Any]]) -> None:
        """Process a batch of ticks."""
        if not ticks:
            return
            
        try:
            # Process each tick in the batch
            for tick in ticks:
                tick_time = tick.get('datetime', datetime.now(self.ist_tz))
                if not isinstance(tick_time, datetime):
                    tick_time = datetime.fromisoformat(tick_time) if isinstance(tick_time, str) else datetime.now(self.ist_tz)
                
                tick_price = float(tick.get('last', 0))
                tick_volume = int(tick.get('volume', 0))
                
                # Initialize first candle if needed
                if self.current_candle is None:
                    self._initialize_new_candle(tick_time)
                
                # Check if we've moved to a new minute
                current_minute = tick_time.replace(second=0, microsecond=0)
                if current_minute > self.current_minute:
                    # Process the completed candle
                    self._process_completed_candle()
                    # Start a new candle
                    self._initialize_new_candle(tick_time)
                
                # Update current candle with tick data
                if self.current_candle['open'] is None:
                    self.current_candle['open'] = tick_price
                
                self.current_candle['high'] = max(
                    self.current_candle['high'] or tick_price, 
                    tick_price
                )
                self.current_candle['low'] = min(
                    self.current_candle['low'] or tick_price, 
                    tick_price
                )
                self.current_candle['close'] = tick_price
                self.current_candle['volume'] += tick_volume
                
        except Exception as e:
            self.logger.error(f"Error processing ticks: {str(e)}", exc_info=True)
    
    def _setup_realtime_data(self) -> None:
        """Set up real-time data subscription."""
        symbol = self.config.get_trading_config().symbol
        
        # Subscribe to index/stock data
        self.data_provider.subscribe_feeds(
            [
                {
                    'stock_code': symbol,
                    'exchange_code': 'NSE',
                    'product_type': 'cash',
                    'get_exchange_quotes': True,
                    'get_market_depth': False
                }
            ],
            self._on_tick
        )
    
    def _initialize_new_candle(self, tick_time: datetime) -> None:
        """Initialize a new candle with the given timestamp."""
        self.current_minute = tick_time.replace(second=0, microsecond=0)
        self.current_candle = {
            'datetime': self.current_minute,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': 0,
            'symbol': self.config.get_trading_config().symbol,
            'interval': '1m'
        }
        self.ticks_buffer = []
        self.logger.debug(f"Initialized new candle at {self.current_minute}")
    
    def _process_completed_candle(self) -> None:
        """Process the completed candle and check for signals."""
        if not self.current_candle or 'close' not in self.current_candle:
            return
            
        self.logger.debug(
            f"Completed 1m candle: {self.current_candle['datetime']} | "
            f"O:{self.current_candle['open']:.2f} H:{self.current_candle['high']:.2f} "
            f"L:{self.current_candle['low']:.2f} C:{self.current_candle['close']:.2f} "
            f"V:{self.current_candle['volume']}"
        )
        
        # Check for signals based on the completed candle
        signal = self._check_for_signals(self.current_candle)
        if signal:
            self.trade_executor.execute_signal(signal)
    
    def _on_tick(self, ticks: List[Dict[str, Any]]) -> None:
        """Handle incoming real-time ticks and aggregate into 1-minute candles.
        
        Args:
            ticks: List of tick data dictionaries with 'last' price and 'volume'
        """
        if not ticks:
            return
            
        for tick in ticks:
            try:
                tick_time = tick.get('datetime', datetime.now(self.ist_tz))
                if not isinstance(tick_time, datetime):
                    tick_time = datetime.fromisoformat(tick_time) if isinstance(tick_time, str) else datetime.now(self.ist_tz)
                
                tick_price = float(tick.get('last', 0))
                tick_volume = int(tick.get('volume', 0))
                
                # Initialize first candle if needed
                if self.current_candle is None:
                    self._initialize_new_candle(tick_time)
                
                # Check if we've moved to a new minute
                current_minute = tick_time.replace(second=0, microsecond=0)
                if current_minute > self.current_minute:
                    # Process the completed candle
                    self._process_completed_candle()
                    # Start a new candle
                    self._initialize_new_candle(tick_time)
                
                # Update current candle with tick data
                if self.current_candle['open'] is None:
                    self.current_candle['open'] = tick_price
                
                self.current_candle['high'] = max(
                    self.current_candle['high'] or tick_price, 
                    tick_price
                )
                self.current_candle['low'] = min(
                    self.current_candle['low'] or tick_price, 
                    tick_price
                )
                self.current_candle['close'] = tick_price
                self.current_candle['volume'] += tick_volume
                
                # Store tick in buffer (useful for more granular analysis if needed)
                self.ticks_buffer.append({
                    'datetime': tick_time,
                    'price': tick_price,
                    'volume': tick_volume
                })
                
            except Exception as e:
                self.logger.error(f"Error processing tick {tick}: {str(e)}", exc_info=True)
    
    def _check_for_signals(self, candle: Dict[str, Any]) -> Optional[TradeSignal]:
        """Check for trading signals based on the latest candle."""
        # Convert single candle to DataFrame for analysis
        df = pd.DataFrame([candle])
        
        # Add previous candle for pattern recognition
        if self.last_candle_time is not None and self.last_candle_time < candle['datetime']:
            # Get previous candle from data provider or cache
            prev_candle = self._get_previous_candle(candle['datetime'])
            if prev_candle is not None:
                df = pd.concat([pd.DataFrame([prev_candle]), df], ignore_index=True)
        
        # Generate signals
        df = self.strategy.generate_signals(df)
        
        # Check for new signals
        if not df.empty and 'signal' in df.columns and pd.notna(df.iloc[-1]['signal']):
            # Get the fully enriched signal from the strategy
            signal_type, signal = self.strategy.analyze_candle_pattern(df, len(df) - 1)
            
            if signal:
                self.last_candle_time = candle['datetime']
                return signal
            else:
                self.logger.warning("Strategy returned no signal despite signal flag")
            
        return None
    
    def _get_previous_candle(self, current_time: datetime) -> Optional[Dict[str, Any]]:
        """Get the previous candle for the given time."""
        # In a real implementation, you'd fetch this from your data store
        # This is a placeholder that would be replaced with actual data retrieval
        return None
    
    def _get_latest_candle(self) -> Optional[Dict[str, Any]]:
        """Get the latest candle data."""
        # In a real implementation, you'd get this from your data provider
        # This is a placeholder that would be replaced with actual data retrieval
        return None
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        now = datetime.now(self.ist_tz)
        market_open = time(9, 15)  # 9:15 AM IST
        market_close = time(15, 30)  # 3:30 PM IST
        
        # Check if it's a weekday (0 = Monday, 6 = Sunday)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
            
        # Check if within market hours
        current_time = now.time()
        return market_open <= current_time <= market_close
    
    def _is_market_closed(self) -> bool:
        """Check if market is closed for the day."""
        now = datetime.now(self.ist_tz)
        market_close = time(15, 30)  # 3:30 PM IST
        return now.time() > market_close
    
    def _prepare_for_trading_day(self) -> None:
        """Prepare for a new trading day."""
        now = datetime.now(self.ist_tz)
        
        # Only run once per day before market opens
        if hasattr(self, '_last_prep_date') and self._last_prep_date == now.date():
            return
        
        self.logger.info(f"Preparing for trading day: {now.date()}")
        
        # Cancel any pending orders from previous day
        self.trade_executor.cancel_all_pending_orders()
        
        # Reset daily counters
        self._last_prep_date = now.date()
    
    def _get_next_expiry(self, current_date: datetime) -> datetime:
        """Get the next expiry date (Thursday)."""
        # Calculate days until next Thursday
        days_until_thursday = (3 - current_date.weekday()) % 7  # 3 = Thursday
        if days_until_thursday == 0:  # If today is Thursday, use next Thursday
            days_until_thursday = 7
            
        expiry_date = (current_date + timedelta(days=days_until_thursday))\
            .replace(hour=15, minute=30, second=0, microsecond=0)
            
        return expiry_date
    
    def _generate_performance_report(self) -> None:
        """Generate a performance report after backtesting."""
        if not self.portfolio.closed_positions:
            self.logger.warning("No closed positions to generate performance report")
            return
        
        # Basic performance metrics
        total_trades = len(self.portfolio.closed_positions)
        winning_trades = sum(1 for p in self.portfolio.closed_positions if p.pnl and p.pnl > 0)
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(p.pnl for p in self.portfolio.closed_positions if p.pnl is not None)
        avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        # Calculate max drawdown
        equity_curve = []
        running_balance = self.config.get_backtest_config().initial_capital
        
        for position in sorted(self.portfolio.closed_positions, key=lambda x: x.exit_time):
            running_balance += position.pnl if position.pnl else 0
            equity_curve.append({
                'date': position.exit_time,
                'equity': running_balance
            })
        
        # Generate report
        report = {
            'summary': {
                'start_date': min(p.entry_time for p in self.portfolio.closed_positions).strftime('%Y-%m-%d'),
                'end_date': max(p.exit_time for p in self.portfolio.closed_positions).strftime('%Y-%m-%d'),
                'initial_capital': self.config.get_backtest_config().initial_capital,
                'final_equity': running_balance,
                'total_return': running_balance - self.config.get_backtest_config().initial_capital,
                'total_return_pct': ((running_balance / self.config.get_backtest_config().initial_capital) - 1) * 100,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_pnl_per_trade': avg_pnl_per_trade,
                'profit_factor': 0,  # Would need to calculate from wins/losses
                'max_drawdown': 0,   # Would need to calculate from equity curve
                'sharpe_ratio': 0    # Would need returns data
            },
            'trades': [
                {
                    'entry_time': p.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_time': p.exit_time.strftime('%Y-%m-%d %H:%M:%S') if p.exit_time else None,
                    'symbol': p.signal.symbol,
                    'direction': p.signal.signal_type.value,
                    'entry_price': p.entry_order.average_price,
                    'exit_price': p.exit_order.average_price if p.exit_order else None,
                    'quantity': p.entry_order.quantity,
                    'pnl': p.pnl,
                    'pnl_pct': p.pnl_percentage,
                    'exit_reason': p.exit_reason
                }
                for p in self.portfolio.closed_positions
            ]
        }
        
        # Save report to file
        report_dir = 'reports'
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance report saved to {report_file}")
        
        # Print summary to console
        self.logger.info("\n" + "="*50)
        self.logger.info("BACKTEST PERFORMANCE SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Period: {report['summary']['start_date']} to {report['summary']['end_date']}")
        self.logger.info(f"Initial Capital: ₹{report['summary']['initial_capital']:,.2f}")
        self.logger.info(f"Final Equity: ₹{report['summary']['final_equity']:,.2f}")
        self.logger.info(f"Total Return: ₹{report['summary']['total_return']:,.2f} ({report['summary']['total_return_pct']:.2f}%)")
        self.logger.info(f"Total Trades: {report['summary']['total_trades']}")
        self.logger.info(f"Winning Trades: {report['summary']['winning_trades']} ({report['summary']['win_rate']:.2f}%)")
        self.logger.info(f"Losing Trades: {report['summary']['losing_trades']}")
        self.logger.info(f"Avg P&L per Trade: ₹{report['summary']['avg_pnl_per_trade']:,.2f}")
        self.logger.info("="*50)
    
    def shutdown(self) -> None:
        """Shut down the trading system gracefully."""
        self.logger.info("Shutting down trading system...")
        self.running = False
        
        # Close all open positions
        self.trade_executor.square_off_all_positions()
        
        # Cancel all pending orders
        self.trade_executor.cancel_all_pending_orders()
        
        self.logger.info("Trading system shutdown complete")
