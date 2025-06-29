# NIFTY Options Trading System

A sophisticated algorithmic trading system designed for trading NIFTY options with a focus on risk management and robust execution. The system supports both live trading and backtesting modes, featuring a modular architecture that separates strategy, execution, and risk management concerns.

## 🌟 Key Features

- **Multi-timeframe Analysis**: Supports multiple timeframes for comprehensive market analysis
- **Robust Signal Generation**: Advanced pattern recognition with configurable parameters
- **Risk Management**: Comprehensive risk controls including position sizing, stop losses, and daily loss limits
- **Live Trading**: Seamless integration with ICICI Direct's Breeze API
- **Backtesting Engine**: Historical simulation with detailed performance metrics
- **Real-time Monitoring**: Live position tracking and performance dashboards
- **Modular Architecture**: Clean separation of concerns for easy maintenance and extension

## 🏗️ System Architecture

### Core Components

1. **Main Application (`main.py`)**
   - Entry point for the trading system
   - Manages system lifecycle and component initialization
   - Handles command-line arguments and configuration
   - Coordinates between different system components

2. **Core Modules (`core/`)**
   - `models.py`: Data models (TradeSignal, Position, etc.)
   - `config_manager.py`: Configuration management
   - `signal_monitor.py`: Real-time signal processing and queue management
   - `orchestrator.py`: Main trading logic and workflow orchestration

3. **Execution Layer (`execution/`)**
   - `trade_executor.py`: Handles order execution and position management
   - `position_manager.py`: Monitors and manages open positions
   - `risk_manager.py`: Implements risk management rules

4. **Data Layer**
   - `data_provider.py`: Abstract interface for market data
   - `breeze_data.py`: ICICI Direct Breeze API integration
   - `historical_data.py`: Historical data management

5. **Strategies (`strategies/`)**
   - `base_strategy.py`: Abstract base class for trading strategies
   - `candlestick_strategy.py`: Candlestick pattern-based strategy
   - `mean_reversion.py`: Mean reversion strategy

6. **Utilities (`utils/`)**
   - `logger.py`: Custom logging configuration
   - `helpers.py`: Common utility functions
   - `performance.py`: Performance metrics and analysis

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- ICICI Direct Breeze API credentials (for live trading)
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Nifty-Options-Trading-System
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your settings in `config/config.ini`:
   ```ini
   [trading]
   symbol = NIFTY
   product_type = options
   quantity = 50
   risk_per_trade = 1.0
   max_positions = 5
   ```
   - `[signals]`: Adjust signal generation parameters

## Usage

### Backtesting Mode

To run a backtest with default settings:

```bash
python main.py --mode backtest
```

Customize the backtest with command-line options:

```bash
python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31 --initial-capital 100000
```

### Live Trading Mode

To run in live trading mode:

```bash
python main.py --mode live
```

### Command-line Arguments

- `--mode`: Operation mode (`live` or `backtest`)
- `--config`: Path to configuration file (default: `config/config.ini`)
- `--start-date`: Start date for backtesting (YYYY-MM-DD)
- `--end-date`: End date for backtesting (YYYY-MM-DD)
- `--initial-capital`: Initial capital for backtesting
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Strategy Overview

The trading strategy is based on 2-candle patterns with the following characteristics:

### Long Signal (Bullish Engulfing with Confirmation)
1. Previous candle is bearish (close < open)
2. Upper wick is larger than the body
3. Upper wick is larger than the lower wick
4. Current candle makes a lower low
5. Current candle closes above previous open
6. Current candle is bullish (close > open)

### Short Signal (Bearish Engulfing with Confirmation)
1. Previous candle is bullish (close > open)
2. Lower wick is larger than the body
3. Lower wick is larger than the upper wick
4. Current candle makes a higher high
5. Current candle closes below previous open
6. Current candle is bearish (close < open)

### Position Management
- **Entry**: Market order on signal confirmation
- **Stop Loss**: Based on ATR (Average True Range)
- **Take Profit**: 1.5x risk-reward ratio
- **Exit**: At stop-loss, take-profit, or end of day

## Project Structure

```
NIFTY-Options-Trading-System/
├── config/                    # Configuration files
│   └── config.ini             # Main configuration file
├── core/                      # Core functionality
│   ├── config_manager.py      # Configuration management
│   ├── models.py              # Data models
│   ├── orchestrator.py        # Trading orchestrator
│   └── signal_monitor.py      # Signal processing
├── data/                      # Market data
│   ├── historical/            # Historical price data
│   └── signals/               # Generated signals
├── execution/                 # Trade execution
│   ├── position_manager.py    # Position management
│   ├── risk_manager.py        # Risk management
│   └── trade_executor.py      # Order execution
├── strategies/                # Trading strategies
│   ├── base_strategy.py       # Base strategy class
│   ├── candlestick_strategy.py # Pattern-based strategy
│   └── mean_reversion.py      # Mean reversion strategy
├── tests/                     # Unit tests
├── utils/                     # Utility functions
│   ├── helpers.py             # Helper functions
│   ├── logger.py              # Logging configuration
│   └── performance.py         # Performance metrics
├── main.py                    # Main application
└── requirements.txt           # Python dependencies
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or support, please contact [Your Name] at [adarsh260506@gmail.com]
- Trade-by-trade breakdown
- Equity curve

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors and contributors are not responsible for any financial losses incurred while using this software. Always test thoroughly with paper trading before using real money.

## 🛠️ Configuration

The system is highly configurable through the `config.ini` file. Key sections include:

- `[trading]`: General trading parameters
- `[api]`: Broker API credentials
- `[risk]`: Risk management settings
- `[signals]`: Signal generation parameters
- `[backtest]`: Backtesting configuration

## 📊 Usage

### Live Trading

```bash
python main.py --mode live --log-level INFO
```

### Backtesting

```bash
python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31
```

### Available Command-line Arguments

```
--mode           Operation mode (live/backtest)
--config         Path to config file
--symbol         Trading symbol (overrides config)
--quantity       Position size (overrides config)
--start-date     Backtest start date (YYYY-MM-DD)
--end-date       Backtest end date (YYYY-MM-DD)
--log-level      Logging level (DEBUG/INFO/WARNING/ERROR)
--generate-report Generate performance report after backtest
```

## 📈 Performance Monitoring

The system provides real-time monitoring through:

1. **Log Files**: Detailed logging of all system activities
2. **Console Output**: Summary of trades and performance metrics
3. **Performance Reports**: Comprehensive HTML reports with:
   - Equity curve
   - Trade statistics
   - Drawdown analysis
   - Risk metrics

## 🔄 Workflow

1. **Signal Generation**
   - Market data is analyzed for trading opportunities
   - Signals are generated based on configured strategies
   - Signals are validated against risk parameters

2. **Order Execution**
   - Valid signals are sent to the execution engine
   - Orders are placed with proper risk controls
   - Order status is monitored and managed

3. **Position Management**
   - Open positions are tracked in real-time
   - Stop-loss and take-profit levels are monitored
   - Positions are adjusted or closed based on market conditions

4. **Risk Management**
   - Position sizing based on account equity
   - Maximum drawdown limits
   - Daily loss limits
   - Maximum position limits

## 🛡️ Risk Management

The system implements multiple layers of risk controls:

1. **Pre-trade Checks**
   - Maximum position size limits
   - Maximum number of open positions
   - Available margin checks

2. **Intra-trade Controls**
   - Stop-loss orders
   - Trailing stops
   - Time-based exits

3. **Portfolio-level Protections**
   - Maximum daily loss limit
   - Maximum drawdown limit
   - Position concentration limits

## 📚 Documentation

### Key Classes

#### TradeSignal
Represents a trading signal with all necessary attributes for execution.

**Key Attributes:**
- `signal_type`: BUY/SELL
- `entry_price`: Entry price
- `stop_loss`: Stop loss price
- `take_profit`: Take profit price
- `symbol`: Trading symbol
- `quantity`: Position size
- `timestamp`: Signal generation time

#### TradeExecutor
Handles order execution and position management.

**Key Methods:**
- `execute_signal()`: Process a trading signal
- `place_order()`: Place an order with the broker
- `cancel_order()`: Cancel an open order
- `get_position()`: Get current position for a symbol

#### PositionManager
Monitors and manages open positions.

**Key Features:**
- Real-time P&L tracking
- Stop-loss monitoring
- Position adjustment
- Risk limit enforcement
