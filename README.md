# Nifty Options Trading System

An algorithmic trading system for trading NIFTY options based on candlestick patterns. The system supports both live trading and backtesting modes, with integration to ICICI Direct's Breeze API for live trading.

## Features

- **Candlestick Pattern Recognition**: Identifies high-probability trading opportunities based on 2-candle patterns.
- **Risk Management**: Implements stop-loss and take-profit levels with configurable risk-reward ratios.
- **Live Trading**: Connects to ICICI Direct's Breeze API for live market data and order execution.
- **Backtesting**: Comprehensive backtesting framework with performance metrics and reporting.
- **Modular Architecture**: Clean separation of concerns with dedicated modules for data handling, strategy, and execution.
- **Options Trading**: Focuses on NIFTY options with ITM+2 strike selection for better risk management.

## Prerequisites

- Python 3.8+
- ICICI Direct Breeze API credentials (for live trading)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Nifty-Option-Algorithm
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Copy the example configuration file and update it with your settings:
   ```bash
   cp config/config.ini.example config/config.ini
   ```

2. Edit `config/config.ini` and update the following sections:
   - `[api]`: Add your ICICI Direct Breeze API credentials
   - `[trading]`: Configure trading parameters like risk per trade, position limits, etc.
   - `[backtest]`: Set backtesting parameters
   - `[options]`: Configure options trading parameters
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
Nifty-Option-Algorithm/
├── config/                    # Configuration files
│   └── config.ini             # Main configuration file
├── core/                      # Core functionality
│   ├── config_manager.py      # Configuration management
│   ├── models.py              # Data models
│   └── orchestrator.py        # Main trading orchestrator
├── data/                      # Data handling
│   └── data_provider.py       # Data provider interface
├── execution/                 # Trade execution
│   └── trade_executor.py      # Order execution and position management
├── strategies/                # Trading strategies
│   └── candle_pattern.py      # Candle pattern strategy implementation
├── logs/                      # Log files
├── reports/                   # Backtest reports
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Logging

Logs are stored in the `logs/` directory with rotation (7 days retention). The log level can be configured in the config file or via command-line argument.

## Backtest Reports

After running a backtest, a detailed performance report is generated in the `reports/` directory, including:

- Summary statistics (win rate, total return, etc.)
- Trade-by-trade breakdown
- Equity curve

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors and contributors are not responsible for any financial losses incurred while using this software. Always test thoroughly with paper trading before using real money.
