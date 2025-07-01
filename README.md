# NIFTY Options Trading System

A sophisticated algorithmic trading system designed for trading NIFTY options with a focus on risk management and robust execution. The system supports both live trading and backtesting modes, featuring a modular architecture that separates strategy, execution, and risk management concerns.

## 🌟 Key Features

### Trading Features
- **Automated Trading**: Fully automated trading of NIFTY options
- **Multiple Timeframes**: Support for various timeframes (1m, 5m, 15m, etc.)
- **WebSocket Integration**: Real-time market data streaming
- **Backtesting Engine**: Historical simulation with detailed analytics
- **Paper Trading**: Risk-free trading with virtual funds

### Risk Management
- **Position Sizing**: Dynamic position sizing based on account balance
- **Stop Loss/Take Profit**: Automated risk management
- **Daily Loss Limits**: Prevent excessive drawdowns
- **Overnight Position Control**: Automatic position management

### Technical Features
- **Modular Architecture**: Clean separation of concerns
- **Strategy Interface**: Easy to implement custom strategies
- **Real-time Monitoring**: Live tracking of positions and P&L
- **Alert System**: Email/Telegram notifications for trades and alerts
- **Logging**: Comprehensive logging for audit and debugging

## 🏗️ System Architecture

The system follows a modular architecture with clear separation of concerns:

### Core Components

1. **Trading Engine (`orchestrator.py`)**
   - Manages the trading lifecycle
   - Coordinates between different components
   - Handles strategy execution and order management

2. **Strategy Layer (`strategies/`)**
   - Implements trading strategies
   - Clean interface for strategy development
   - Multiple strategy support

3. **Data Layer (`data_provider.py`)**
   - Market data acquisition
   - WebSocket and REST API integration
   - Historical and real-time data handling

4. **Execution Layer (`execution/`)**
   - Order placement and management
   - Position tracking
   - Risk management

5. **Monitoring & Reporting**
   - Real-time P&L tracking
   - Trade logging
   - Performance analytics

## Project Structure

```
NIFTY-Options-Algorithm/
├── config/                    # Configuration files
│   └── config.ini             # Main configuration file
│
├── core/                     # Core business logic
│   ├── config_manager.py      # Configuration management
│   ├── exceptions.py          # Custom exceptions
│   ├── models.py              # Data models (TradeSignal, Position, etc.)
│   ├── orchestrator.py        # Main trading workflow
│   ├── reporting.py           # Reporting and analytics
│   ├── signal_monitor.py      # Real-time signal processing
│   └── websocket/             # WebSocket implementation
│       └── websocket_handler.py  # WebSocket handler for real-time data
│
├── data/                     # Market data
│   └── data_provider.py       # Data provider interface and implementation
│
├── docs/                     # Documentation
│   └── retry_logic.md         # Documentation on retry mechanisms
│
├── execution/                # Trade execution
│   ├── position_manager.py    # Position tracking and management
│   └── trade_executor.py      # Order execution logic
│
├── strategies/               # Trading strategies
│   ├── base_strategy.py       # Abstract strategy class
│   ├── candle_pattern.py      # Candle pattern strategy
│   └── strategy_interface.txt  # Strategy interface documentation
│
├── tests/                    # Test cases
│   └── test_retry.py          # Test cases for retry logic
│
├── ITM_Options_Examples_Final.pdf  # Example strategy document
├── main.py                    # Application entry point
└── requirements.txt           # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- ICICI Direct Breeze API credentials
- Virtual environment (recommended)

### Environment Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment Variables**
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file and add your ICICI Direct Breeze API credentials:
     ```
     # Required
     BREEZE_API_KEY=your_api_key_here
     BREEZE_API_SECRET=your_api_secret_here
     
     # Optional
     BREEZE_SESSION_TOKEN=your_session_token_here  # Can be generated if not provided
     BREEZE_API_URL=https://api.icicidirect.com/breezeapi/api/v2/
     ```

   - **Security Note**: Never commit your `.env` file to version control. It's already included in `.gitignore`.

3. **Generate Session Token (if needed)**
   If you don't have a session token, the system can generate one for you on first run.

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

## Strategy Implementation

### Strategy Interface

The system uses a clean interface for implementing trading strategies. Each strategy must implement the following methods:

```python
class BaseStrategy(ABC):
    @abstractmethod
    def check_entry_conditions(self, data: pd.DataFrame) -> bool:
        """Check if entry conditions are met."""
        pass
    
    @abstractmethod
    def generate_entry_signal(self, data: pd.DataFrame) -> Optional[TradeSignal]:
        """Generate entry signal if conditions are met."""
        pass
    
    @abstractmethod
    def check_exit_conditions(self, data: pd.DataFrame, position: TradeSignal) -> bool:
        """Check if exit conditions are met."""
        pass
    
    @abstractmethod
    def generate_exit_signal(self, data: pd.DataFrame, position: TradeSignal) -> Optional[TradeSignal]:
        """Generate exit signal if conditions are met."""
        pass
```

### Candle Pattern Strategy

The default strategy is based on candlestick patterns with the following characteristics:

#### Long Signal (Bullish Engulfing with Confirmation)
1. Previous candle is bearish (close < open)
2. Upper wick is larger than the body
3. Upper wick is larger than the lower wick
4. Current candle makes a lower low
5. Current candle closes above previous open
6. Current candle is bullish (close > open)

#### Short Signal (Bearish Engulfing with Confirmation)
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
- **Overnight Positions**: Automatically closed before market close if not holding overnight

### Real-time Features
- WebSocket integration for live market data
- Real-time position monitoring
- Automatic position sizing based on account balance
- Email/Telegram alerts for trade signals and important events

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
--mode           Operation mode (live/backtest) [default: live]
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
