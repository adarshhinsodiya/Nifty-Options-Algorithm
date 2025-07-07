# 📈 NIFTY Options Auto-Trading Bot (Python + BreezeConnect)

## Overview

This is a fully automated trading bot built in Python, designed to trade **NIFTY options** based on real-time price action and candle pattern signals. It connects to the **BreezeConnect API** (by ICICI Direct) to fetch live tick data and place market orders in the NSE F&O segment. It uses advanced reversal detection logic to identify long and short entries, complete with stop-loss (SL), take-profit (TP), and automatic square-off.

## ✅ Features

- ✅ **Live Tick Data Streaming** via WebSocket (NIFTY + options)
- ✅ **1-Minute Candle Formation** from spot data
- ✅ **Signal Evaluation** using wick rejection + engulfing logic
- ✅ **Dynamic SL/TP** using ATR-based volatility adaptation
- ✅ **Position Sizing** based on capital and risk percent
- ✅ **Failsafe Auto Square-Off** between 15:20–15:25 IST
- ✅ **CSV Logging** for trades, signals, and open positions
- ✅ **WebSocket Reconnection** and graceful shutdown support

## 🔧 Setup Instructions

### 1. **Clone the Repository**

```bash
git clone https://github.com/your-username/nifty-options-bot.git
cd nifty-options-bot
```

### 2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

> You must have a valid [BreezeConnect](https://api.icicidirect.com/) account and access to WebSocket APIs.

### 3. **Environment Setup**

Create a `.env` file in the root directory with:

```ini
BREEZE_API_KEY=your_api_key
BREEZE_API_SECRET=your_api_secret
BREEZE_SESSION_TOKEN=your_session_token
```

### 4. **Configuration**

Edit `config/config.ini`:

```ini
[trading]
quantity=75
max_quantity=525
```

## 📊 Strategy Logic

The strategy uses a modified **bullish/bearish engulfing pattern** with **wick rejection** logic to generate buy/sell signals.

### Buy Conditions:
- Previous candle bearish
- Upper wick > body and lower wick
- Current candle makes lower low
- Current close > previous open (engulfing)
- Current candle bullish

### Sell Conditions:
- Previous candle bullish
- Lower wick > body and upper wick
- Current makes higher high
- Current close < previous open
- Current candle bearish

## 🧠 Risk Management

- **Stop-Loss**: Previous candle's low/high ± ATR × 1.5
- **Take-Profit**: Based on RR ratio (default 2.0)
- **Position Sizing**: Adjusted dynamically using available capital and option premium

## 📁 File Structure

```
.
├── final.py                  # Main bot logic
├── config/
│   └── config.ini            # Quantity settings
├── data/
│   ├── open_position.csv     # Tracks current position
│   ├── trade_log.csv         # Logs order history
│   └── signal_log.csv        # Logs buy/sell signal events
├── .env                      # API credentials (excluded from repo)
└── requirements.txt          # All dependencies
```

## 🚦 Usage

```bash
python final.py
```

Logs will be generated to both console and `trading_bot.log`.

## 🧪 Testing

This bot is **paper-trade ready**. Live trading requires proper margin setup and exchange approvals (if deploying through Tradetron or broker-hosted infra).

## 📩 Contact & Support

For any questions, improvements, or integration help:

- 📧 Email: adarshhinsodiya@gmail.com
- 📞 Phone: [your number, optional]
- 🔗 LinkedIn / GitHub: [optional links]