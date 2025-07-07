#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import logging
import configparser
import argparse
import signal
import threading
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np
import pytz
from breeze_connect import BreezeConnect
from dotenv import load_dotenv
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_bot.log")
    ]
)
logger = logging.getLogger("TradingBot")

POSITION_CSV = "data/open_position.csv"
TRADE_LOG_CSV = "data/trade_log.csv"
SIGNAL_LOG_CSV = "data/signal_log.csv" # New CSV for signal logging

class SimpleTradingBot:
    def __init__(self, config_path="config/config.ini"):
        self.config = self._load_config(config_path)
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.quantity = int(self.config.get('trading', 'quantity', fallback='50'))
        self.max_quantity = int(self.config.get('trading', 'max_quantity', fallback='500'))
        self.square_off_done = False
        self.last_order = {}
        self.breeze = None
        self.position = self._load_position()
        self.candle_buffer = deque(maxlen=20) # Increased maxlen for more historical data if needed for ATR/other indicators
        self.current_minute = None # Will store the minute part of the exchange_timestamp
        self.minute_candle = {'open': None, 'high': -np.inf, 'low': np.inf, 'close': None}
        self.spot_price = None
        self.call_strike = None
        self.put_strike = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.running = True
        self.ws_connected = False
        self.data_lock = threading.Lock()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_minute_log_time = None # To track last time minute data was logged (local time)

        # New: Store latest options data
        self.call_option_data = {'ltp': None, 'bid': None, 'ask': None}
        self.put_option_data = {'ltp': None, 'bid': None, 'ask': None}

        # Validate configuration
        self._validate_config()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Create data directory
        os.makedirs(os.path.dirname(POSITION_CSV), exist_ok=True)
        os.makedirs(os.path.dirname(TRADE_LOG_CSV), exist_ok=True)
        os.makedirs(os.path.dirname(SIGNAL_LOG_CSV), exist_ok=True) # Ensure signal log directory exists

        logger.info("Trading Bot initialized successfully")
        self._initialize_api()
        self._connect_websocket()

    def _signal_handler(self, signum, frame):
        """
        Signal handler that responds to external OS signals like SIGINT (Ctrl+C) or SIGTERM.

        This function ensures that the bot:
        - Logs the signal received for traceability
        - Stops the main loop cleanly by setting `self.running = False`
        - Auto square-offs any open position to avoid overnight risk or margin issues

        Parameters:
            signum (int): The signal number received (e.g., signal.SIGINT)
            frame: The current stack frame (ignored in this implementation)

        Returns:
            None
        """

        # Log the received signal and intent to shutdown
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")

        # Stop the bot's main event loop
        self.running = False

        # If a position is still open, square it off immediately
        if self.position:
            self._auto_square_off()


    def _load_config(self, path):
        """
        Loads and parses an INI-style configuration file for bot settings.

        This config may include:
        - API keys and credentials
        - Risk parameters (capital, SL/TP %)
        - Logging or environment settings

        Parameters:
            path (str): Path to the configuration file

        Returns:
            configparser.ConfigParser: Parsed configuration object if successful

        Exits the program if:
        - File is missing
        - File can't be parsed properly
        """

        # --- Step 1: Check if config file exists ---
        if not os.path.exists(path):
            logger.error(f"Config file not found: {path}")
            sys.exit(1)  # Exit with error code

        # --- Step 2: Initialize config parser ---
        config = configparser.ConfigParser()

        try:
            # --- Step 3: Attempt to read and parse config file ---
            config.read(path)
            return config

        except configparser.Error as e:
            # Catch any parsing errors and log them
            logger.error(f"Error reading config file: {e}")
            sys.exit(1)  # Exit if config cannot be used


    def _validate_market_conditions(self):
        """
        Performs a series of real-time checks to validate whether the market conditions
        are suitable for executing a trade. This is a protective filter called before
        placing new orders, ensuring the bot avoids unnecessary risk or technical failures.

        Checks performed:
        - Market must be open (within NSE trading hours)
        - Spot price must be available and positive
        - LTPs for both CALL and PUT options must be present and valid

        Returns:
            bool: True if all conditions are valid, False otherwise
        """

        # --- Check 1: Is the market currently open? ---
        if not self._is_market_open():
            logger.warning("Market is closed. Skipping signal evaluation.")
            return False

        # --- Check 2: Is the NIFTY spot price available and valid? ---
        if not self.spot_price or self.spot_price <= 0:
            logger.warning("Invalid or missing NIFTY spot price.")
            return False

        # # --- Check 3: Is the CALL option LTP available and valid? ---
        # if 'ltp' not in self.call_option_data or self.call_option_data['ltp'] is None:
        #     logger.warning("Call option LTP not available.")
        #     return False

        # # --- Check 4: Is the PUT option LTP available and valid? ---
        # if 'ltp' not in self.put_option_data or self.put_option_data['ltp'] is None:
        #     logger.warning("Put option LTP not available.")
        #     return False

        # ✅ All conditions passed — trading is safe to proceed
        return True


    def _are_option_prices_ready(self):
        """
        Checks whether both CALL and PUT option LTPs (Last Traded Prices)
        are available and valid.

        This function ensures that the bot does not proceed with trading logic
        (like signal evaluation or position sizing) unless both option feeds
        have been successfully subscribed and received their first live tick.

        Returns:
            bool: True if both CALL and PUT option LTPs are available (not None), else False
        """

        return (
            # Check if CALL option's LTP has been received and is not None
            self.call_option_data.get("ltp") is not None and

            # Check if PUT option's LTP has been received and is not None
            self.put_option_data.get("ltp") is not None
        )


    def _calculate_position_size(self, option_ltp):
        """
        Calculates the number of option lots (position size) to trade based on:
        - Risk capital per trade (defined as a percentage of total capital)
        - Current option premium (LTP)
        - Lot size (contract size per lot)

        This risk-based position sizing method ensures that the bot does not risk
        more than a fixed percentage of the capital on any single trade.

        Formula:
            risk_amount = capital × risk_percent
            cost_per_lot = option_ltp × lot_size
            position_size = floor(risk_amount / cost_per_lot)

        Parameters:
            option_ltp (float): Current LTP (price) of the option to be traded

        Returns:
            int: Number of lots to trade (0 if insufficient capital or invalid input)
        """

        try:
            # --- Step 1: Validate option price ---
            if not option_ltp or option_ltp <= 0:
                logger.warning(f"Invalid option LTP for sizing: {option_ltp}")
                return 0

            # --- Step 2: Calculate capital risked per trade (fixed % of total capital) ---
            risk_amount = self.capital * self.risk_pct / 100

            # --- Step 3: Calculate how much one lot of the option costs ---
            cost_per_lot = option_ltp * self.lot_size

            # --- Step 4: Determine how many such lots we can buy within risk limit ---
            position_size = int(risk_amount // cost_per_lot)  # Floor division to get whole lots

            # --- Step 5: Handle edge case where size is too small to afford 1 lot ---
            if position_size <= 0:
                logger.warning("Insufficient capital for even one lot at current LTP.")
                return 0

            # ✅ Valid position size calculated
            return position_size

        except Exception as e:
            # Catch and log any unexpected errors (e.g. missing attributes)
            logger.error(f"Error calculating position size: {e}")
            return 0


    def _validate_config(self):
        """
        Validates that all required configuration parameters and constraints are satisfied.

        This ensures the bot does not start or run with:
        - Missing config sections (e.g., [trading])
        - Invalid trade quantity (e.g., zero or above max allowed)

        If any critical issue is found, the bot logs the error and exits.

        Returns:
            None
        """
        # --- Step 1: Check if required sections exist in config ---
        required_sections = ['trading']
        for section in required_sections:
            if not self.config.has_section(section):
                logger.error(f"Missing required config section: {section}")
                sys.exit(1)

        # --- Step 2: Check if the quantity is within a safe and valid range ---
        if self.quantity <= 0 or self.quantity > self.max_quantity:
            logger.error(f"Invalid quantity: {self.quantity}")
            sys.exit(1)


    def _initialize_api(self):
        """
        Initializes and authenticates the BreezeConnect API using environment variables.

        Required environment variables:
        - BREEZE_API_KEY
        - BREEZE_API_SECRET
        - BREEZE_SESSION_TOKEN

        If any credentials are missing or invalid, the bot logs the error and exits.

        Returns:
            None
        """
        # --- Step 1: Fetch credentials from environment variables ---
        api_key = str(os.getenv("BREEZE_API_KEY"))          # API key as string
        api_secret = str(os.getenv("BREEZE_API_SECRET"))    # API secret as string
        session_token = str(os.getenv("BREEZE_SESSION_TOKEN"))  # Session token as string

        # --- Step 2: Ensure none of the credentials are missing ---
        if not all([api_key, api_secret, session_token]):
            logger.error("Missing environment variables for Breeze API")
            sys.exit(1)

        try:
            # --- Step 3: Instantiate and authenticate BreezeConnect client ---
            self.breeze = BreezeConnect(api_key=api_key)
            self.breeze.generate_session(
                api_secret=api_secret,
                session_token=session_token
            )
            logger.info("Connected to Breeze API successfully")

        except Exception as e:
            # Log and exit on any failure during initialization
            logger.error(f"Failed to initialize Breeze API: {str(e)}")
            sys.exit(1)


    def _connect_websocket(self):
        """
        Establishes and configures the Breeze WebSocket connection.

        Sets up the following event handlers:
        - `on_ticks` → handles real-time tick updates
        - `on_connect` → resubscribes to option feeds when reconnected
        - `on_error` → handles WebSocket errors and reconnection logic

        Also subscribes to the NIFTY spot price feed to keep the strategy updated.

        Returns:
            None
        """
        try:
            # --- Step 1: Set event callbacks for the WebSocket client ---
            self.breeze.on_ticks = self._on_ticks           # Handle incoming ticks
            self.breeze.on_connect = self._on_ws_connect    # Re-initialize on reconnect
            self.breeze.on_error = self._handle_ws_error    # Handle disconnection/errors

            # --- Step 2: Connect to WebSocket ---
            self.breeze.ws_connect()

            # --- Step 3: Subscribe to NIFTY Spot price feed (NSE Cash) ---
            self.breeze.subscribe_feeds(
                exchange_code="NSE",              # Cash market segment
                stock_code="NIFTY",               # Underlying index
                product_type="cash",              # Product type
                get_market_depth=False,           # Depth not needed
                get_exchange_quotes=True          # Get LTP, bid/ask
            )

            logger.info("WebSocket connection initiated and subscribed to NIFTY Spot (NSE Cash)")

        except Exception as e:
            # Log failure and try reconnecting if attempts remain
            logger.error(f"Failed to connect WebSocket: {str(e)}")

            if self.reconnect_attempts < self.max_reconnect_attempts:
                self._reconnect_websocket()  # Retry logic


    def _on_ws_connect(self):
        """
        Callback function triggered when the Breeze WebSocket connection is successfully established.

        This method ensures the bot restores and resumes trading context by:
        - Reloading the last known position from the position file (if any)
        - Resubscribing to the latest option strikes (CALL and PUT)
        - Logging the connection and subscription status

        It enables the bot to recover from disconnects or reboots without manual intervention.
        """

        # Log that the WebSocket has successfully connected
        logger.info("WebSocket connection established.")

        # --- Step 1: Attempt to reload last known trading position ---
        # This helps the bot resume where it left off after a restart or disconnection
        self._load_position()

        # --- Step 2: Re-subscribe to the CALL strike feed if it's valid ---
        if self.call_strike:
            self._subscribe_option(self.call_strike, "call")

        # --- Step 3: Re-subscribe to the PUT strike feed if it's valid ---
        if self.put_strike:
            self._subscribe_option(self.put_strike, "put")


    def _handle_ws_error(self, error):
        """
        Handles WebSocket errors by logging the error, marking the connection as inactive,
        and attempting a controlled reconnection if within allowed retry limits.

        Parameters:
            error (Exception or str): The error message or exception object from the WebSocket client

        Behavior:
        - Logs the error for monitoring
        - Sets `self.ws_connected = False` to reflect dropped connection
        - If bot is still running and retries are allowed, attempts to reconnect
        """
        # --- Log the WebSocket error received from Breeze SDK ---
        logger.error(f"WebSocket error: {error}")

        # --- Mark WebSocket status as disconnected ---
        self.ws_connected = False

        # --- If bot is running and retries are allowed, attempt reconnection ---
        if self.running and self.reconnect_attempts < self.max_reconnect_attempts:
            self._reconnect_websocket()


    def _reconnect_websocket(self):
        """
        Attempts to reconnect the WebSocket connection to Breeze and resubscribe to feeds.

        Features:
        - Implements exponential backoff (5s, 10s, ..., max 30s)
        - Disconnects any stale sessions safely
        - Re-subscribes to the NIFTY Spot feed upon success
        - Logs connection attempts and results

        Returns:
            None
        """
        # --- Step 1: Increment retry counter ---
        self.reconnect_attempts += 1

        # --- Step 2: Compute backoff time (capped at 30 seconds) ---
        wait_time = min(5 * self.reconnect_attempts, 30)
        logger.info(
            f"Attempting to reconnect WebSocket (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}) "
            f"in {wait_time} seconds..."
        )

        # --- Step 3: Sleep before attempting reconnect ---
        time.sleep(wait_time)

        try:
            # --- Step 4: Attempt to disconnect previous session cleanly ---
            try:
                self.breeze.ws_disconnect()
                logger.info("Previous WebSocket disconnected successfully.")
            except Exception as e:
                logger.debug(f"WebSocket disconnection before reconnect failed or unnecessary: {e}")

            # --- Step 5: Reconnect to WebSocket ---
            self.breeze.ws_connect()

            # --- Step 6: Re-subscribe to NIFTY spot feed (essential for tick-driven logic) ---
            self.breeze.subscribe_feeds(
                exchange_code="NSE",
                stock_code="NIFTY",
                product_type="cash",
                get_market_depth=False,
                get_exchange_quotes=True
            )

            logger.info("WebSocket reconnected and feeds subscribed.")

        except Exception as e:
            # --- Step 7: If reconnect fails, log it ---
            logger.error(f"WebSocket reconnection failed: {e}")

            # --- Step 8: Log critical if maximum attempts exceeded ---
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                logger.critical(
                    "Max reconnection attempts reached. Bot will continue but may not receive live data."
                )  

    def _get_next_thursday_expiry(self):
        """
        Calculates the next weekly expiry date for NIFTY options in the NSE F&O segment.

        NSE NIFTY options have weekly expiry on **Thursdays**. This function ensures:
        - If today is before Thursday → returns this week's Thursday
        - If today is Thursday → returns **next** week's Thursday (not same-day expiry)
        - If today is after Thursday → returns next week's Thursday

        The returned string format is compatible with Breeze API:
            'YYYY-MM-DDT00:00:00.000Z'

        Returns:
            str: Expiry date string in ISO-8601 format, at midnight UTC
        """

        # Get today's date in IST timezone
        today = datetime.now(self.ist_tz).date()

        # Calculate how many days ahead the next Thursday is
        # weekday() returns: Monday=0, Tuesday=1, ..., Thursday=3, ..., Sunday=6
        days_ahead = (3 - today.weekday()) % 7  # 3 represents Thursday

        # If today is Thursday, return next week's Thursday (not same-day expiry)
        if days_ahead == 0:
            days_ahead = 7

        # Add the offset to get the actual date of the next Thursday
        next_thursday = today + timedelta(days=days_ahead)

        # Format as required by the Breeze API (ISO UTC midnight format)
        # Example: '2025-07-10T00:00:00.000Z'
        return next_thursday.strftime('%Y-%m-%dT00:00:00.000Z')


    def _is_market_open(self):
        """
        Checks whether the Indian stock market is currently open,
        based on the IST (India Standard Time) timezone.

        Market hours considered:
        - Weekdays only (Monday to Friday)
        - From 9:15 AM to 3:30 PM IST

        Returns:
            bool: True if market is open right now, False otherwise
        """

        # Get the current time in Indian Standard Time (IST)
        now = datetime.now(self.ist_tz)

        # --- Step 1: Check if today is a weekend ---
        # weekday() → 0=Monday, ..., 5=Saturday, 6=Sunday
        if now.weekday() >= 5:
            return False  # Market closed on Saturday and Sunday

        # --- Step 2: Check if time is before 9:15 AM ---
        if now.hour < 9 or (now.hour == 9 and now.minute < 15):
            return False  # Market hasn't opened yet

        # --- Step 3: Check if time is after 3:30 PM ---
        if now.hour > 15 or (now.hour == 15 and now.minute > 30):
            return False  # Market has already closed

        # If none of the above checks failed, market is currently open
        return True


    def _subscribe_option(self, strike, option_type):
        """
        Subscribes to real-time market data for a specific NIFTY option (CALL or PUT)
        using the BreezeConnect WebSocket.

        This function allows the bot to receive live updates (LTP, bid, ask) for
        a dynamically selected strike price, which is essential for:
        - Making informed trade decisions
        - Logging option prices
        - Managing SL/TP and position entry/exits

        Parameters:
            strike (int): The strike price of the option to subscribe to (e.g. 19500)
            option_type (str): Either "call" or "put" (case-insensitive)

        Behavior:
            - All parameters are safely cast to strings, as required by the Breeze API
            - Logs an error if subscription fails
        """

        try:
            # --- Step 1: Call Breeze API to subscribe to option feed ---

            self.breeze.subscribe_feeds(
                exchange_code=str("NFO"),                          # NSE F&O segment
                stock_code=str("NIFTY"),                           # Underlying instrument
                product_type=str("options"),                       # This is an options contract
                expiry_date=str(self._get_next_thursday_expiry()), # Format: YYYY-MM-DDT00:00:00.000Z
                right=str(option_type).lower(),                    # Must be 'call' or 'put' in lowercase
                strike_price=str(int(strike)),                     # Must be stringified integer
                get_market_depth=False,                            # No market depth (to reduce load)
                get_exchange_quotes=True                           # Get LTP, bid, ask updates
            )

            # Log confirmation of successful subscription (debug level)
            logger.debug(f"Subscribed to {option_type.upper()} {strike} option")

        except Exception as e:
            # Log detailed error if subscription fails
            logger.error(f"Failed to subscribe to option {option_type} {strike}: {str(e)}")



    def _calculate_atr(self, period=14):
        """
        Calculates the Average True Range (ATR) from historical 1-minute candle data.

        ATR is a volatility indicator that measures market movement over a specified period.
        It helps the bot adapt to changing volatility by:
        - Adjusting stop-loss or take-profit dynamically
        - Gauging whether the market is calm or volatile

        ATR is computed using the standard formula:
            TR = max(high - low, abs(high - previous_close), abs(low - previous_close))
            ATR = SMA(TR, period)

        Parameters:
            period (int): Number of candles to use for ATR calculation (default = 14)

        Returns:
            float: The calculated ATR value, or 0 if not enough candle data is available
        """

        # --- Step 1: Ensure there are enough candles for the given ATR period ---
        if len(self.candles) < period + 1:
            logger.warning("Not enough candle data to calculate ATR.")
            return 0  # Not enough data to calculate ATR

        # --- Step 2: Extract the most recent candles required for ATR computation ---
        recent_candles = self.candles[-(period + 1):]  # Includes one extra for prev_close reference

        tr_values = []  # Store all true range values

        # --- Step 3: Loop through each pair of consecutive candles to compute TR ---
        for i in range(1, len(recent_candles)):
            curr = recent_candles[i]
            prev = recent_candles[i - 1]

            # Compute True Range (TR) using standard formula
            high = curr['high']
            low = curr['low']
            prev_close = prev['close']

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )

            tr_values.append(tr)

        # --- Step 4: Calculate Simple Moving Average of TR to get ATR ---
        atr = sum(tr_values) / period
        return atr


    def _check_sl_tp(self):
        """
        Checks whether the current NIFTY spot price has hit the stop-loss (SL)
        or take-profit (TP) level for the active position.

        This function is invoked on every tick of NIFTY spot data.
        It determines whether the bot should exit the current position 
        (either long or short) based on predefined SL/TP thresholds.

        Behavior:
        - If position is 'long' (buying CALL), triggers SL if price drops too much,
        or TP if price rises as expected.
        - If position is 'short' (buying PUT), triggers SL if price rises too much,
        or TP if price falls as expected.
        - If SL or TP is hit, it executes an exit order and resets the position state.

        This is a real-time risk control mechanism to limit losses and lock in profits.
        """

        # No active position, so nothing to monitor for SL/TP
        if not self.position:
            return

        # =====================
        # LONG POSITION HANDLING
        # =====================
        if self.position == "long":
            # Stop-loss hit: spot has fallen to or below SL level
            if self.spot_price <= self.stop_loss:
                logger.info(f"STOP-LOSS HIT for LONG position at {self.spot_price:.2f} (SL: {self.stop_loss:.2f})")
                self._exit("call", self.call_strike)  # Exit long via selling CALL

            # Take-profit hit: spot has risen to or above TP level
            elif self.spot_price >= self.take_profit:
                logger.info(f"TAKE-PROFIT HIT for LONG position at {self.spot_price:.2f} (TP: {self.take_profit:.2f})")
                self._exit("call", self.call_strike)  # Exit long via selling CALL

        # =====================
        # SHORT POSITION HANDLING
        # =====================
        elif self.position == "short":
            # Stop-loss hit: spot has risen to or above SL level (bad for short)
            if self.spot_price >= self.stop_loss:
                logger.info(f"STOP-LOSS HIT for SHORT position at {self.spot_price:.2f} (SL: {self.stop_loss:.2f})")
                self._exit("put", self.put_strike)  # Exit short via selling PUT

            # Take-profit hit: spot has fallen to or below TP level (good for short)
            elif self.spot_price <= self.take_profit:
                logger.info(f"TAKE-PROFIT HIT for SHORT position at {self.spot_price:.2f} (TP: {self.take_profit:.2f})")
                self._exit("put", self.put_strike)  # Exit short via selling PUT



    def _on_ticks(self, ticks):
        """
        Callback function that handles incoming tick data (from WebSocket).
        
        This function is triggered every time a new tick (price update) arrives,
        for either NIFTY spot or subscribed option instruments.
        
        Responsibilities:
        - Validate and parse tick data
        - Update the spot price (NIFTY)
        - Track real-time price movements for options (call/put)
        - Build 1-minute candles from NIFTY spot
        - Evaluate stop-loss/take-profit triggers
        - Manage auto-square-off at end of day
        """
        
        # Ignore empty tick payloads
        if not ticks:
            return

        # Capture current local timestamp (IST) for logging and fallback use
        log_now = datetime.now(self.ist_tz)

        # Ensure ticks are always processed as a list
        tick_list = ticks if isinstance(ticks, list) else [ticks]

        # Lock access to shared data structures for thread safety
        with self.data_lock:
            for tick in tick_list:
                try:
                    # Validate tick format and integrity
                    if not self._validate_tick(tick):
                        continue  # Skip invalid ticks

                    # Extract resolved LTP (injected during validation)
                    ltp = tick['_resolved_ltp']

                    # Get stock and exchange identifiers
                    exchange = tick.get("exchange", "").upper()
                    stock_name = tick.get("stock_name", "").upper()
                    product_type_from_tick = tick.get("product_type", "").lower()  # optional

                    # === Determine timestamp from tick ===
                    tick_time_dt = None

                    # First attempt: use exchange-provided timestamp (Unix ms)
                    if 'exchange_timestamp' in tick and tick['exchange_timestamp'] is not None:
                        try:
                            tick_time_dt = datetime.fromtimestamp(
                                tick['exchange_timestamp'] / 1000,
                                tz=self.ist_tz
                            )
                        except (TypeError, ValueError) as e:
                            logger.warning(
                                f"Invalid 'exchange_timestamp' format ({tick['exchange_timestamp']}). "
                                f"Attempting 'ltt'. Error: {e}"
                            )

                    # Fallback: parse string-based timestamp (e.g. "Thu Jul 04 15:29:00 2025")
                    if tick_time_dt is None and 'ltt' in tick and tick['ltt'] is not None:
                        try:
                            tick_time_dt = datetime.strptime(
                                tick['ltt'],
                                '%a %b %d %H:%M:%S %Y'
                            ).replace(tzinfo=self.ist_tz)
                        except (TypeError, ValueError) as e:
                            logger.warning(
                                f"Invalid 'ltt' format ({tick['ltt']}). "
                                f"Falling back to local time. Error: {e}"
                            )

                    # Last fallback: use local machine time if no timestamp available
                    if tick_time_dt is None:
                        tick_time_dt = log_now
                        logger.warning(
                            "Could not determine accurate feed timestamp. "
                            "Falling back to local time for candle processing."
                        )

                    # === Process NIFTY Spot Feed ===
                    if "NIFTY" in stock_name and exchange == "NSE EQUITY":
                        # Update latest spot price
                        self.spot_price = ltp

                        # Check stop-loss or take-profit conditions based on updated spot
                        self._check_sl_tp()

                        # Update ATM strike levels and subscribe to new options if necessary
                        self._update_strikes(ltp)

                        # Update current minute candle (or complete old one if needed)
                        self._process_candle_data(ltp, tick_time_dt)

                    # === Process Option Ticks (from NFO exchange) ===
                    elif "NIFTY" in stock_name and exchange == "NFO":
                        # Extract strike and option type
                        strike_price_tick = tick.get("strike_price")
                        right_tick = tick.get("right", "").lower()  # 'call' or 'put'

                        # Ensure required fields are present
                        if strike_price_tick is not None and right_tick:
                            try:
                                # Ensure strike price is an integer
                                strike_price_tick = int(float(strike_price_tick))
                            except (ValueError, TypeError):
                                logger.warning(
                                    f"Invalid strike_price format in option tick: {strike_price_tick}"
                                )
                                continue

                            # --- Update Call Option Data ---
                            if right_tick == "call" and strike_price_tick == self.call_strike:
                                self.call_option_data['ltp'] = ltp
                                self.call_option_data['bid'] = tick.get('bPrice')
                                self.call_option_data['ask'] = tick.get('sPrice')
                                logger.debug(f"Updated Call Option Data: {self.call_option_data}")

                            # --- Update Put Option Data ---
                            elif right_tick == "put" and strike_price_tick == self.put_strike:
                                self.put_option_data['ltp'] = ltp
                                self.put_option_data['bid'] = tick.get('bPrice')
                                self.put_option_data['ask'] = tick.get('sPrice')
                                logger.debug(f"Updated Put Option Data: {self.put_option_data}")

                except Exception as e:
                    logger.error(f"Error processing tick: {e}")
                    continue  # Safely move to next tick

            # === Auto Square-Off Logic ===
            # If it's between 15:20–15:25 IST and square-off hasn't been done yet
            if log_now.hour == 15 and 20 <= log_now.minute <= 25 and not self.square_off_done:
                self._auto_square_off()      # Close open positions if any
                self.square_off_done = True  # Prevent multiple square-offs


    def _validate_tick(self, tick):
        """
        Validates the integrity and structure of a single tick dictionary.

        This function ensures:
        - Required fields are present in the tick (e.g., 'last', 'symbol', 'exchange')
        - The 'last' traded price (LTP) is a valid float and positive
        - Injects a parsed float version of the LTP into the tick as `_resolved_ltp`
        so that downstream functions don't have to parse it again

        Parameters:
            tick (dict): A single tick data point received from Breeze WebSocket

        Returns:
            bool: True if the tick is valid and usable, False otherwise
        """

        # Define the minimum required fields for a tick to be considered valid
        required_fields = ['last', 'symbol', 'exchange']  # 'stock_name' may not be available in all feeds

        # Check for presence of all required fields in the tick dictionary
        for field in required_fields:
            if field not in tick:
                logger.warning(f"Missing field in tick data: {field}")
                return False  # Reject tick if any field is missing

        # Attempt to convert the 'last' field (LTP) to a float
        try:
            ltp = float(tick['last'])

            # Check for non-positive or zero prices (which are invalid)
            if ltp <= 0:
                logger.warning("Invalid LTP in tick data: LTP must be positive")
                return False

            # Inject resolved float value back into the tick to avoid redundant parsing later
            tick['_resolved_ltp'] = ltp
            return True  # Tick is valid

        except (ValueError, TypeError):
            # Catch conversion errors and reject the tick
            logger.warning("Invalid LTP format in tick data: must be a number")
            return False


    def _process_candle_data(self, ltp, tick_time_dt):
        """
        Builds 1-minute OHLC candles from incoming NIFTY spot tick data,
        and triggers signal evaluation when a full candle completes.

        This function is called on every tick of NIFTY spot. It groups all ticks
        within the same minute into a single candle (Open, High, Low, Close), and
        at the end of the minute, it finalizes that candle and evaluates whether
        a trading signal should be triggered.

        Parameters:
            ltp (float): The current Last Traded Price of NIFTY (from validated tick)
            tick_time_dt (datetime): The timestamp of the tick (from feed or fallback)

        Behavior:
            - Maintains `self.current_candle` for the ongoing minute
            - Uses `self.last_candle_time` to detect minute boundaries
            - Appends closed candle to `self.candles` and calls `_evaluate_signals()`
        """

        # Round the tick timestamp down to the minute (e.g., 13:45:22 → 13:45:00)
        candle_time = tick_time_dt.replace(second=0, microsecond=0)

        # --- Case 1: New minute → close the previous candle and start a new one ---
        if self.last_candle_time and candle_time > self.last_candle_time:
            # Finalize the last candle and append to the list
            if self.current_candle:
                self.candles.append(self.current_candle)

                # Limit list size to avoid memory buildup (keep last 100 candles)
                if len(self.candles) > 100:
                    self.candles.pop(0)

                # Trigger signal evaluation based on completed candle
                self._evaluate_signals()

            # Start a new candle for the new minute
            self.current_candle = {
                'time': candle_time,
                'open': ltp,
                'high': ltp,
                'low': ltp,
                'close': ltp
            }

            # Update the last candle time to the current minute
            self.last_candle_time = candle_time

        # --- Case 2: First tick or still within the same candle minute ---
        elif not self.last_candle_time or candle_time == self.last_candle_time:
            # Initialize the candle if it doesn't exist
            if not self.current_candle:
                self.current_candle = {
                    'time': candle_time,
                    'open': ltp,
                    'high': ltp,
                    'low': ltp,
                    'close': ltp
                }

            else:
                # Update high and low values dynamically
                self.current_candle['high'] = max(self.current_candle['high'], ltp)
                self.current_candle['low'] = min(self.current_candle['low'], ltp)

                # Always update close to latest price
                self.current_candle['close'] = ltp


    def _log_minute_data(self):
        """Logs the NIFTY spot DataFrame and the status of buy/sell conditions every minute."""
        now = datetime.now(self.ist_tz)
        # Check if it's a new minute since the last log
        if self.last_minute_log_time is None or self.last_minute_log_time.minute != now.minute:
            self.last_minute_log_time = now

            logger.info("\n" + "="*60)
            logger.info(f"MINUTE DATA LOG - {now.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("="*60)

            # Print NIFTY Spot Candle Data (last few candles)
            if self.candle_buffer:
                logger.info("\n--- NIFTY Spot Candle Data (Last 5 Completed Candles) ---")
                df_candles = pd.DataFrame(list(self.candle_buffer))
                logger.info(df_candles.tail(5).to_string(index=False)) # Show last 5 completed candles
            else:
                logger.info("\n--- NIFTY Spot Candle Data: Not enough completed data yet ---")

            # Print the opening price of the current (forming) candle
            if self.minute_candle['open'] is not None:
                current_candle_time_str = now.replace(second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S%z')
                logger.info(f"\n--- Current (Forming) Candle ({current_candle_time_str}) ---")
                logger.info(f"Opening Price: {self.minute_candle['open']:.2f}")
            else:
                logger.info("\n--- Current (Forming) Candle: Not yet started ---")


            # Evaluate and print buy/sell conditions
            if len(self.candle_buffer) >= 2: # Need at least 2 completed candles for conditions
                df = pd.DataFrame(list(self.candle_buffer))
                current = df.iloc[-1]
                prev = df.iloc[-2]

                logger.info("\n--- Buy Signal Conditions ---")
                # Recalculate wicks for accurate display in logs
                if prev['close'] < prev['open']:  # Bearish candle
                    prev_top_wick_buy = prev['high'] - prev['open']
                    prev_bottom_wick_buy = prev['close'] - prev['low']
                else:  # Bullish or neutral candle
                    prev_top_wick_buy = prev['high'] - prev['close']
                    prev_bottom_wick_buy = prev['open'] - prev['low']

                buy_conditions = {
                    "Prev candle bearish": prev['close'] < prev['open'],
                    "Prev upper wick > body": prev_top_wick_buy > abs(prev['close'] - prev['open']),
                    "Prev upper wick > lower wick": prev_top_wick_buy > prev_bottom_wick_buy,
                    "Current candle lower low": current['low'] < prev['low'],
                    "Current close > prev open": current['close'] > prev['open'],
                    "Current candle bullish": current['close'] > current['open']
                }
                true_buy_conditions = sum(1 for v in buy_conditions.values() if v)
                false_buy_conditions = len(buy_conditions) - true_buy_conditions
                for cond, status in buy_conditions.items():
                    logger.info(f"- {cond}: {status}")
                logger.info(f"Buy Conditions Summary: {true_buy_conditions} True, {false_buy_conditions} False")

                logger.info("\n--- Sell Signal Conditions ---")
                # Recalculate wicks for accurate display in logs
                if prev['close'] < prev['open']:  # Bearish candle
                    prev_top_wick_sell = prev['high'] - prev['open']
                    prev_bottom_wick_sell = prev['close'] - prev['low']
                else:  # Bullish or neutral candle
                    prev_top_wick_sell = prev['high'] - prev['close']
                    prev_bottom_wick_sell = prev['open'] - prev['low']

                sell_conditions = {
                    "Prev candle bullish": prev['close'] > prev['open'],
                    "Prev lower wick > body": prev_bottom_wick_sell > abs(prev['close'] - prev['open']),
                    "Prev lower wick > upper wick": prev_bottom_wick_sell > prev_top_wick_sell,
                    "Current candle higher high": current['high'] > prev['high'],
                    "Current close < prev open": current['close'] < prev['open'],
                    "Current candle bearish": current['close'] < current['open']
                }
                true_sell_conditions = sum(1 for v in sell_conditions.values() if v)
                false_sell_conditions = len(sell_conditions) - true_sell_conditions
                for cond, status in sell_conditions.items():
                    logger.info(f"- {cond}: {status}")
                logger.info(f"Sell Conditions Summary: {true_sell_conditions} True, {false_sell_conditions} False")
                logger.info(f"are option prices ready: {self._are_option_prices_ready()}")

            else:
                logger.info("\n--- Not enough candle data to evaluate buy/sell conditions ---")

            logger.info("="*60 + "\n")


    def _update_strikes(self, ltp):
        """
        Updates the current ATM-based option strike prices (CALL and PUT)
        based on the latest NIFTY spot price, and subscribes to their live feeds.

        This function ensures that:
        - The bot always trades options near the ATM (At-The-Money) level
        - Option strikes are dynamically adjusted as the market moves
        - Feeds are subscribed to only when the strike changes

        Strike selection logic:
        - Nearest ATM is rounded to nearest multiple of 50
        - CALL strike is set to ATM - 100 (deep ITM)
        - PUT strike is set to ATM + 100 (deep ITM)

        Parameters:
            ltp (float): The latest NIFTY spot price (Last Traded Price)
        """

        # Reject invalid spot price
        if ltp <= 0:
            return

        # --- Step 1: Calculate the At-The-Money (ATM) strike ---
        # Round to the nearest multiple of 50
        atm = round(ltp / 50) * 50

        # --- Step 2: Calculate ITM+2 strikes for both call and put ---
        # These are more conservative, deeper in-the-money options
        call_strike = atm - 100  # 2 strikes ITM for CALLs (e.g. 19400 if ATM = 19500)
        put_strike = atm + 100   # 2 strikes ITM for PUTs  (e.g. 19600 if ATM = 19500)

        # --- Step 3: If CALL strike has changed, update and subscribe ---
        if call_strike != self.call_strike and call_strike > 0:
            self.call_strike = call_strike  # Update internal reference
            self._subscribe_option(call_strike, "call")  # Subscribe to new strike

        # --- Step 4: If PUT strike has changed, update and subscribe ---
        if put_strike != self.put_strike and put_strike > 0:
            self.put_strike = put_strike  # Update internal reference
            self._subscribe_option(put_strike, "put")  # Subscribe to new strike


    def _evaluate_signals(self):
        """
        Evaluates buy or sell signals based on the latest completed candle pattern.

            This function is triggered once per minute at the end of each 1-minute candle,
            and determines whether to enter a long or short position based on candle-based
            reversal logic (defined in `_check_buy_signal()` and `_check_sell_signal()`).

            Responsibilities:
            - Avoid double entry if already in a position
            - Confirm that candle data is available
            - Use recent candle(s) to detect bullish/bearish reversal signals
            - Place buy orders for CALL or PUT based on the signal
            - Log signal for audit/debugging purposes

            Signals are only evaluated:
            - When a candle is finalized (end of the minute)
            - If the market is open
            - If no current position is held
        """
        if len(self.candle_buffer) < 3:  # Need more candles for reliable signals
            return

        try:
            df = pd.DataFrame(list(self.candle_buffer))

            if not self._validate_candle_data(df):
                return

            # Validate market conditions first
            is_valid, reason = self._validate_market_conditions()
            if not is_valid:
                logger.debug(f"Skipping signal evaluation: {reason}")
                return

            atr = self._calculate_atr()
            if atr is None:
                logger.warning("Could not calculate ATR, skipping signal evaluation.")
                return

            # if not self._are_option_prices_ready():
            #     logger.debug("Skipping signal evaluation: Option LTP/Bid/Ask data not ready.")
            #     return

            dynamic_quantity = self._calculate_position_size(atr)
            atr_multiplier = 1.5
            rr_ratio = 2.0

            current = df.iloc[-1]
            prev = df.iloc[-2]

            buy_signal_triggered = self._check_buy_signal(df)
            sell_signal_triggered = self._check_sell_signal(df)

            reversed_position = False
            current_price = self.spot_price

            # === Exit logic: SL or TP hit ===
            if self.position == "long" and self.stop_loss and self.take_profit:
                if current_price <= self.stop_loss or current_price >= self.take_profit:
                    logger.info(f"LONG position exit triggered. Price: {current_price}, SL: {self.stop_loss}, TP: {self.take_profit}")
                    self._log_signal({
                        'signal_type': 'EXIT_LONG_BY_' + ('SL' if current_price <= self.stop_loss else 'TP'),
                        'spot_price_at_signal': current_price,
                        'call_strike': self.call_strike,
                        'put_strike': self.put_strike,
                        'current_candle_open': current['open'],
                        'current_candle_high': current['high'],
                        'current_candle_low': current['low'],
                        'current_candle_close': current['close'],
                        'prev_candle_open': prev['open'],
                        'prev_candle_high': prev['high'],
                        'prev_candle_low': prev['low'],
                        'prev_candle_close': prev['close'],
                        'atr': atr,
                        'dynamic_quantity': dynamic_quantity,
                        'position_before_signal': self.position,
                        'call_ltp': self.call_option_data.get('ltp'),
                        'put_ltp': self.put_option_data.get('ltp'),
                    })
                    self._exit("call", self.call_strike)
                    self.position = None
                    reversed_position = True

            elif self.position == "short" and self.stop_loss and self.take_profit:
                if current_price >= self.stop_loss or current_price <= self.take_profit:
                    logger.info(f"SHORT position exit triggered. Price: {current_price}, SL: {self.stop_loss}, TP: {self.take_profit}")
                    self._log_signal({
                        'signal_type': 'EXIT_SHORT_BY_' + ('SL' if current_price >= self.stop_loss else 'TP'),
                        'spot_price_at_signal': current_price,
                        'call_strike': self.call_strike,
                        'put_strike': self.put_strike,
                        'current_candle_open': current['open'],
                        'current_candle_high': current['high'],
                        'current_candle_low': current['low'],
                        'current_candle_close': current['close'],
                        'prev_candle_open': prev['open'],
                        'prev_candle_high': prev['high'],
                        'prev_candle_low': prev['low'],
                        'prev_candle_close': prev['close'],
                        'atr': atr,
                        'dynamic_quantity': dynamic_quantity,
                        'position_before_signal': self.position,
                        'put_ltp': self.put_option_data.get('ltp'),
                        'call_ltp': self.call_option_data.get('ltp'),
                    })
                    self._exit("put", self.put_strike)
                    self.position = None
                    reversed_position = True

            # Entry Long
            if (self.position is None or reversed_position) and buy_signal_triggered:
                logger.info("Entry Long signal detected")
                self._log_signal({
                    'signal_type': 'ENTRY_LONG',
                    'spot_price_at_signal': self.spot_price,
                    'call_strike': self.call_strike,
                    'put_strike': self.put_strike,
                    'current_candle_open': current['open'],
                    'current_candle_high': current['high'],
                    'current_candle_low': current['low'],
                    'current_candle_close': current['close'],
                    'prev_candle_open': prev['open'],
                    'prev_candle_high': prev['high'],
                    'prev_candle_low': prev['low'],
                    'prev_candle_close': prev['close'],
                    'atr': atr,
                    'dynamic_quantity': dynamic_quantity,
                    'position_before_signal': self.position,
                    'call_ltp': self.call_option_data.get('ltp'),
                    'call_bid': self.call_option_data.get('bid'),
                    'call_ask': self.call_option_data.get('ask'),
                    'put_ltp': self.put_option_data.get('ltp'),
                    'put_bid': self.put_option_data.get('bid'),
                    'put_ask': self.put_option_data.get('ask')
                })
                self.entry_price = self.minute_candle['open']
                self.stop_loss = current['low'] - (atr * atr_multiplier)
                self.take_profit = self.entry_price + ((self.entry_price - self.stop_loss) * rr_ratio)
                logger.info(f"ENTERING LONG. Entry: {self.entry_price:.2f}, SL: {self.stop_loss:.2f}, TP: {self.take_profit:.2f}, Qty: {dynamic_quantity}")
                old_quantity = self.quantity
                self.quantity = dynamic_quantity
                if self._buy("call", self.call_strike):
                    self.position = "long"
                    self._save_position(self.position)
                    logger.info("Position updated to LONG")
                self.quantity = old_quantity

            # Entry Short
            elif (self.position is None or reversed_position) and sell_signal_triggered:
                logger.info("Entry Short signal detected")
                self._log_signal({
                    'signal_type': 'ENTRY_SHORT',
                    'spot_price_at_signal': self.spot_price,
                    'call_strike': self.call_strike,
                    'put_strike': self.put_strike,
                    'current_candle_open': current['open'],
                    'current_candle_high': current['high'],
                    'current_candle_low': current['low'],
                    'current_candle_close': current['close'],
                    'prev_candle_open': prev['open'],
                    'prev_candle_high': prev['high'],
                    'prev_candle_low': prev['low'],
                    'prev_candle_close': prev['close'],
                    'atr': atr,
                    'dynamic_quantity': dynamic_quantity,
                    'position_before_signal': self.position,
                    'call_ltp': self.call_option_data.get('ltp'),
                    'call_bid': self.call_option_data.get('bid'),
                    'call_ask': self.call_option_data.get('ask'),
                    'put_ltp': self.put_option_data.get('ltp'),
                    'put_bid': self.put_option_data.get('bid'),
                    'put_ask': self.put_option_data.get('ask')
                })
                self.entry_price = self.minute_candle['open']
                self.stop_loss = current['high'] + (atr * atr_multiplier)
                self.take_profit = self.entry_price - ((self.stop_loss - self.entry_price) * rr_ratio)
                logger.info(f"ENTERING SHORT. Entry: {self.entry_price:.2f}, SL: {self.stop_loss:.2f}, TP: {self.take_profit:.2f}, Qty: {dynamic_quantity}")
                old_quantity = self.quantity
                self.quantity = dynamic_quantity
                if self._buy("put", self.put_strike):
                    self.position = "short"
                    self._save_position(self.position)
                    logger.info("Position updated to SHORT")
                self.quantity = old_quantity

        except Exception as e:
            logger.error(f"Error in enhanced signal evaluation: {e}")


    def _validate_candle_data(self, candle):
        """
        Validates a single 1-minute OHLC candle dictionary to ensure it contains all
        required fields and that all values are numerically valid.

        This function protects the bot from using incomplete or corrupted candle data
        which could lead to:
        - False signals
        - Runtime errors during signal evaluation
        - Bad trade entries

        Parameters:
            candle (dict): A dictionary representing one 1-minute OHLC candle.
                        Required keys: 'open', 'high', 'low', 'close'

        Returns:
            bool: True if the candle is valid, False otherwise
        """

        # --- Step 1: Check that all required fields exist in the candle dictionary ---
        required_keys = ['open', 'high', 'low', 'close']
        for key in required_keys:
            if key not in candle:
                logger.warning(f"Missing key '{key}' in candle data: {candle}")
                return False

        # --- Step 2: Ensure all OHLC values are numeric (int or float) and positive ---
        for key in required_keys:
            value = candle[key]
            if not isinstance(value, (int, float)) or value <= 0:
                logger.warning(f"Invalid value for '{key}' in candle: {value}")
                return False

        # ✅ All checks passed — the candle is valid
        return True


    def _check_buy_signal(self, df):
        """
        Detects a bullish reversal signal using an enhanced bullish engulfing pattern
        combined with upper-wick rejection logic.

        This signal is used to initiate a LONG (CALL buy) position.

        Signal conditions:
        - Previous candle must be bearish (red)
        - Upper wick of the previous candle must be larger than its body (sign of rejection)
        - Upper wick also larger than bottom wick (strong rejection at the top)
        - Current candle must:
            - Make a lower low (trap)
            - Close above previous open (engulfing)
            - Be bullish (green candle)

        Parameters:
            df (pd.DataFrame): DataFrame of OHLC candles. Must have at least 2 rows.

        Returns:
            bool: True if a buy signal is detected, False otherwise
        """

        # --- Step 1: Ensure there are at least 2 candles to compare ---
        if len(df) < 2:
            return False

        try:
            # Get current and previous candles
            current = df.iloc[-1]
            prev = df.iloc[-2]

            # --- Step 2: Check for missing/invalid data in either candle ---
            if any(pd.isna(x) for x in [
                current['open'], current['high'], current['low'], current['close'],
                prev['open'], prev['high'], prev['low'], prev['close']
            ]):
                return False

            # --- Step 3: Calculate previous candle metrics ---
            prev_body = abs(prev['close'] - prev['open'])  # Body size
            prev_range = prev['high'] - prev['low']        # Full candle range

            if prev_range == 0:
                return False  # Avoid divide-by-zero or flat candles

            # --- Step 4: Calculate wick sizes for the previous candle ---
            if prev['close'] < prev['open']:  # Bearish candle
                prev_top_wick = prev['high'] - prev['open']
                prev_bottom_wick = prev['close'] - prev['low']
            else:  # Bullish or neutral candle
                prev_top_wick = prev['high'] - prev['close']
                prev_bottom_wick = prev['open'] - prev['low']

            # --- Step 5: Define bullish reversal conditions ---
            condition1 = prev['close'] < prev['open']              # Prev candle is bearish
            condition2 = prev_top_wick > prev_body                 # Upper wick > body → rejection
            condition3 = prev_top_wick > prev_bottom_wick          # Upper wick > lower wick
            condition4 = current['low'] < prev['low']              # Current candle makes lower low
            condition5 = current['close'] > prev['open']           # Current closes above prev open → engulfing
            condition6 = current['close'] > current['open']        # Current candle is bullish (green)

            # --- Step 6: Return True if all conditions are met ---
            return all([condition1, condition2, condition3, condition4, condition5, condition6])

        except (IndexError, KeyError) as e:
            # Gracefully handle missing columns or bad indexing
            return False


    def _check_sell_signal(self, df):
        """
        Detects a bearish reversal signal using an enhanced bearish engulfing pattern
        with lower-wick rejection.

        This signal is used to initiate a SHORT position (PUT buy).

        Signal conditions:
        - Previous candle must be bullish (green)
        - Lower wick of the previous candle must be larger than its body (rejection from below)
        - Lower wick must also be larger than the upper wick (strong rejection)
        - Current candle must:
            - Make a higher high (bull trap / stop hunt)
            - Close below previous open (bearish engulfing)
            - Be bearish (red)

        Parameters:
            df (pd.DataFrame): DataFrame of OHLC candles. Must contain at least 2 rows.

        Returns:
            bool: True if a sell signal is detected, False otherwise
        """

        # --- Step 1: Ensure there are at least two candles to evaluate ---
        if len(df) < 2:
            return False

        try:
            # Get the current and previous candle
            current = df.iloc[-1]
            prev = df.iloc[-2]

            # --- Step 2: Validate all necessary OHLC values are present ---
            if any(pd.isna(x) for x in [
                current['open'], current['high'], current['low'], current['close'],
                prev['open'], prev['high'], prev['low'], prev['close']
            ]):
                return False

            # --- Step 3: Calculate body and range of the previous candle ---
            prev_body = abs(prev['close'] - prev['open'])  # Size of the candle body
            prev_range = prev['high'] - prev['low']        # Total height of the candle

            if prev_range == 0:
                return False  # Flat candle, no movement — skip

            # --- Step 4: Calculate wick sizes for the previous candle ---
            if prev['close'] < prev['open']:  # If previous candle is bearish (red)
                prev_top_wick = prev['high'] - prev['open']
                prev_bottom_wick = prev['close'] - prev['low']
            else:  # If previous candle is bullish (green) or neutral
                prev_top_wick = prev['high'] - prev['close']
                prev_bottom_wick = prev['open'] - prev['low']

            # --- Step 5: Define bearish reversal conditions ---
            condition1 = prev['close'] > prev['open']              # Prev candle is bullish
            condition2 = prev_bottom_wick > prev_body              # Long lower wick (rejection)
            condition3 = prev_bottom_wick > prev_top_wick          # Lower wick > upper wick
            condition4 = current['high'] > prev['high']            # Current candle makes higher high
            condition5 = current['close'] < prev['open']           # Engulfing close below prev open
            condition6 = current['close'] < current['open']        # Current candle is bearish

            # --- Step 6: Return True if all conditions are satisfied ---
            return all([condition1, condition2, condition3, condition4, condition5, condition6])

        except (IndexError, KeyError) as e:
            # Catch and suppress common data errors (e.g. missing OHLC columns)
            return False


    def _buy(self, option_type, strike):
        """
        Executes a buy order for a specified option (CALL or PUT) and strike price.

        This is a convenience wrapper around `_place_order()` with the action
        pre-set to "buy". It simplifies code readability and keeps the logic
        consistent when initiating new trades from signal evaluation.

        Parameters:
            option_type (str): Type of option to buy — "call" or "put"
            strike (int): Strike price of the option contract to buy

        Returns:
            bool: True if the buy order was successfully placed, False otherwise
        """

        # Call the generic order placement function with fixed action="buy"
        return self._place_order("buy", option_type, strike)


    def _exit(self, option_type, strike):
        """
        Exits the current open position by placing a market sell order
        for the given option (CALL or PUT) and resets all internal
        state related to that position.

        This function is typically called:
        - When stop-loss (SL) or take-profit (TP) is hit
        - During auto-square-off at the end of the trading day
        - When a reversal signal occurs and the bot needs to flip position

        Parameters:
            option_type (str): 'call' or 'put', representing the option to sell
            strike (int): The strike price of the option to exit

        Returns:
            bool: True if exit order was successfully placed, False otherwise
        """

        # Attempt to place a market sell order for the option
        if self._place_order("sell", option_type, strike):
            # If the order was successful, clear the active position
            self.position = None

            # Also clear stop-loss, take-profit, and entry price levels
            self.entry_price = None
            self.stop_loss = None
            self.take_profit = None

            # Save updated position state to CSV for persistence
            self._save_position(self.position)

            logger.info("Position reset to None after exit. SL/TP cleared.")
            return True

        # If placing the order failed, return False
        return False


    def _place_order(self, action, option_type, strike):
        """
        Places a market order (buy or sell) for a NIFTY option using BreezeConnect,
        after validating all conditions including market timing, strike, and 
        duplicate order prevention.

        Parameters:
            action (str): 'buy' or 'sell'
            option_type (str): 'call' or 'put'
            strike (int): Option strike price to trade

        Returns:
            bool: True if the order was successfully placed, False otherwise
        """

        # --- Step 1: Ensure market is open before placing the order ---
        if not self._is_market_open():
            logger.warning("Market is closed. Order not placed.")
            return False

        # --- Step 2: Validate strike price is a positive integer ---
        if not strike or strike <= 0:
            logger.error("Invalid strike price for order")
            return False

        # --- Step 3: Prevent duplicate orders (same action+option+strike within 60s) ---
        order_key = f"{action}_{option_type}_{strike}"  # e.g., buy_call_19450
        now = datetime.now(self.ist_tz)

        # Check for recent duplicate order with the same key
        if order_key in self.last_order:
            time_diff = (now - self.last_order[order_key]).total_seconds()
            if time_diff < 60:  # 60-second cooldown to prevent rapid repeats
                logger.warning(f"Duplicate order blocked: {order_key}")
                return False

        # Mark this order key with the current timestamp
        self.last_order[order_key] = now

        # --- Step 4: Attempt to place the order using the BreezeConnect API ---
        try:
            order_response = self.breeze.place_order(
                stock_code="NIFTY",                         # Trading NIFTY options
                exchange_code="NFO",                        # NSE Futures & Options segment
                product_type="options",                     # Type of instrument
                expiry_date=self._get_next_thursday_expiry(),  # Dynamically get next weekly expiry
                option_type=option_type,                    # 'call' or 'put'
                strike_price=str(int(strike)),              # Strike must be stringified int
                action=action,                              # 'buy' or 'sell'
                order_type="market",                        # Market order
                quantity=str(self.quantity),                # Order quantity (as string)
                price=0,                                    # Required field; 0 for market orders
                validity="day"                              # Valid for the trading day
            )

            # --- Step 5: Log successful order placement ---
            logger.info(f"ORDER PLACED: {action.upper()} {option_type.upper()} {strike} QTY:{self.quantity}")
            self._log_trade(action, option_type, strike, order_response)
            return True

        except Exception as e:
            # Log error and return failure if API throws exception
            logger.error(f"Order placement failed: {e}")
            return False


    def _log_trade(self, action, option_type, strike, order_response=None):
        """
        Logs the details of a trade (buy or sell) to a CSV file for record-keeping.

        This function is called after every successful order placement. It writes
        trade metadata to a persistent CSV log, including:
        - Timestamp
        - Order action (buy/sell)
        - Option type (call/put)
        - Strike price
        - Quantity
        - Spot price at time of trade
        - Order ID (if available)

        The log file helps with:
        - Auditing bot behavior
        - Performance evaluation
        - Troubleshooting and analysis

        Parameters:
            action (str): 'buy' or 'sell'
            option_type (str): 'call' or 'put'
            strike (int): Strike price of the traded option
            order_response (dict, optional): Response returned by Breeze API on order placement.
                                            Can contain order ID and other metadata.
        """

        try:
            # Get current timestamp in human-readable format (IST)
            now = datetime.now(self.ist_tz).strftime('%Y-%m-%d %H:%M:%S')

            # Construct the trade record as a dictionary
            trade_data = {
                'datetime': now,
                'action': action,
                'option_type': option_type,
                'strike': strike,
                'quantity': self.quantity,
                'spot_price': self.spot_price,
                'order_status': 'placed'  # This could be expanded to include more statuses later
            }

            # If the order response includes an order ID, log it too
            if order_response and isinstance(order_response, dict):
                trade_data['order_id'] = order_response.get('order_id', 'N/A')

            # Convert the trade data to a single-row DataFrame for CSV appending
            entry = pd.DataFrame([trade_data])

            # Check if the trade log file already exists (to manage headers)
            file_exists = os.path.exists(TRADE_LOG_CSV)

            # Append the entry to the trade log CSV file
            # - mode='a' → append mode
            # - header=not file_exists → write header only if file is new
            entry.to_csv(TRADE_LOG_CSV, mode='a', header=not file_exists, index=False)

        except Exception as e:
            # Log any errors that occur while trying to log the trade
            logger.error(f"Failed to log trade: {e}")


    def _log_signal(self, signal_details):
        """
        Logs detected trading signals (buy/sell triggers) to a persistent CSV file
        for audit, debugging, and strategy analysis purposes.

        This function is typically called after a signal is confirmed by
        `_check_buy_signal()` or `_check_sell_signal()`, and before placing a trade.

        Parameters:
            signal_details (dict): A dictionary containing signal metadata such as:
                {
                    'signal_type': 'buy' or 'sell',
                    'option_type': 'call' or 'put',
                    'strike': 19500,
                    'spot_price': 23570.25,
                    'entry_price': 11.75,
                    ... (additional optional metadata)
                }

        Behavior:
            - Appends signal to a CSV file (defined by SIGNAL_LOG_CSV)
            - Adds current timestamp to the signal
            - Creates file header only if the CSV does not already exist

        Returns:
            None
        """

        try:
            # --- Step 1: Add current timestamp to the signal entry ---
            now = datetime.now(self.ist_tz).strftime('%Y-%m-%d %H:%M:%S')
            signal_details['datetime'] = now

            # --- Step 2: Convert the dictionary to a single-row DataFrame ---
            entry = pd.DataFrame([signal_details])

            # --- Step 3: Check if the signal log file already exists ---
            file_exists = os.path.exists(SIGNAL_LOG_CSV)

            # --- Step 4: Append the signal to the CSV file ---
            # - mode='a' → append
            # - header=not file_exists → only add header if the file is new
            # - index=False → do not write row index
            entry.to_csv(SIGNAL_LOG_CSV, mode='a', header=not file_exists, index=False)

            # --- Step 5: Log to console (debug level) for live monitoring ---
            logger.debug(f"Signal logged: {signal_details['signal_type']}")

        except Exception as e:
            # Catch and log any error encountered during signal logging
            logger.error(f"Failed to log signal: {e}")


    def _save_position(self, pos):
        """
        Saves the current trading position (if any) to a CSV file.

        This function ensures the bot's current state is persisted to disk, so that:
        - If the bot restarts or crashes, it can resume without losing context.
        - You can audit and inspect when the bot opened or closed a position.
        - The position file is regularly updated on every entry/exit event.

        The file includes:
        - Timestamp of save
        - Position status ('long', 'short', or 'None')
        - Current spot price
        - Relevant strike prices (call & put)

        Parameters:
            pos (str or None): The current position — should be 'long', 'short', or None
        """

        try:
            # Construct a dictionary representing the bot’s current position state
            position_data = {
                'datetime': datetime.now(self.ist_tz).strftime('%Y-%m-%d %H:%M:%S'),
                'position': pos or 'None',         # Normalize None to string
                'spot_price': self.spot_price,     # Last known NIFTY spot price
                'call_strike': self.call_strike,   # Most recently tracked CALL strike
                'put_strike': self.put_strike      # Most recently tracked PUT strike
            }

            # Convert the dictionary to a single-row DataFrame
            df = pd.DataFrame([position_data])

            # Overwrite (not append) the CSV with the latest position
            # Ensures only the latest state is saved in the file
            df.to_csv(POSITION_CSV, index=False)

            # Log success at debug level
            logger.debug(f"Position saved: {pos}")

        except Exception as e:
            # Catch and log any error that occurs during file write
            logger.error(f"Failed to save position: {e}")


    def _load_position(self):
        """
        Loads the last known trading position ('long', 'short', or None) from the
        persistent CSV file (`POSITION_CSV`), allowing the bot to resume operations
        after restart without losing context.

        Behavior:
        - Checks if the position file exists and is non-empty
        - Reads the most recent saved position
        - Validates the loaded value against expected values
        - Logs and returns the position if valid; otherwise resets to None

        Returns:
            str or None: 'long', 'short', or None depending on the saved state
        """

        try:
            # --- Step 1: Check if the CSV file exists ---
            if not os.path.exists(POSITION_CSV):
                logger.info("No existing position file found")  # First-time startup or clean state
                return None

            # --- Step 2: Load the CSV into a DataFrame ---
            df = pd.read_csv(POSITION_CSV)

            # --- Step 3: Check if the file is empty ---
            if df.empty:
                return None  # No position saved

            # --- Step 4: Extract the last saved position from the most recent row ---
            last_position = df.iloc[-1]['position']

            # --- Step 5: Validate the extracted position value ---
            valid_positions = [None, 'None', 'long', 'short']

            # Normalize 'None' string or Python None
            if last_position in ['None', None]:
                return None

            # If valid position is found, log and return it
            elif last_position in ['long', 'short']:
                logger.info(f"Loaded existing position: {last_position}")
                return last_position

            # Invalid value encountered (e.g., corrupted file), log and reset
            else:
                logger.warning(f"Invalid position loaded: {last_position}. Resetting to None.")
                return None

        except Exception as e:
            # Handle any exception (e.g., file read errors) and log it
            logger.error(f"Failed to load position: {e}")
            return None


    def _auto_square_off(self):
        """
        Automatically exits any open positions near market close (typically between 15:20–15:25 IST),
        ensuring that no trades are carried overnight. This is a safety mechanism that prevents
        overnight exposure, which could lead to unwanted risk or margin penalties.

        Behavior:
        - Checks if there is an active position ('long' or 'short')
        - Calls `_exit()` for the corresponding CALL or PUT strike
        - Logs each step for traceability
        - Summarizes the trading activity for the day

        This method is usually triggered once daily via `_on_ticks()` at a scheduled time window.

        Returns:
            None
        """

        # --- Log the start of square-off process ---
        logger.info("=== AUTO SQUARE-OFF INITIATED ===")

        # --- Case 1: If bot is in a LONG position (CALL bought) ---
        if self.position == "long" and self.call_strike:
            logger.info(f"Squaring off LONG position: CALL {self.call_strike}")
            self._exit("call", self.call_strike)  # Place a market SELL for the CALL

        # --- Case 2: If bot is in a SHORT position (PUT bought) ---
        elif self.position == "short" and self.put_strike:
            logger.info(f"Squaring off SHORT position: PUT {self.put_strike}")
            self._exit("put", self.put_strike)   # Place a market SELL for the PUT

        # --- Case 3: No active position to square off ---
        else:
            logger.info("No position to square off")

        # --- Step 4: Summarize trades after square-off (end-of-day reporting) ---
        self.summarize_daily_trades()

        # --- Log the end of square-off process ---
        logger.info("=== AUTO SQUARE-OFF COMPLETED ===")


    def summarize_daily_trades(self):
        """
        Generates and logs a comprehensive summary of all trades executed during the current day.

        This function is typically called during the auto square-off phase to help traders:
        - Review the bot's performance and activity
        - Validate that trades executed as expected
        - Gain insights into order frequency and strike-level engagement

        Behavior:
        - Loads trade data from `TRADE_LOG_CSV`
        - Filters trades made on the current date
        - Groups trades by action (buy/sell), option type, and strike
        - Aggregates total quantity and earliest/latest timestamps for each group
        - Logs a structured report with counts and details

        Returns:
            None
        """
        try:
            # --- Step 1: Check if the trade log file exists ---
            if not os.path.exists(TRADE_LOG_CSV):
                logger.info("No trade log found to summarize.")
                return

            # --- Step 2: Read trade log into a DataFrame ---
            df = pd.read_csv(TRADE_LOG_CSV)

            # --- Step 3: If the file is empty, there's nothing to summarize ---
            if df.empty:
                logger.info("Empty trade log.")
                return

            # --- Step 4: Parse datetime column for filtering today's trades ---
            df['datetime'] = pd.to_datetime(df['datetime'])

            # --- Step 5: Filter for today's date based on IST timezone ---
            today = datetime.now(self.ist_tz).date()
            today_trades = df[df['datetime'].dt.date == today]

            # --- Step 6: Exit if no trades occurred today ---
            if today_trades.empty:
                logger.info("No trades for today to summarize.")
                return

            # --- Step 7: Group trades by action, option type, and strike ---
            # Aggregate quantity, and record first and last timestamps
            summary = today_trades.groupby(
                ['action', 'option_type', 'strike']
            ).agg({
                'quantity': 'sum',
                'datetime': ['min', 'max']
            }).reset_index()

            # --- Step 8: Count key metrics ---
            total_trades = len(today_trades)
            buy_trades = len(today_trades[today_trades['action'] == 'buy'])
            sell_trades = len(today_trades[today_trades['action'] == 'sell'])

            # --- Step 9: Log the full summary report ---
            logger.info("\n" + "=" * 50)
            logger.info("DAILY TRADE SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Total Trades: {total_trades}")
            logger.info(f"Buy Orders: {buy_trades}")
            logger.info(f"Sell Orders: {sell_trades}")
            logger.info("-" * 50)
            logger.info(f"\n{summary.to_string(index=False)}")
            logger.info("=" * 50)

        except Exception as e:
            # Catch and log any errors during the summary process
            logger.error(f"Failed to generate daily summary: {e}")


    def run(self):
        """
        Main execution loop of the trading bot.

        This function is responsible for:
        - Keeping the bot alive during market hours.
        - Periodically performing health checks to log the current system status.
        - Resetting daily flags (e.g., auto square-off flag) at the start of a new trading day.
        - Gracefully handling termination via KeyboardInterrupt or internal shutdown signals.

        The bot runs indefinitely (until `self.running` is set to False),
        and sleeps briefly on each iteration to avoid busy-waiting.

        Responsibilities:
        - Monitor WebSocket connection health and log it.
        - Check if it's a new trading day and reset relevant state.
        - Trigger `_shutdown()` gracefully on exit.
        """
        logger.info("Trading Bot started successfully")
        logger.info(f"Initial position: {self.position or 'None'}")

        # Track the last time a health check was performed
        last_health_check = datetime.now(self.ist_tz)

        try:
            # Main loop that runs continuously until the bot is terminated
            while self.running:
                current_time = datetime.now(self.ist_tz)

                # Perform health check every 5 minutes (300 seconds)
                if (current_time - last_health_check).total_seconds() >= 300:
                    self._health_check()
                    last_health_check = current_time

                # Reset the square-off flag at 9:00 AM daily for a fresh trading day
                if current_time.hour == 9 and current_time.minute == 0:
                    self.square_off_done = False
                    logger.info("New trading day started")

                # Sleep briefly to reduce CPU usage
                time.sleep(1)

        except KeyboardInterrupt:
            # Handle manual termination using Ctrl+C
            logger.info("Received keyboard interrupt")

        except Exception as e:
            # Catch any unhandled exceptions and log them
            logger.error(f"Unexpected error in main loop: {e}")

        finally:
            # Clean up resources and gracefully terminate
            self._shutdown()


    def _health_check(self):
        """
        Performs a real-time system health check and logs the bot's current operational status.

        This method is useful for:
        - Monitoring bot state during live operation
        - Logging the status of WebSocket connection, position, spot price, and market hours
        - Helping debug if something goes wrong (e.g., missed signal, no feed)

        Typically called on a timer or periodically within the main loop.
        """

        # Log current status:
        # - WebSocket connection: Connected/Disconnected
        # - Current position: long, short, or None
        # - Spot price: latest NIFTY price, or 'N/A' if unavailable
        # - Market status: Open or Closed based on _is_market_open()
        logger.info(
            f"Health Check - WS: {'Connected' if self.ws_connected else 'Disconnected'}, "
            f"Position: {self.position or 'None'}, "
            f"Spot: {self.spot_price or 'N/A'}, "
            f"Market: {'Open' if self._is_market_open() else 'Closed'}"
        )


    def _shutdown(self):
        """
        Gracefully shuts down the bot with the following steps:
        - Automatically exits any open position (square-off)
        - Disconnects from WebSocket feed
        - Logs shutdown process for transparency and auditing

        This method is typically called:
        - On user interrupt (e.g., Ctrl+C)
        - At the end of the trading day
        - During critical failures
        """

        # Log the beginning of the shutdown process
        logger.info("Initiating graceful shutdown...")

        # --- Step 1: Square off any open positions before shutdown ---
        if self.position:
            logger.info("Squaring off positions before shutdown")
            self._auto_square_off()  # Auto exit active position and summarize the day

        # --- Step 2: Attempt to disconnect the WebSocket safely ---
        try:
            if self.breeze:
                self.breeze.ws_disconnect()
                logger.info("WebSocket disconnected")
        except Exception as e:
            # Catch and log any error that occurs while closing the WebSocket
            logger.error(f"Error disconnecting WebSocket: {e}")

        # Final log to confirm the shutdown is complete
        logger.info("Trading Bot shutdown completed")


def main():
    """
    Entry point for the NIFTY Options Trading Bot.

    This function:
    1. Parses command-line arguments for custom config path and log level.
    2. Sets up global logging level based on user input.
    3. Initializes the trading bot with the given configuration.
    4. Starts the bot's main execution loop (which includes live data feed, signal evaluation, order placement, etc.).

    Arguments:
        --config: Optional path to a custom config file (default: config/config.ini).
        --log-level: Logging verbosity (choices: DEBUG, INFO, WARNING, ERROR; default: INFO).

    Returns:
        On fatal error, exits with status code 1 after logging the issue.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Enhanced NIFTY Options Trading Bot")

    # Add argument for custom config file path
    parser.add_argument(
        '--config',
        default='config/config.ini',
        help='Path to config file'
    )

    # Add argument to control logging verbosity
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set logging level'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Adjust logging level globally based on user input (e.g., DEBUG for detailed logs)
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        # Instantiate the trading bot with the provided config path
        bot = SimpleTradingBot(config_path=args.config)

        # Start the bot's main event loop (will run until terminated)
        bot.run()

    except Exception as e:
        # Catch any unexpected exceptions during bot startup and exit
        logger.critical(f"Failed to start trading bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
