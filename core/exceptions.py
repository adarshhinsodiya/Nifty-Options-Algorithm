"""
Custom exceptions for the trading system.
"""

class TradingError(Exception):
    """Base class for trading-related exceptions."""
    pass

class MaxRetriesExceededError(TradingError):
    """Raised when the maximum number of retries is exceeded."""
    pass

class OrderError(TradingError):
    """Raised when there is an error with order placement or modification."""
    pass

class DataError(TradingError):
    """Raised when there is an error with market data."""
    pass

class SessionError(TradingError):
    """Raised when there is an error with the trading session."""
    pass

class RiskLimitExceededError(TradingError):
    """Raised when a risk limit is exceeded."""
    pass

class InsufficientFundsError(TradingError):
    """Raised when there are insufficient funds to place an order."""
    pass

class InvalidSignalError(TradingError):
    """Raised when an invalid trading signal is detected."""
    pass
