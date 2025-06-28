"""
Data models for the trading system.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any

class TradeSignalType(Enum):
    """Types of trading signals."""
    LONG = "LONG"
    SHORT = "SHORT"

class OrderType(Enum):
    """Types of orders."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SLM = "SLM"

class OrderStatus(Enum):
    """Order statuses."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class PositionStatus(Enum):
    """Position statuses."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    STOPPED_OUT = "STOPPED_OUT"
    TAKE_PROFIT = "TAKE_PROFIT"
    EOD_EXIT = "EOD_EXIT"

@dataclass
class TradeSignal:
    """Represents a trading signal."""
    signal_type: TradeSignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    strike: int
    option_type: str  # 'ce' for call, 'pe' for put
    timestamp: datetime
    spot_price: float
    expiry_date: Optional[datetime] = None
    symbol: str = "NIFTY"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            'signal_type': self.signal_type.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'strike': self.strike,
            'option_type': self.option_type,
            'timestamp': self.timestamp.isoformat(),
            'spot_price': self.spot_price,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'symbol': self.symbol
        }

@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    order_type: OrderType
    quantity: int
    price: float
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float = 0.0
    order_timestamp: datetime = field(default_factory=datetime.utcnow)
    filled_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'average_price': self.average_price,
            'order_timestamp': self.order_timestamp.isoformat(),
            'filled_timestamp': self.filled_timestamp.isoformat() if self.filled_timestamp else None
        }

@dataclass
class Position:
    """Represents a trading position."""
    position_id: str
    signal: TradeSignal
    entry_order: Order
    exit_order: Optional[Order] = None
    status: PositionStatus = PositionStatus.OPEN
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    
    def calculate_pnl(self) -> None:
        """Calculate P&L for the position."""
        if self.exit_order and self.exit_order.average_price > 0:
            if self.signal.signal_type == TradeSignalType.LONG:
                self.pnl = (self.exit_order.average_price - self.entry_order.average_price) * self.entry_order.quantity
            else:  # SHORT
                self.pnl = (self.entry_order.average_price - self.exit_order.average_price) * self.entry_order.quantity
            
            self.pnl_percentage = (self.pnl / (self.entry_order.average_price * self.entry_order.quantity)) * 100
    
    def close_position(self, exit_order: Order, reason: str) -> None:
        """Close the position with the given exit order."""
        self.exit_order = exit_order
        self.exit_time = exit_order.filled_timestamp or datetime.utcnow()
        self.exit_price = exit_order.average_price
        self.exit_reason = reason
        self.status = PositionStatus[reason.upper()] if hasattr(PositionStatus, reason.upper()) else PositionStatus.CLOSED
        self.calculate_pnl()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'position_id': self.position_id,
            'signal': self.signal.to_dict(),
            'entry_order': self.entry_order.to_dict() if self.entry_order else None,
            'exit_order': self.exit_order.to_dict() if self.exit_order else None,
            'status': self.status.value,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage
        }

@dataclass
class Portfolio:
    """Represents a trading portfolio."""
    initial_capital: float
    current_cash: float
    positions: List[Position] = field(default_factory=list)
    closed_positions: List[Position] = field(default_factory=list)
    
    def add_position(self, position: Position, commission: float = 0.0) -> None:
        """Add a new position to the portfolio.
        
        Args:
            position: Position to add
            commission: Commission rate per trade (as a decimal, e.g., 0.0005 for 0.05%)
        """
        self.positions.append(position)
        trade_value = position.entry_order.average_price * abs(position.entry_order.quantity)
        commission_cost = trade_value * commission
        self.current_cash -= (trade_value + commission_cost)
        position.entry_order.commission = commission_cost  # Store commission for reporting
    
    def close_position(self, position: Position, exit_order: Order, reason: str, commission: float = 0.0) -> None:
        """Close an open position.
        
        Args:
            position: Position to close
            exit_order: Exit order details
            reason: Reason for closing the position
            commission: Commission rate per trade (as a decimal, e.g., 0.0005 for 0.05%)
        """
        if position not in self.positions:
            raise ValueError("Position not found in open positions")
            
        position.close_position(exit_order, reason)
        
        # Calculate and deduct commission
        trade_value = exit_order.average_price * abs(exit_order.quantity)
        commission_cost = trade_value * commission
        exit_order.commission = commission_cost  # Store commission for reporting
        
        # Update portfolio cash (subtract commission from proceeds)
        self.current_cash += (trade_value - commission_cost)
        
        self.positions.remove(position)
        self.closed_positions.append(position)
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [p for p in self.positions if p.status == PositionStatus.OPEN]
    
    def get_closed_positions(self) -> List[Position]:
        """Get all closed positions."""
        return self.closed_positions
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value including open positions."""
        position_value = 0.0
        for position in self.positions:
            symbol = position.signal.symbol
            if symbol in current_prices:
                position_value += current_prices[symbol] * position.entry_order.quantity
        return self.current_cash + position_value
    
    def get_total_pnl(self, include_commissions: bool = True) -> float:
        """Calculate total P&L for all closed positions.
        
        Args:
            include_commissions: Whether to include commission costs in P&L calculation
            
        Returns:
            float: Total P&L, optionally including commission costs
        """
        total_pnl = 0.0
        for position in self.closed_positions:
            if position.pnl is not None:
                pnl = position.pnl
                if include_commissions and hasattr(position.entry_order, 'commission'):
                    pnl -= position.entry_order.commission
                if include_commissions and position.exit_order and hasattr(position.exit_order, 'commission'):
                    pnl -= position.exit_order.commission
                total_pnl += pnl
        return total_pnl
    
    def get_win_rate(self) -> float:
        """Calculate win rate for closed positions."""
        if not self.closed_positions:
            return 0.0
        wins = sum(1 for p in self.closed_positions if p.pnl and p.pnl > 0)
        return (wins / len(self.closed_positions)) * 100
