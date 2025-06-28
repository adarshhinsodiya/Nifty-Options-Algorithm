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
    """Represents a trading signal with enhanced fields for live trading.
    
    Attributes:
        signal_type: Type of signal (LONG/SHORT)
        entry_price: Entry price for the trade
        stop_loss: Stop loss price
        take_profit: Take profit/target price
        strike: Strike price for options
        option_type: 'CE' for call, 'PE' for put
        timestamp: When the signal was generated
        spot_price: Underlying spot price when signal was generated
        expiry_date: Expiry date for the option
        symbol: Trading symbol (e.g., 'NIFTY')
        signal_id: Unique identifier for the signal
        quantity: Number of contracts/shares
        lot_size: Lot size for the instrument
        reason: Reason/description for the signal
        metadata: Additional metadata for the signal
        status: Current status of the signal
        parent_signal_id: ID of parent signal if this is a follow-up signal
    """
    signal_type: TradeSignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    strike: int
    option_type: str  # 'CE' for call, 'PE' for put
    timestamp: datetime
    spot_price: float
    expiry_date: Optional[datetime] = None
    symbol: str = "NIFTY"
    signal_id: str = field(default_factory=lambda: f"sig_{int(datetime.utcnow().timestamp() * 1000)}")
    quantity: int = 1
    lot_size: int = 50  # Default NIFTY lot size
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "PENDING"  # PENDING, PROCESSING, EXECUTED, REJECTED, EXPIRED
    parent_signal_id: Optional[str] = None
    
    def __post_init__(self):
        # Ensure option type is uppercase
        if self.option_type:
            self.option_type = self.option_type.upper()
    
    @property
    def is_call(self) -> bool:
        """Check if this is a call option signal."""
        return self.option_type.upper() == 'CE'
    
    @property
    def is_put(self) -> bool:
        """Check if this is a put option signal."""
        return self.option_type.upper() == 'PE'
    
    @property
    def is_long(self) -> bool:
        """Check if this is a long position signal."""
        return self.signal_type == TradeSignalType.LONG
    
    @property
    def is_short(self) -> bool:
        """Check if this is a short position signal."""
        return self.signal_type == TradeSignalType.SHORT
    
    @property
    def contract_symbol(self) -> str:
        """Generate the full contract symbol (e.g., 'NIFTY25JAN2345000CE')."""
        if not self.expiry_date:
            return f"{self.symbol}{self.strike}{self.option_type}"
            
        # Format: NIFTY + 2-digit day + 3-char month + 2-digit year + strike + option type
        expiry_str = self.expiry_date.strftime("%d%b%y").upper()
        return f"{self.symbol}{expiry_str}{self.strike}{self.option_type}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary with all fields."""
        return {
            'signal_id': self.signal_id,
            'signal_type': self.signal_type.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'strike': self.strike,
            'option_type': self.option_type,
            'timestamp': self.timestamp.isoformat(),
            'spot_price': self.spot_price,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'lot_size': self.lot_size,
            'reason': self.reason,
            'status': self.status,
            'parent_signal_id': self.parent_signal_id,
            'contract_symbol': self.contract_symbol,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeSignal':
        """Create a TradeSignal from a dictionary."""
        from datetime import datetime
        from dateutil.parser import parse
        
        # Handle timestamp conversion
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = parse(timestamp)
        
        # Handle expiry_date conversion
        expiry_date = data.get('expiry_date')
        if isinstance(expiry_date, str):
            expiry_date = parse(expiry_date)
        
        return cls(
            signal_type=TradeSignalType[data['signal_type']],
            entry_price=float(data['entry_price']),
            stop_loss=float(data['stop_loss']),
            take_profit=float(data['take_profit']),
            strike=int(data['strike']),
            option_type=data['option_type'],
            timestamp=timestamp or datetime.utcnow(),
            spot_price=float(data.get('spot_price', 0)),
            expiry_date=expiry_date,
            symbol=data.get('symbol', 'NIFTY'),
            signal_id=data.get('signal_id'),
            quantity=int(data.get('quantity', 1)),
            lot_size=int(data.get('lot_size', 50)),
            reason=data.get('reason'),
            metadata=data.get('metadata', {}),
            status=data.get('status', 'PENDING'),
            parent_signal_id=data.get('parent_signal_id')
        )

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
