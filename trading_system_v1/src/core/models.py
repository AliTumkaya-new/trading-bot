from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class MarketType(str, Enum):
    CRYPTO = "crypto"
    BIST = "bist"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class SignalType(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(slots=True)
class Candle:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    market: MarketType


@dataclass(slots=True)
class Signal:
    symbol: str
    market: MarketType
    signal: SignalType
    score: float
    strategy_name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RiskDecision:
    allowed: bool
    reason: str
    max_position_notional: float
    stop_loss_pct: float
    take_profit_pct: float


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    market: MarketType
    side: Side
    quantity: float
    price: Optional[float] = None
    signal_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Fill:
    symbol: str
    market: MarketType
    side: Side
    quantity: float
    price: float
    fee: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class Position:
    symbol: str
    market: MarketType
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0

    def update_from_fill(self, fill: Fill) -> None:
        if fill.side == Side.BUY:
            new_notional = (self.quantity * self.avg_price) + (fill.quantity * fill.price)
            self.quantity += fill.quantity
            self.avg_price = new_notional / self.quantity if self.quantity else 0.0
            self.realized_pnl -= fill.fee
            return

        sell_qty = min(self.quantity, fill.quantity)
        pnl = (fill.price - self.avg_price) * sell_qty
        self.quantity -= sell_qty
        self.realized_pnl += pnl - fill.fee
        if self.quantity <= 0:
            self.quantity = 0.0
            self.avg_price = 0.0
