from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from core.models import Fill, OrderRequest, Position, Side


@dataclass
class PaperBroker:
    fee_rate: float = 0.001
    cash: float = 10_000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    short_positions: Dict[str, dict] = field(default_factory=dict)  # SHORT tracking

    def submit_market_order(self, order: OrderRequest, mark_price: float) -> Fill:
        if mark_price <= 0:
            raise ValueError("mark_price must be positive")

        notional = order.quantity * mark_price
        fee = notional * self.fee_rate

        # SHORT position handling
        if order.side == Side.SELL and order.symbol not in self.positions:
            # Opening a new SHORT position
            if (notional + fee) > self.cash:
                raise ValueError("insufficient paper cash for short margin")

            fill = Fill(
                symbol=order.symbol,
                market=order.market,
                side=order.side,
                quantity=order.quantity,
                price=mark_price,
                fee=fee,
            )

            self.short_positions[order.symbol] = {
                "quantity": order.quantity,
                "entry_price": mark_price,
                "market": order.market,
            }

            # Lock the margin (notional value)
            self.cash -= fee  # Only deduct fee; margin is conceptual
            return fill

        # Close SHORT position (BUY to cover)
        if order.side == Side.BUY and order.symbol in self.short_positions:
            short = self.short_positions[order.symbol]
            pnl = (short["entry_price"] - mark_price) * short["quantity"]

            fill = Fill(
                symbol=order.symbol,
                market=order.market,
                side=order.side,
                quantity=short["quantity"],
                price=mark_price,
                fee=fee,
            )

            self.cash += pnl - fee
            del self.short_positions[order.symbol]
            return fill

        # Standard LONG handling
        if order.side == Side.BUY and (notional + fee) > self.cash:
            raise ValueError("insufficient paper cash")

        fill = Fill(
            symbol=order.symbol,
            market=order.market,
            side=order.side,
            quantity=order.quantity,
            price=mark_price,
            fee=fee,
        )

        position = self.positions.setdefault(
            order.symbol,
            Position(symbol=order.symbol, market=order.market),
        )
        position.update_from_fill(fill)

        if order.side == Side.BUY:
            self.cash -= notional + fee
        else:
            self.cash += notional - fee

        if position.quantity == 0 and position.realized_pnl == 0:
            self.positions.pop(order.symbol, None)

        return fill
