from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from core.models import Fill, MarketType, OrderRequest, Position, Side


@dataclass
class PaperBroker:
    fee_rate: float = 0.001
    cash: float = 10_000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    short_positions: Dict[str, dict] = field(default_factory=dict)  # SHORT tracking

    def submit_market_order(self, order: OrderRequest, mark_price: float, leverage: int = 1) -> Fill:
        if mark_price <= 0:
            raise ValueError("mark_price must be positive")

        notional = order.quantity * mark_price
        fee = notional * self.fee_rate
        # Kaldıraçlı işlemlerde sadece margin (teminat) düşülür
        margin = notional / leverage if leverage > 1 else notional

        # SHORT position handling
        if order.side == Side.SELL and order.symbol not in self.positions:
            if (margin + fee) > self.cash:
                raise ValueError(f"insufficient paper cash (need ₺{margin + fee:.2f}, have ₺{self.cash:.2f})")

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
                "leverage": leverage,
            }

            self.cash -= (margin + fee)
            return fill

        # Close SHORT position (BUY to cover)
        if order.side == Side.BUY and order.symbol in self.short_positions:
            short = self.short_positions[order.symbol]
            pnl = (short["entry_price"] - mark_price) * short["quantity"]
            lev = short.get("leverage", 1)
            original_margin = (short["entry_price"] * short["quantity"]) / lev if lev > 1 else 0

            fill = Fill(
                symbol=order.symbol,
                market=order.market,
                side=order.side,
                quantity=short["quantity"],
                price=mark_price,
                fee=fee,
            )

            # Margin geri dön + P&L
            self.cash += original_margin + pnl - fee
            del self.short_positions[order.symbol]
            return fill

        # Standard LONG handling
        if order.side == Side.BUY and (margin + fee) > self.cash:
            raise ValueError(f"insufficient paper cash (need ₺{margin + fee:.2f}, have ₺{self.cash:.2f})")

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
            self.cash -= (margin + fee)
        else:
            # Selling existing LONG — margin geri döner
            self.cash += notional - fee

        if position.quantity == 0 and position.realized_pnl == 0:
            self.positions.pop(order.symbol, None)

        return fill
