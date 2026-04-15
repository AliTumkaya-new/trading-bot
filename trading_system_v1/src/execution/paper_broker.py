from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict

from core.models import Fill, MarketType, OrderRequest, Position, Side


@dataclass
class PaperBroker:
    fee_rate: float = 0.001
    slippage_pct: float = 0.0005  # %0.05 slippage (gerçekçi piyasa etkisi)
    cash: float = 10_000.0
    daily_pnl: float = 0.0        # Günlük P&L takibi
    daily_loss_limit: float = -500.0  # Günlük max zarar (₺)
    positions: Dict[str, Position] = field(default_factory=dict)
    short_positions: Dict[str, dict] = field(default_factory=dict)  # SHORT tracking

    def _apply_slippage(self, price: float, side: Side) -> float:
        """Gerçekçi slippage: alışta fiyat yukarı, satışta aşağı kayar."""
        slip = random.uniform(0, self.slippage_pct)
        if side == Side.BUY:
            return price * (1 + slip)   # Alışta daha pahalıya alırsın
        return price * (1 - slip)       # Satışta daha ucuza satarsın

    def submit_market_order(self, order: OrderRequest, mark_price: float, leverage: int = 1) -> Fill:
        if mark_price <= 0:
            raise ValueError("mark_price must be positive")

        # Günlük zarar limiti kontrolü
        if self.daily_pnl <= self.daily_loss_limit:
            raise ValueError(f"daily_loss_limit reached (₺{self.daily_pnl:.2f})")

        # Slippage uygula
        exec_price = self._apply_slippage(mark_price, order.side)

        notional = order.quantity * exec_price
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
                price=exec_price,
                fee=fee,
            )

            self.short_positions[order.symbol] = {
                "quantity": order.quantity,
                "entry_price": exec_price,
                "market": order.market,
                "leverage": leverage,
            }

            self.cash -= (margin + fee)
            return fill

        # Close SHORT position (BUY to cover)
        if order.side == Side.BUY and order.symbol in self.short_positions:
            short = self.short_positions[order.symbol]
            pnl = (short["entry_price"] - exec_price) * short["quantity"]
            lev = short.get("leverage", 1)
            original_margin = (short["entry_price"] * short["quantity"]) / lev if lev > 1 else 0

            fill = Fill(
                symbol=order.symbol,
                market=order.market,
                side=order.side,
                quantity=short["quantity"],
                price=exec_price,
                fee=fee,
            )

            # Margin geri dön + P&L
            self.daily_pnl += pnl
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
            price=exec_price,
            fee=fee,
        )

        position = self.positions.setdefault(
            order.symbol,
            Position(symbol=order.symbol, market=order.market),
        )

        if order.side == Side.BUY:
            position.leverage = leverage
            position.update_from_fill(fill)
            self.cash -= (margin + fee)
        else:
            # Selling existing LONG — kaldıraçlı pozisyonda margin + PnL geri döner
            pos_lev = position.leverage if position.leverage > 1 else leverage
            entry_notional = position.avg_price * order.quantity
            pnl = (mark_price - position.avg_price) * order.quantity
            locked_margin = entry_notional / pos_lev if pos_lev > 1 else entry_notional
            position.update_from_fill(fill)
            self.daily_pnl += pnl
            self.cash += locked_margin + pnl - fee

        if position.quantity == 0 and position.realized_pnl == 0:
            self.positions.pop(order.symbol, None)

        return fill
