from __future__ import annotations

from core.config import RiskConfig
from core.models import RiskDecision, Signal, SignalType


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def evaluate(self, signal: Signal, current_open_positions: int) -> RiskDecision:
        if signal.signal not in (SignalType.LONG, SignalType.SHORT):
            return RiskDecision(
                allowed=False,
                reason="signal_not_actionable",
                max_position_notional=0.0,
                stop_loss_pct=self.config.default_stop_loss_pct,
                take_profit_pct=self.config.default_take_profit_pct,
            )

        if current_open_positions >= self.config.max_open_positions:
            return RiskDecision(
                allowed=False,
                reason="max_open_positions_reached",
                max_position_notional=0.0,
                stop_loss_pct=self.config.default_stop_loss_pct,
                take_profit_pct=self.config.default_take_profit_pct,
            )

        # Dynamic stop loss based on ATR if available
        atr_val = signal.metadata.get("atr", 0.0)
        close = signal.metadata.get("close", 0.0)
        if atr_val > 0 and close > 0:
            atr_stop_pct = (atr_val * 2.0) / close  # 2x ATR stop
            stop_loss_pct = max(self.config.default_stop_loss_pct, min(atr_stop_pct, 0.08))
        else:
            stop_loss_pct = self.config.default_stop_loss_pct

        # Take profit = 2x stop loss (minimum 2:1 reward-to-risk)
        take_profit_pct = max(self.config.default_take_profit_pct, stop_loss_pct * 2.0)

        # Confidence-based sizing: higher confidence → bigger position
        confidence = signal.metadata.get("confidence_pct", 50.0)
        confidence_mult = max(0.5, min(1.0, confidence / 100.0))

        per_trade_risk_budget = self.config.capital_tl * self.config.risk_per_trade_pct
        max_position_by_capital = self.config.capital_tl * self.config.max_position_pct

        # Position size: risk budget / stop loss percentage (adjusted by confidence)
        max_position_notional = min(
            max_position_by_capital,
            (per_trade_risk_budget / max(stop_loss_pct, 1e-9)) * confidence_mult,
        )

        # Score-based filter: don't risk money on weak signals
        composite_score = abs(signal.score)
        if composite_score < 35:
            return RiskDecision(
                allowed=False,
                reason=f"signal_too_weak (score={composite_score:.1f})",
                max_position_notional=0.0,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )

        return RiskDecision(
            allowed=True,
            reason="approved",
            max_position_notional=round(max_position_notional, 2),
            stop_loss_pct=round(stop_loss_pct, 4),
            take_profit_pct=round(take_profit_pct, 4),
        )
