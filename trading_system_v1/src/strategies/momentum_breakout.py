from __future__ import annotations

import pandas as pd

from core.models import MarketType, Signal, SignalType
from strategies.base import Strategy


class MomentumBreakoutStrategy(Strategy):
    name = "momentum_breakout_v1"

    def __init__(self, fast_ma: int = 20, slow_ma: int = 50, breakout_window: int = 20) -> None:
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.breakout_window = breakout_window

    def generate(self, symbol: str, market: MarketType, df: pd.DataFrame) -> Signal:
        if len(df) < max(self.fast_ma, self.slow_ma, self.breakout_window) + 2:
            return Signal(
                symbol=symbol,
                market=market,
                signal=SignalType.FLAT,
                score=0.0,
                strategy_name=self.name,
                metadata={"reason": "not_enough_data"},
            )

        work = df.copy()
        work["fast_ma"] = work["close"].rolling(self.fast_ma).mean()
        work["slow_ma"] = work["close"].rolling(self.slow_ma).mean()
        work["prior_high"] = work["high"].rolling(self.breakout_window).max().shift(1)
        work["returns"] = work["close"].pct_change()
        work["volatility"] = work["returns"].rolling(self.breakout_window).std()

        last = work.iloc[-1]
        close = float(last["close"])
        fast_ma = float(last["fast_ma"])
        slow_ma = float(last["slow_ma"])
        prior_high = float(last["prior_high"])
        volatility = float(last["volatility"]) if pd.notna(last["volatility"]) else 0.0

        if pd.isna(fast_ma) or pd.isna(slow_ma) or pd.isna(prior_high):
            return Signal(
                symbol=symbol,
                market=market,
                signal=SignalType.FLAT,
                score=0.0,
                strategy_name=self.name,
                metadata={"reason": "indicators_not_ready"},
            )

        trend_ok = close > fast_ma > slow_ma
        breakout_ok = close > prior_high

        if trend_ok and breakout_ok:
            # Higher score when breakout exceeds prior range by more and volatility remains contained.
            breakout_strength = (close / prior_high) - 1.0 if prior_high else 0.0
            score = max(0.0, (breakout_strength * 1000) - (volatility * 100))
            return Signal(
                symbol=symbol,
                market=market,
                signal=SignalType.LONG,
                score=round(score, 4),
                strategy_name=self.name,
                metadata={
                    "close": close,
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "prior_high": prior_high,
                    "volatility": volatility,
                    "breakout_strength": breakout_strength,
                },
            )

        return Signal(
            symbol=symbol,
            market=market,
            signal=SignalType.FLAT,
            score=0.0,
            strategy_name=self.name,
            metadata={
                "close": close,
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "prior_high": prior_high,
                "volatility": volatility,
                "reason": "filter_not_passed",
            },
        )
