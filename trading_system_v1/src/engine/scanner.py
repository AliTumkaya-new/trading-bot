from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

import pandas as pd

from core.models import MarketType, Signal, SignalType
from data.base import MarketDataAdapter
from strategies.base import Strategy

logger = logging.getLogger("trading_system")


class MarketScanner:
    def __init__(self, adapter: MarketDataAdapter, strategy: Strategy, market: MarketType) -> None:
        self.adapter = adapter
        self.strategy = strategy
        self.market = market

    def scan(self, symbols: Iterable[str], interval: str, limit: int) -> List[Tuple[Signal, pd.DataFrame]]:
        results: List[Tuple[Signal, pd.DataFrame]] = []
        for symbol in symbols:
            try:
                df = self.adapter.fetch_ohlcv(symbol=symbol, interval=interval, limit=limit)
                signal = self.strategy.generate(symbol=symbol, market=self.market, df=df)
                results.append((signal, df))
            except Exception as e:
                logger.warning("⚠️ %s veri alınamadı, atlanıyor: %s", symbol, e)
                continue
        return sorted(results, key=lambda item: item[0].score, reverse=True)

    @staticmethod
    def actionable(results: List[Tuple[Signal, pd.DataFrame]]) -> List[Tuple[Signal, pd.DataFrame]]:
        return [item for item in results if item[0].signal != SignalType.FLAT]
