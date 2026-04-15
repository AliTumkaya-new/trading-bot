from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

import pandas as pd


class MarketDataAdapter(ABC):
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Return OHLCV data with columns: timestamp, open, high, low, close, volume."""

    def fetch_many(self, symbols: Iterable[str], interval: str, limit: int) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            results[symbol] = self.fetch_ohlcv(symbol=symbol, interval=interval, limit=limit)
        return results

    @staticmethod
    def validate_frame(df: pd.DataFrame, required: List[str]) -> None:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if df.empty:
            raise ValueError("Received empty market data frame")
