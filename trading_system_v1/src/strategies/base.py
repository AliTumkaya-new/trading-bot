from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from core.models import MarketType, Signal


class Strategy(ABC):
    name: str = "base_strategy"

    @abstractmethod
    def generate(self, symbol: str, market: MarketType, df: pd.DataFrame) -> Signal:
        raise NotImplementedError
