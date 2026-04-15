from __future__ import annotations

from typing import Dict

import pandas as pd
import requests

from core.config import AppConfig
from data.base import MarketDataAdapter


class BinanceSpotAdapter(MarketDataAdapter):
    INTERVAL_MAP: Dict[str, str] = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }

    def __init__(self, config: AppConfig, timeout: int = 10) -> None:
        self.config = config
        self.timeout = timeout

    def fetch_ohlcv(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        mapped_interval = self.INTERVAL_MAP.get(interval, interval)
        url = f"{self.config.binance_base_url}/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": mapped_interval,
            "limit": limit,
        }

        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        raw = response.json()

        df = pd.DataFrame(
            raw,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )

        if df.empty:
            raise ValueError(f"No Binance OHLCV received for {symbol}")

        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df["open_time"], unit="ms", utc=True),
                "open": df["open"].astype(float),
                "high": df["high"].astype(float),
                "low": df["low"].astype(float),
                "close": df["close"].astype(float),
                "volume": df["volume"].astype(float),
            }
        )
        self.validate_frame(out, ["timestamp", "open", "high", "low", "close", "volume"])
        return out
