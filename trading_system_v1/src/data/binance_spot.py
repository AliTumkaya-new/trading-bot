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

    # Fallback URL'ler — ana API engellenirse sırayla dener
    FALLBACK_URLS = [
        "https://data-api.binance.vision",
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com",
    ]

    def __init__(self, config: AppConfig, timeout: int = 10) -> None:
        self.config = config
        self.timeout = timeout

    def fetch_ohlcv(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        mapped_interval = self.INTERVAL_MAP.get(interval, interval)
        params = {
            "symbol": symbol.upper(),
            "interval": mapped_interval,
            "limit": limit,
        }

        # Ana URL + fallback'ler
        urls_to_try = [self.config.binance_base_url] + self.FALLBACK_URLS
        last_error = None

        for base_url in urls_to_try:
            url = f"{base_url}/api/v3/klines"
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                if response.status_code == 451:
                    last_error = f"HTTP 451 from {base_url}"
                    continue
                response.raise_for_status()
                raw = response.json()
                break
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                continue
        else:
            raise ConnectionError(f"All Binance endpoints failed. Last error: {last_error}")

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
