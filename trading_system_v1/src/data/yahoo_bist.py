from __future__ import annotations

import pandas as pd
import yfinance as yf

from data.base import MarketDataAdapter


class YahooBISTAdapter(MarketDataAdapter):
    """
    Research-only adapter for Borsa Istanbul symbols via Yahoo Finance.
    Expected symbol format example: THYAO.IS
    """

    INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "60m",
        "1d": "1d",
    }

    PERIOD_MAP = {
        "1m": "7d",
        "5m": "30d",
        "15m": "60d",
        "1h": "730d",
        "1d": "2y",
    }

    def fetch_ohlcv(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        mapped_interval = self.INTERVAL_MAP.get(interval, interval)
        period = self.PERIOD_MAP.get(interval, "2y")

        df = yf.download(
            tickers=symbol,
            interval=mapped_interval,
            period=period,
            progress=False,
            auto_adjust=True,
            threads=False,
        )

        if df.empty:
            raise ValueError(f"No Yahoo Finance OHLCV received for {symbol}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        timestamp_col = "Datetime" if "Datetime" in df.columns else "Date"

        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df[timestamp_col], utc=True),
                "open": df["Open"].astype(float),
                "high": df["High"].astype(float),
                "low": df["Low"].astype(float),
                "close": df["Close"].astype(float),
                "volume": df["Volume"].fillna(0).astype(float),
            }
        )

        # Drop rows where close is NaN (incomplete bars)
        out = out.dropna(subset=["close"])
        out = out.tail(limit).reset_index(drop=True)
        self.validate_frame(out, ["timestamp", "open", "high", "low", "close", "volume"])
        return out
