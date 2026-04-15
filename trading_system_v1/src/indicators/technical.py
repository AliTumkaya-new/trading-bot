from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# RSI – Relative Strength Index
# ---------------------------------------------------------------------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ---------------------------------------------------------------------------
# MACD – Moving Average Convergence Divergence
# ---------------------------------------------------------------------------
def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram}
    )


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------
def compute_bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    pct_b = (series - lower) / (upper - lower)
    bandwidth = (upper - lower) / sma
    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_middle": sma,
            "bb_lower": lower,
            "bb_pct_b": pct_b,
            "bb_bandwidth": bandwidth,
        }
    )


# ---------------------------------------------------------------------------
# EMA – Exponential Moving Average
# ---------------------------------------------------------------------------
def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# ADX – Average Directional Index
# ---------------------------------------------------------------------------
def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (
        plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    )
    minus_di = 100.0 * (
        minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    )
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return pd.DataFrame(
        {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}
    )


# ---------------------------------------------------------------------------
# Stochastic RSI
# ---------------------------------------------------------------------------
def compute_stochastic_rsi(
    series: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> pd.DataFrame:
    rsi = compute_rsi(series, rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    k = stoch_rsi.rolling(k_smooth).mean() * 100.0
    d = k.rolling(d_smooth).mean()
    return pd.DataFrame({"stoch_rsi_k": k, "stoch_rsi_d": d})


# ---------------------------------------------------------------------------
# ATR – Average True Range
# ---------------------------------------------------------------------------
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# OBV – On Balance Volume
# ---------------------------------------------------------------------------
def compute_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff())
    obv = (direction * df["volume"]).fillna(0).cumsum()
    return obv


# ---------------------------------------------------------------------------
# VWAP – Volume Weighted Average Price
# ---------------------------------------------------------------------------
def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum()
    return cumulative_tp_vol / cumulative_vol.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Ichimoku Cloud (Simplified)
# ---------------------------------------------------------------------------
def compute_ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2.0
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2.0
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(kijun)
    senkou_span_b = (
        (high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2.0
    ).shift(kijun)
    chikou_span = df["close"].shift(-kijun)
    return pd.DataFrame(
        {
            "ichi_tenkan": tenkan_sen,
            "ichi_kijun": kijun_sen,
            "ichi_senkou_a": senkou_span_a,
            "ichi_senkou_b": senkou_span_b,
            "ichi_chikou": chikou_span,
        }
    )


# ---------------------------------------------------------------------------
# Support & Resistance (pivot-based)
# ---------------------------------------------------------------------------
def compute_pivot_levels(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    h, l, c = float(last["high"]), float(last["low"]), float(last["close"])
    pivot = (h + l + c) / 3.0
    r1 = 2.0 * pivot - l
    s1 = 2.0 * pivot - h
    r2 = pivot + (h - l)
    s2 = pivot - (h - l)
    r3 = h + 2.0 * (pivot - l)
    s3 = l - 2.0 * (h - pivot)
    return {
        "pivot": pivot,
        "r1": r1,
        "s1": s1,
        "r2": r2,
        "s2": s2,
        "r3": r3,
        "s3": s3,
    }


# ---------------------------------------------------------------------------
# Master: Compute ALL indicators on a DataFrame
# ---------------------------------------------------------------------------
def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    # EMAs
    for period in (9, 21, 55, 100, 200):
        work[f"ema_{period}"] = compute_ema(work["close"], period)

    # RSI
    work["rsi"] = compute_rsi(work["close"], 14)

    # MACD
    macd_df = compute_macd(work["close"])
    work = pd.concat([work, macd_df], axis=1)

    # Bollinger Bands
    bb_df = compute_bollinger_bands(work["close"])
    work = pd.concat([work, bb_df], axis=1)

    # ADX
    adx_df = compute_adx(work)
    work = pd.concat([work, adx_df], axis=1)

    # Stochastic RSI
    stoch_df = compute_stochastic_rsi(work["close"])
    work = pd.concat([work, stoch_df], axis=1)

    # ATR
    work["atr"] = compute_atr(work)

    # OBV
    work["obv"] = compute_obv(work)

    # VWAP
    work["vwap"] = compute_vwap(work)

    # Ichimoku
    ichi_df = compute_ichimoku(work)
    work = pd.concat([work, ichi_df], axis=1)

    return work
