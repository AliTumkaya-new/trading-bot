from __future__ import annotations

import math
from typing import Any, Dict

import pandas as pd

from core.models import MarketType, Signal, SignalType
from indicators.technical import (
    compute_all_indicators,
    compute_pivot_levels,
)
from strategies.base import Strategy


class AdvancedCompositeStrategy(Strategy):
    """
    Profesyonel seviye kompozit sinyal üretici.
    10 farklı teknik indikatörü ağırlıklı puanlayarak tek bir skor üretir.

    Ağırlıklar:
      Trend (EMA hizalama)     : %20
      MACD                     : %15
      RSI                      : %12
      Bollinger Bands           : %10
      ADX (trend gücü)         : %10
      Stochastic RSI           : %8
      Volume / OBV             : %10
      Ichimoku Cloud           : %10
      VWAP pozisyonu           : %5

    Skor aralığı: -100 … +100
      > +45  → LONG  (güçlü al)
      < -45  → SHORT (güçlü sat)
      Arası  → FLAT  (bekle)
    """

    name = "advanced_composite_v2"

    LONG_THRESHOLD = 45.0
    SHORT_THRESHOLD = -45.0

    WEIGHTS: Dict[str, float] = {
        "trend_ema": 0.20,
        "macd": 0.15,
        "rsi": 0.12,
        "bollinger": 0.10,
        "adx": 0.10,
        "stochastic_rsi": 0.08,
        "volume_obv": 0.10,
        "ichimoku": 0.10,
        "vwap": 0.05,
    }

    MIN_BARS = 210  # need enough data for long EMAs/Ichimoku

    def generate(self, symbol: str, market: MarketType, df: pd.DataFrame) -> Signal:
        if len(df) < self.MIN_BARS:
            return self._flat(symbol, market, {"reason": "not_enough_data", "bars": len(df)})

        work = compute_all_indicators(df)
        last = work.iloc[-1]
        prev = work.iloc[-2]

        scores: Dict[str, float] = {}
        details: Dict[str, Any] = {}

        # ------ 1. Trend – EMA alignment (9 > 21 > 55 > 100 > 200) ------
        ema_vals = [float(last[f"ema_{p}"]) for p in (9, 21, 55, 100, 200)]
        close = float(last["close"])
        aligned_bull = all(ema_vals[i] >= ema_vals[i + 1] for i in range(len(ema_vals) - 1))
        aligned_bear = all(ema_vals[i] <= ema_vals[i + 1] for i in range(len(ema_vals) - 1))

        if aligned_bull and close > ema_vals[0]:
            trend_score = 100.0
        elif aligned_bear and close < ema_vals[0]:
            trend_score = -100.0
        else:
            above_count = sum(1 for e in ema_vals if close > e)
            trend_score = (above_count / len(ema_vals)) * 200.0 - 100.0
        scores["trend_ema"] = trend_score
        details["ema_alignment"] = "bull" if aligned_bull else ("bear" if aligned_bear else "mixed")

        # ------ 2. MACD ------
        macd_val = float(last["macd"]) if pd.notna(last["macd"]) else 0.0
        macd_sig = float(last["macd_signal"]) if pd.notna(last["macd_signal"]) else 0.0
        macd_hist = float(last["macd_hist"]) if pd.notna(last["macd_hist"]) else 0.0
        prev_hist = float(prev["macd_hist"]) if pd.notna(prev["macd_hist"]) else 0.0

        macd_score = 0.0
        if macd_val > macd_sig:
            macd_score += 40.0
        else:
            macd_score -= 40.0
        if macd_hist > 0:
            macd_score += 30.0
        else:
            macd_score -= 30.0
        if macd_hist > prev_hist:
            macd_score += 30.0
        else:
            macd_score -= 30.0
        scores["macd"] = max(-100.0, min(100.0, macd_score))
        details["macd_hist"] = round(macd_hist, 6)

        # ------ 3. RSI ------
        rsi = float(last["rsi"]) if pd.notna(last["rsi"]) else 50.0
        if rsi < 30:
            rsi_score = 80.0 + (30 - rsi) * 0.67  # oversold = bullish
        elif rsi > 70:
            rsi_score = -80.0 - (rsi - 70) * 0.67  # overbought = bearish
        elif rsi < 50:
            rsi_score = (50 - rsi) * 1.6  # slightly bullish
        else:
            rsi_score = -(rsi - 50) * 1.6  # slightly bearish
        scores["rsi"] = max(-100.0, min(100.0, rsi_score))
        details["rsi"] = round(rsi, 2)

        # ------ 4. Bollinger Bands ------
        pct_b = float(last["bb_pct_b"]) if pd.notna(last["bb_pct_b"]) else 0.5
        bandwidth = float(last["bb_bandwidth"]) if pd.notna(last["bb_bandwidth"]) else 0.0

        if pct_b < 0.0:
            bb_score = 80.0  # below lower band – oversold bounce potential
        elif pct_b > 1.0:
            bb_score = -50.0 if bandwidth > 0.04 else 30.0  # momentum continuation vs overbought
        elif pct_b < 0.2:
            bb_score = 60.0
        elif pct_b > 0.8:
            bb_score = -30.0
        else:
            bb_score = (0.5 - pct_b) * 100.0
        scores["bollinger"] = max(-100.0, min(100.0, bb_score))
        details["bb_pct_b"] = round(pct_b, 4)

        # ------ 5. ADX ------
        adx = float(last["adx"]) if pd.notna(last["adx"]) else 0.0
        plus_di = float(last["plus_di"]) if pd.notna(last["plus_di"]) else 0.0
        minus_di = float(last["minus_di"]) if pd.notna(last["minus_di"]) else 0.0

        if adx > 25:
            if plus_di > minus_di:
                adx_score = min(100.0, adx * 2.0)
            else:
                adx_score = max(-100.0, -adx * 2.0)
        else:
            adx_score = 0.0  # no trend
        scores["adx"] = adx_score
        details["adx"] = round(adx, 2)

        # ------ 6. Stochastic RSI ------
        stoch_k = float(last["stoch_rsi_k"]) if pd.notna(last["stoch_rsi_k"]) else 50.0
        stoch_d = float(last["stoch_rsi_d"]) if pd.notna(last["stoch_rsi_d"]) else 50.0

        if stoch_k < 20:
            stoch_score = 80.0
        elif stoch_k > 80:
            stoch_score = -80.0
        else:
            stoch_score = (50.0 - stoch_k) * 1.33
        if stoch_k > stoch_d:
            stoch_score += 20.0
        else:
            stoch_score -= 20.0
        scores["stochastic_rsi"] = max(-100.0, min(100.0, stoch_score))
        details["stoch_rsi_k"] = round(stoch_k, 2)

        # ------ 7. Volume / OBV ------
        vol_now = float(last["volume"]) if pd.notna(last["volume"]) else 0.0
        vol_avg = float(work["volume"].rolling(20).mean().iloc[-1]) if len(work) >= 20 else vol_now
        obv_now = float(last["obv"]) if pd.notna(last["obv"]) else 0.0
        obv_prev = float(work["obv"].iloc[-20]) if len(work) >= 20 else obv_now

        vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0
        obv_trend = 1 if obv_now > obv_prev else -1

        vol_score = 0.0
        if vol_ratio > 1.5 and obv_trend > 0:
            vol_score = 80.0
        elif vol_ratio > 1.5 and obv_trend < 0:
            vol_score = -60.0
        elif vol_ratio > 1.0 and obv_trend > 0:
            vol_score = 40.0
        elif vol_ratio < 0.5:
            vol_score = 0.0  # low volume = no conviction
        else:
            vol_score = obv_trend * 20.0
        scores["volume_obv"] = max(-100.0, min(100.0, vol_score))
        details["vol_ratio"] = round(vol_ratio, 2)

        # ------ 8. Ichimoku Cloud ------
        tenkan = float(last["ichi_tenkan"]) if pd.notna(last["ichi_tenkan"]) else close
        kijun = float(last["ichi_kijun"]) if pd.notna(last["ichi_kijun"]) else close
        senkou_a = float(last["ichi_senkou_a"]) if pd.notna(last["ichi_senkou_a"]) else close
        senkou_b = float(last["ichi_senkou_b"]) if pd.notna(last["ichi_senkou_b"]) else close

        ichi_score = 0.0
        if close > max(senkou_a, senkou_b):
            ichi_score += 50.0  # above cloud = bullish
        elif close < min(senkou_a, senkou_b):
            ichi_score -= 50.0  # below cloud = bearish
        if tenkan > kijun:
            ichi_score += 30.0
        else:
            ichi_score -= 30.0
        if close > tenkan:
            ichi_score += 20.0
        else:
            ichi_score -= 20.0
        scores["ichimoku"] = max(-100.0, min(100.0, ichi_score))
        details["ichi_cloud"] = "above" if close > max(senkou_a, senkou_b) else (
            "below" if close < min(senkou_a, senkou_b) else "inside"
        )

        # ------ 9. VWAP position ------
        vwap = float(last["vwap"]) if pd.notna(last["vwap"]) else close
        vwap_pct = ((close - vwap) / vwap) * 100.0 if vwap > 0 else 0.0
        vwap_score = max(-100.0, min(100.0, vwap_pct * 20.0))
        scores["vwap"] = vwap_score
        details["vwap_diff_pct"] = round(vwap_pct, 4)

        # ------ Composite weighted score ------
        composite = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        composite = round(composite, 2)

        # Agreement count (how many indicators agree on direction)
        bullish_count = sum(1 for v in scores.values() if v > 20)
        bearish_count = sum(1 for v in scores.values() if v < -20)
        confidence = max(bullish_count, bearish_count) / len(scores) * 100.0

        pivots = compute_pivot_levels(df)
        atr = float(last["atr"]) if pd.notna(last["atr"]) else 0.0

        meta: Dict[str, Any] = {
            "close": close,
            "composite_score": composite,
            "confidence_pct": round(confidence, 1),
            "bullish_indicators": bullish_count,
            "bearish_indicators": bearish_count,
            "indicator_scores": {k: round(v, 2) for k, v in scores.items()},
            "atr": round(atr, 6),
            "pivots": {k: round(v, 4) for k, v in pivots.items()},
            **details,
        }

        if composite >= self.LONG_THRESHOLD and confidence >= 40:
            return Signal(
                symbol=symbol,
                market=market,
                signal=SignalType.LONG,
                score=composite,
                strategy_name=self.name,
                metadata=meta,
            )
        elif composite <= self.SHORT_THRESHOLD and confidence >= 40:
            return Signal(
                symbol=symbol,
                market=market,
                signal=SignalType.SHORT,
                score=abs(composite),
                strategy_name=self.name,
                metadata=meta,
            )
        else:
            return self._flat(symbol, market, meta)

    def _flat(self, symbol: str, market: MarketType, meta: Dict[str, Any]) -> Signal:
        return Signal(
            symbol=symbol,
            market=market,
            signal=SignalType.FLAT,
            score=0.0,
            strategy_name=self.name,
            metadata=meta,
        )
