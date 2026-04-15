from __future__ import annotations

"""
Agresif Kısa Vadeli Momentum Stratejisi
========================================
Hedef: Yüksek hacimli, güçlü momentum hareketlerini yakalayıp kısa sürede
sermayeyi katlamak.

Sinyal koşulları (LONG):
  1. Hacim patlaması   : Son mum hacmi > 2x ortalama hacim
  2. Fiyat ivmesi      : ROC(5) > 0 VE ivmeleniyor (ROC artıyor)
  3. Breakout           : Fiyat > son 10 mumun en yüksek seviyesi
  4. RSI momentum      : RSI 50-75 arasında (güçlü ama aşırı alım değil)
  5. MACD momentum     : MACD histogramı pozitif ve artıyor
  6. EMA trend         : Fiyat > EMA9 > EMA21
  7. Squeeze breakout  : Bollinger bandwidth genişliyor (volatilite patlaması)
  8. OBV yükseliş      : OBV > OBV EMA(10) — akıllı para girişi

Skor: -100 … +100
  > +35  → LONG (agresif gir)
  < -35  → SHORT
"""

import math
from typing import Any, Dict

import numpy as np
import pandas as pd

from core.models import MarketType, Signal, SignalType
from indicators.technical import (
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_macd,
    compute_obv,
    compute_rsi,
    compute_stochastic_rsi,
    compute_vwap,
)
from strategies.base import Strategy


class AggressiveMomentumStrategy(Strategy):
    name = "aggressive_momentum_v3"

    LONG_THRESHOLD = 35.0
    SHORT_THRESHOLD = -35.0

    WEIGHTS: Dict[str, float] = {
        "volume_surge": 0.22,
        "price_momentum": 0.18,
        "breakout": 0.15,
        "rsi_momentum": 0.10,
        "macd_accel": 0.12,
        "ema_trend": 0.08,
        "squeeze": 0.08,
        "obv_flow": 0.07,
    }

    MIN_BARS = 60

    def generate(self, symbol: str, market: MarketType, df: pd.DataFrame) -> Signal:
        if len(df) < self.MIN_BARS:
            return self._flat(symbol, market, {"reason": "not_enough_data", "bars": len(df)})

        work = df.copy()

        # --- Compute indicators ---
        work["ema_9"] = compute_ema(work["close"], 9)
        work["ema_21"] = compute_ema(work["close"], 21)
        work["rsi"] = compute_rsi(work["close"], 14)
        work["atr"] = compute_atr(work, 14)
        work["obv"] = compute_obv(work)
        work["obv_ema"] = compute_ema(work["obv"], 10)
        work["vwap"] = compute_vwap(work)

        macd_df = compute_macd(work["close"], fast=8, slow=21, signal=5)
        work = pd.concat([work, macd_df], axis=1)

        bb_df = compute_bollinger_bands(work["close"], period=20, std_dev=2.0)
        work = pd.concat([work, bb_df], axis=1)

        stoch_df = compute_stochastic_rsi(work["close"], rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3)
        work = pd.concat([work, stoch_df], axis=1)

        # ROC (Rate of Change)
        work["roc_5"] = work["close"].pct_change(5) * 100.0
        work["roc_3"] = work["close"].pct_change(3) * 100.0
        work["roc_1"] = work["close"].pct_change(1) * 100.0

        # Volume analysis
        work["vol_sma_20"] = work["volume"].rolling(20).mean()
        work["vol_ratio"] = work["volume"] / work["vol_sma_20"].replace(0, np.nan)

        # Recent high/low for breakout
        work["high_10"] = work["high"].rolling(10).max().shift(1)
        work["low_10"] = work["low"].rolling(10).min().shift(1)

        # Bandwidth change (squeeze detection)
        work["bandwidth_prev"] = work["bb_bandwidth"].shift(1)

        last = work.iloc[-1]
        prev = work.iloc[-2]
        close = float(last["close"])
        atr = float(last["atr"]) if pd.notna(last["atr"]) else 0.0

        scores: Dict[str, float] = {}
        details: Dict[str, Any] = {}

        # ------ 1. Volume Surge (en önemli) ------
        vol_ratio = float(last["vol_ratio"]) if pd.notna(last["vol_ratio"]) else 1.0
        prev_vol_ratio = float(prev["vol_ratio"]) if pd.notna(prev["vol_ratio"]) else 1.0

        if vol_ratio > 3.0:
            vol_score = 100.0  # Devasa hacim patlaması
        elif vol_ratio > 2.0:
            vol_score = 80.0
        elif vol_ratio > 1.5:
            vol_score = 50.0
        elif vol_ratio > 1.0:
            vol_score = 20.0
        elif vol_ratio < 0.5:
            vol_score = -40.0  # Düşük hacim = tehlike
        else:
            vol_score = 0.0

        # Hacim yönü (fiyat yukarı + hacim artışı = güçlü)
        roc_1 = float(last["roc_1"]) if pd.notna(last["roc_1"]) else 0.0
        if roc_1 > 0 and vol_ratio > 1.5:
            vol_score = min(100, vol_score + 20)
        elif roc_1 < 0 and vol_ratio > 1.5:
            vol_score = max(-100, -abs(vol_score))  # Satış baskısı

        scores["volume_surge"] = max(-100, min(100, vol_score))
        details["vol_ratio"] = round(vol_ratio, 2)

        # ------ 2. Price Momentum (ROC ivmesi) ------
        roc_5 = float(last["roc_5"]) if pd.notna(last["roc_5"]) else 0.0
        roc_3 = float(last["roc_3"]) if pd.notna(last["roc_3"]) else 0.0
        prev_roc_5 = float(prev["roc_5"]) if pd.notna(prev["roc_5"]) else 0.0

        # İvmelenme: ROC artıyor mu?
        acceleration = roc_5 - prev_roc_5

        if roc_5 > 5.0 and acceleration > 0:
            mom_score = 100.0  # Güçlü yukarı ivme, hızlanıyor
        elif roc_5 > 3.0:
            mom_score = 70.0
        elif roc_5 > 1.0 and acceleration > 0:
            mom_score = 40.0
        elif roc_5 < -3.0:
            mom_score = -70.0
        elif roc_5 < -1.0:
            mom_score = -30.0
        else:
            mom_score = roc_5 * 10.0

        scores["price_momentum"] = max(-100, min(100, mom_score))
        details["roc_5"] = round(roc_5, 2)
        details["acceleration"] = round(acceleration, 2)

        # ------ 3. Breakout ------
        high_10 = float(last["high_10"]) if pd.notna(last["high_10"]) else close
        low_10 = float(last["low_10"]) if pd.notna(last["low_10"]) else close

        if close > high_10:
            breakout_pct = ((close - high_10) / high_10) * 100.0 if high_10 > 0 else 0
            breakout_score = min(100, 50.0 + breakout_pct * 20.0)
        elif close < low_10:
            breakdown_pct = ((low_10 - close) / low_10) * 100.0 if low_10 > 0 else 0
            breakout_score = max(-100, -50.0 - breakdown_pct * 20.0)
        else:
            # Range içinde — nötr
            range_pos = (close - low_10) / (high_10 - low_10) if (high_10 - low_10) > 0 else 0.5
            breakout_score = (range_pos - 0.5) * 40.0

        scores["breakout"] = max(-100, min(100, breakout_score))
        details["close_vs_high10"] = f"{close:.4f} vs {high_10:.4f}"

        # ------ 4. RSI Momentum ------
        rsi = float(last["rsi"]) if pd.notna(last["rsi"]) else 50.0
        prev_rsi = float(prev["rsi"]) if pd.notna(prev["rsi"]) else 50.0
        rsi_rising = rsi > prev_rsi

        if 50 < rsi < 70 and rsi_rising:
            rsi_score = 80.0  # Güçlü momentum bölgesi
        elif 40 < rsi < 50 and rsi_rising:
            rsi_score = 50.0  # Yükselişe geçiyor
        elif rsi > 80:
            rsi_score = -30.0  # Aşırı alım — dikkat
        elif rsi < 25:
            rsi_score = 60.0  # Aşırı satım = dip fırsatı (kısa vadede)
        elif rsi > 70:
            rsi_score = -10.0
        elif rsi < 30:
            rsi_score = 40.0
        else:
            rsi_score = (rsi - 50.0) * 1.5

        scores["rsi_momentum"] = max(-100, min(100, rsi_score))
        details["rsi"] = round(rsi, 2)

        # ------ 5. MACD Acceleration ------
        macd_hist = float(last["macd_hist"]) if pd.notna(last["macd_hist"]) else 0.0
        prev_hist = float(prev["macd_hist"]) if pd.notna(prev["macd_hist"]) else 0.0
        hist_change = macd_hist - prev_hist

        if macd_hist > 0 and hist_change > 0:
            macd_score = 100.0  # Pozitif ve güçleniyor
        elif macd_hist > 0 and hist_change < 0:
            macd_score = 20.0  # Pozitif ama zayıflıyor
        elif macd_hist < 0 and hist_change > 0:
            macd_score = 40.0  # Negatif ama dönüyor
        elif macd_hist < 0 and hist_change < 0:
            macd_score = -100.0  # Negatif ve kötüleşiyor
        else:
            macd_score = 0.0

        scores["macd_accel"] = max(-100, min(100, macd_score))
        details["macd_hist"] = round(macd_hist, 6)

        # ------ 6. EMA Trend (kısa vadeli) ------
        ema9 = float(last["ema_9"]) if pd.notna(last["ema_9"]) else close
        ema21 = float(last["ema_21"]) if pd.notna(last["ema_21"]) else close

        if close > ema9 > ema21:
            ema_score = 100.0
        elif close > ema9 and ema9 < ema21:
            ema_score = 30.0  # Fiyat EMA9 üstünde ama trend henüz onaylanmadı
        elif close < ema9 < ema21:
            ema_score = -100.0
        elif close < ema9:
            ema_score = -40.0
        else:
            ema_score = 0.0

        scores["ema_trend"] = max(-100, min(100, ema_score))

        # ------ 7. Squeeze Breakout (BB genişleme) ------
        bandwidth = float(last["bb_bandwidth"]) if pd.notna(last["bb_bandwidth"]) else 0.0
        bandwidth_prev = float(last["bandwidth_prev"]) if pd.notna(last["bandwidth_prev"]) else 0.0
        pct_b = float(last["bb_pct_b"]) if pd.notna(last["bb_pct_b"]) else 0.5

        if bandwidth > bandwidth_prev and pct_b > 0.8:
            squeeze_score = 100.0  # Volatilite patlıyor + fiyat üst bantta
        elif bandwidth > bandwidth_prev and pct_b < 0.2:
            squeeze_score = -80.0  # Volatilite patlıyor + fiyat alt bantta
        elif bandwidth > bandwidth_prev:
            squeeze_score = 40.0 if pct_b > 0.5 else -40.0
        elif bandwidth < bandwidth_prev * 0.8:
            squeeze_score = 0.0  # Sıkışma — hareket yakın ama yön belirsiz
        else:
            squeeze_score = (pct_b - 0.5) * 40.0

        scores["squeeze"] = max(-100, min(100, squeeze_score))
        details["bb_bandwidth"] = round(bandwidth, 4)
        details["bb_pct_b"] = round(pct_b, 4)

        # ------ 8. OBV Flow (akıllı para) ------
        obv_now = float(last["obv"]) if pd.notna(last["obv"]) else 0.0
        obv_ema = float(last["obv_ema"]) if pd.notna(last["obv_ema"]) else 0.0

        if obv_now > obv_ema * 1.05:
            obv_score = 80.0  # Güçlü alıcı akışı
        elif obv_now > obv_ema:
            obv_score = 30.0
        elif obv_now < obv_ema * 0.95:
            obv_score = -80.0  # Satıcı baskısı
        else:
            obv_score = -20.0

        scores["obv_flow"] = max(-100, min(100, obv_score))

        # ------ Composite Score ------
        composite = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        composite = round(composite, 2)

        bullish_count = sum(1 for v in scores.values() if v > 15)
        bearish_count = sum(1 for v in scores.values() if v < -15)
        confidence = max(bullish_count, bearish_count) / len(scores) * 100.0

        # Volatilite yüzdesi (ATR/close) — yüksek = daha fazla hareket potansiyeli
        volatility_pct = (atr / close * 100) if close > 0 else 0.0

        meta: Dict[str, Any] = {
            "close": close,
            "composite_score": composite,
            "confidence_pct": round(confidence, 1),
            "bullish_indicators": bullish_count,
            "bearish_indicators": bearish_count,
            "indicator_scores": {k: round(v, 2) for k, v in scores.items()},
            "atr": round(atr, 6),
            "atr_pct": round(volatility_pct, 2),
            **details,
        }

        if composite >= self.LONG_THRESHOLD and confidence >= 35:
            return Signal(
                symbol=symbol,
                market=market,
                signal=SignalType.LONG,
                score=composite,
                strategy_name=self.name,
                metadata=meta,
            )
        elif composite <= self.SHORT_THRESHOLD and confidence >= 35:
            return Signal(
                symbol=symbol,
                market=market,
                signal=SignalType.SHORT,
                score=abs(composite),
                strategy_name=self.name,
                metadata=meta,
            )
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
