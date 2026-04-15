from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class RiskConfig:
    capital_tl: float = 5_000.0          # Ana sermaye (TL)
    risk_per_trade_pct: float = 0.04     # İşlem başına max risk %4 (agresif)
    max_open_positions: int = 4          # Daha az pozisyon, daha yoğun
    max_position_pct: float = 0.35       # Tek pozisyon max sermayenin %35'i
    default_stop_loss_pct: float = 0.025 # %2.5 sıkı stop loss
    default_take_profit_pct: float = 0.08  # %8 take profit (3:1 R/R)
    trailing_stop_pct: float = 0.015     # %1.5 trailing stop
    crypto_leverage: int = 3             # Kripto kaldıraç (3x)


@dataclass(slots=True)
class ScannerConfig:
    crypto_symbols: List[str] = field(default_factory=lambda: [
        # Yüksek hacim & volatilite coinler
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
        "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
        "SUIUSDT", "PEPEUSDT", "WIFUSDT", "APTUSDT",
        "NEARUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
        "FETUSDT", "RENDERUSDT", "TIAUSDT", "JUPUSDT",
    ])
    bist_symbols: List[str] = field(default_factory=lambda: [
        # Yüksek hacimli, momentum potansiyeli olan BIST hisseleri
        "THYAO.IS", "ASELS.IS", "TUPRS.IS", "GARAN.IS",
        "KCHOL.IS", "SAHOL.IS", "FROTO.IS", "BIMAS.IS",
    ])
    lookback_bars: int = 200             # 1h mumlar için yeterli geçmiş
    interval: str = "1h"                 # 1 saatlik — daha hızlı sinyaller
    bist_interval: str = "1d"            # BIST için günlük (Yahoo 1h desteklemiyor)
    cycle_interval_minutes: int = 120    # Tarama döngüsü aralığı (dakika) — 2h


@dataclass(slots=True)
class AppConfig:
    binance_base_url: str = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
    twelvedata_api_key: str = os.getenv("TWELVEDATA_API_KEY", "")
    timezone: str = "Europe/Istanbul"
    risk: RiskConfig = field(default_factory=RiskConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
