"""
Adaptif Makine Öğrenmesi Modülü
================================
Trading stratejisini kapalı işlem sonuçlarından öğrenerek optimize eder.

3 Katman:
  1. Adaptif Ağırlık Optimizasyonu — indikatör ağırlıklarını EMA ile günceller
  2. Sembol Performans Profili  — her sembolün başarı oranını takip eder
  3. RF Sınıflandırıcı (yeterli veri olunca) — sinyal kalitesini tahmin eder
"""
from __future__ import annotations

import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logging_utils import get_logger

logger = get_logger("ml_learner")

# Default indicator weights (başlangıç değerleri)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "volume_surge": 0.22,
    "price_momentum": 0.18,
    "breakout": 0.15,
    "rsi_momentum": 0.10,
    "macd_accel": 0.12,
    "ema_trend": 0.08,
    "squeeze": 0.08,
    "obv_flow": 0.07,
}

# Öğrenme hızı — ağırlıkların ne kadar hızlı değişeceğini kontrol eder
LEARNING_RATE = 0.15
# Minimum ağırlık — hiçbir indikatör 0'ın altına düşemez
MIN_WEIGHT = 0.02
# RF model eğitimi için gereken minimum kapanmış işlem sayısı
MIN_TRADES_FOR_RF = 10

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "ml_model.pkl"


class AdaptiveLearner:
    """Her kapanan işlemden öğrenir ve strateji parametrelerini optimize eder."""

    def __init__(self, db_conn_factory):
        """
        Parameters
        ----------
        db_conn_factory : callable
            database.db._conn gibi bir context manager factory.
        """
        self._conn = db_conn_factory
        self._ensure_tables()
        self._rf_model = None
        self._load_rf_model()

    # ------------------------------------------------------------------
    # DB Tablo Kurulumu
    # ------------------------------------------------------------------
    def _ensure_tables(self) -> None:
        with self._conn() as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS indicator_performance (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol            TEXT NOT NULL,
                market            TEXT NOT NULL,
                direction         TEXT NOT NULL,
                outcome           TEXT NOT NULL,
                pnl               REAL NOT NULL,
                pnl_pct           REAL NOT NULL,
                hold_bars         INTEGER NOT NULL DEFAULT 0,
                indicator_scores  TEXT NOT NULL,
                composite_score   REAL NOT NULL,
                confidence        REAL NOT NULL,
                created_at        TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS learned_weights (
                id            INTEGER PRIMARY KEY CHECK (id = 1),
                weights       TEXT NOT NULL,
                total_updates INTEGER NOT NULL DEFAULT 0,
                win_rate      REAL NOT NULL DEFAULT 0,
                avg_pnl       REAL NOT NULL DEFAULT 0,
                updated_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS symbol_performance (
                symbol        TEXT PRIMARY KEY,
                market        TEXT NOT NULL,
                total_trades  INTEGER NOT NULL DEFAULT 0,
                wins          INTEGER NOT NULL DEFAULT 0,
                losses        INTEGER NOT NULL DEFAULT 0,
                total_pnl     REAL NOT NULL DEFAULT 0,
                avg_pnl       REAL NOT NULL DEFAULT 0,
                best_pnl      REAL NOT NULL DEFAULT 0,
                worst_pnl     REAL NOT NULL DEFAULT 0,
                updated_at    TEXT NOT NULL
            );
            """)

    # ------------------------------------------------------------------
    # 1. Ağırlık Yönetimi
    # ------------------------------------------------------------------
    def get_learned_weights(self) -> Dict[str, float]:
        """Öğrenilmiş ağırlıkları DB'den yükle; yoksa varsayılanları döndür."""
        with self._conn() as con:
            row = con.execute("SELECT weights FROM learned_weights WHERE id=1").fetchone()
            if row:
                return json.loads(row["weights"])
        return dict(DEFAULT_WEIGHTS)

    def _save_weights(self, weights: Dict[str, float], stats: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        w_json = json.dumps(weights)
        with self._conn() as con:
            existing = con.execute("SELECT id FROM learned_weights WHERE id=1").fetchone()
            if existing:
                con.execute(
                    """UPDATE learned_weights
                       SET weights=?, total_updates=total_updates+1,
                           win_rate=?, avg_pnl=?, updated_at=?
                       WHERE id=1""",
                    (w_json, stats.get("win_rate", 0), stats.get("avg_pnl", 0), now),
                )
            else:
                con.execute(
                    """INSERT INTO learned_weights (id, weights, total_updates, win_rate, avg_pnl, updated_at)
                       VALUES (1, ?, 1, ?, ?, ?)""",
                    (w_json, stats.get("win_rate", 0), stats.get("avg_pnl", 0), now),
                )

    # ------------------------------------------------------------------
    # 2. İşlem Sonucu Kaydetme (her kapalı işlem sonrası çağrılır)
    # ------------------------------------------------------------------
    def record_outcome(
        self,
        symbol: str,
        market: str,
        direction: str,
        pnl: float,
        pnl_pct: float,
        indicator_scores: Dict[str, float],
        composite_score: float,
        confidence: float,
        hold_bars: int = 0,
    ) -> None:
        """Kapanan işlemin sonucunu kaydet ve öğren."""
        outcome = "win" if pnl > 0 else "loss"
        now = datetime.now(timezone.utc).isoformat()

        with self._conn() as con:
            con.execute(
                """INSERT INTO indicator_performance
                   (symbol, market, direction, outcome, pnl, pnl_pct, hold_bars,
                    indicator_scores, composite_score, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, market, direction, outcome, pnl, pnl_pct, hold_bars,
                 json.dumps(indicator_scores), composite_score, confidence, now),
            )

        # Sembol performansı güncelle
        self._update_symbol_performance(symbol, market, pnl, outcome)

        # Ağırlıkları güncelle
        self._adapt_weights(indicator_scores, outcome, pnl_pct)

        # RF model güncelleme kontrolü
        self._maybe_retrain_rf()

        logger.info(
            "🧠 ML ÖĞRENME: %s %s | Sonuç: %s | P&L: %+.2f%% | Güncellenen ağırlıklar kaydedildi",
            symbol, direction.upper(), outcome.upper(), pnl_pct,
        )

    # ------------------------------------------------------------------
    # 3. Adaptif Ağırlık Güncelleme
    # ------------------------------------------------------------------
    def _adapt_weights(
        self,
        indicator_scores: Dict[str, float],
        outcome: str,
        pnl_pct: float,
    ) -> None:
        """
        Gradient-benzeri ağırlık güncellemesi:
        - Kazanan işlemde: doğru yönde skor veren indikatörlerin ağırlığını artır
        - Kaybeden işlemde: yanlış yönde skor veren indikatörlerin ağırlığını azalt
        """
        weights = self.get_learned_weights()

        # İşlem yönünü belirle (positive score = LONG doğru, negative = SHORT doğru)
        is_win = outcome == "win"

        for ind, score in indicator_scores.items():
            if ind not in weights:
                continue

            # İndikatör doğru mu tahmin etti?
            # Win durumunda: yüksek abs(score) = doğru tahmin → ağırlık artır
            # Loss durumunda: yüksek abs(score) = yanlış tahmin → ağırlık azalt
            score_strength = abs(score) / 100.0  # 0-1 arası normalize
            adjustment = LEARNING_RATE * score_strength

            if is_win:
                # Kazandık → bu indikatöre güveni artır
                weights[ind] += adjustment
            else:
                # Kaybettik → bu indikatöre güveni azalt
                # Ama sadece güçlü sinyal veren indikatörler cezalanır
                if score_strength > 0.3:
                    weights[ind] -= adjustment * 0.5  # Ceza daha hafif

            weights[ind] = max(MIN_WEIGHT, weights[ind])

        # Normalize: toplam = 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 4) for k, v in weights.items()}

        # İstatistik hesapla
        stats = self._compute_stats()
        self._save_weights(weights, stats)

    # ------------------------------------------------------------------
    # 4. Sembol Performansı
    # ------------------------------------------------------------------
    def _update_symbol_performance(self, symbol: str, market: str, pnl: float, outcome: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as con:
            row = con.execute("SELECT * FROM symbol_performance WHERE symbol=?", (symbol,)).fetchone()
            if row:
                total = row["total_trades"] + 1
                wins = row["wins"] + (1 if outcome == "win" else 0)
                losses = row["losses"] + (1 if outcome == "loss" else 0)
                total_pnl = row["total_pnl"] + pnl
                avg_pnl = total_pnl / total
                best = max(row["best_pnl"], pnl)
                worst = min(row["worst_pnl"], pnl)
                con.execute(
                    """UPDATE symbol_performance
                       SET total_trades=?, wins=?, losses=?, total_pnl=?,
                           avg_pnl=?, best_pnl=?, worst_pnl=?, updated_at=?
                       WHERE symbol=?""",
                    (total, wins, losses, total_pnl, avg_pnl, best, worst, now, symbol),
                )
            else:
                con.execute(
                    """INSERT INTO symbol_performance
                       (symbol, market, total_trades, wins, losses, total_pnl,
                        avg_pnl, best_pnl, worst_pnl, updated_at)
                       VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?)""",
                    (symbol, market,
                     1 if outcome == "win" else 0,
                     1 if outcome == "loss" else 0,
                     pnl, pnl, pnl, pnl, now),
                )

    def get_symbol_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._conn() as con:
            row = con.execute("SELECT * FROM symbol_performance WHERE symbol=?", (symbol,)).fetchone()
            return dict(row) if row else None

    def get_all_symbol_stats(self) -> List[Dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM symbol_performance ORDER BY total_pnl DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # 5. Sinyal Kalite Tahmini (RF Model)
    # ------------------------------------------------------------------
    def predict_signal_quality(self, indicator_scores: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Eğitilmiş RF modeli ile sinyalin kalitesini tahmin et.
        Yeterli veri yoksa None döner.
        """
        if self._rf_model is None:
            return None

        features = self._scores_to_features(indicator_scores)
        if features is None:
            return None

        try:
            proba = self._rf_model.predict_proba([features])[0]
            prediction = self._rf_model.predict([features])[0]
            return {
                "predicted_outcome": "win" if prediction == 1 else "loss",
                "win_probability": round(float(proba[1]) if len(proba) > 1 else 0.0, 3),
                "loss_probability": round(float(proba[0]), 3),
                "model_confidence": round(float(max(proba)) * 100, 1),
            }
        except Exception as e:
            logger.warning("RF tahmin hatası: %s", e)
            return None

    def _scores_to_features(self, indicator_scores: Dict[str, float]) -> Optional[List[float]]:
        """İndikatör skorlarını sabit sıralı feature vektörüne dönüştür."""
        feature_keys = sorted(DEFAULT_WEIGHTS.keys())
        try:
            return [indicator_scores.get(k, 0.0) for k in feature_keys]
        except Exception:
            return None

    def _maybe_retrain_rf(self) -> None:
        """Yeterli veri varsa RF modelini yeniden eğit."""
        with self._conn() as con:
            count = con.execute("SELECT COUNT(*) as cnt FROM indicator_performance").fetchone()["cnt"]

        if count < MIN_TRADES_FOR_RF:
            return

        # Her 10 işlemde bir yeniden eğit
        if count % 10 != 0:
            return

        self._train_rf_model()

    def _train_rf_model(self) -> None:
        """RandomForest modelini tüm geçmiş verilerle eğit."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
        except ImportError:
            logger.warning("scikit-learn kurulu değil; RF model eğitimi atlanıyor.")
            return

        with self._conn() as con:
            rows = con.execute(
                "SELECT indicator_scores, outcome FROM indicator_performance"
            ).fetchall()

        if len(rows) < MIN_TRADES_FOR_RF:
            return

        feature_keys = sorted(DEFAULT_WEIGHTS.keys())
        X, y = [], []
        for row in rows:
            scores = json.loads(row["indicator_scores"])
            features = [scores.get(k, 0.0) for k in feature_keys]
            X.append(features)
            y.append(1 if row["outcome"] == "win" else 0)

        X = np.array(X)
        y = np.array(y)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced",
        )

        # Cross-validation ile doğruluk kontrolü
        if len(X) >= 10:
            try:
                cv_scores = cross_val_score(model, X, y, cv=min(5, len(X) // 2), scoring="accuracy")
                logger.info(
                    "🧠 RF CV Doğruluk: %.1f%% (±%.1f%%)",
                    cv_scores.mean() * 100, cv_scores.std() * 100,
                )
            except Exception:
                pass

        model.fit(X, y)
        self._rf_model = model

        # Feature importance logla
        importances = dict(zip(feature_keys, model.feature_importances_))
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        logger.info("🧠 RF Feature Importance:")
        for feat, imp in sorted_imp:
            logger.info("   %s: %.3f", feat, imp)

        # Modeli kaydet
        try:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)
            logger.info("🧠 RF model kaydedildi: %s", MODEL_PATH)
        except Exception as e:
            logger.warning("RF model kaydetme hatası: %s", e)

    def _load_rf_model(self) -> None:
        """Kaydedilmiş RF modelini yükle."""
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    self._rf_model = pickle.load(f)
                logger.info("🧠 RF model yüklendi: %s", MODEL_PATH)
            except Exception as e:
                logger.warning("RF model yükleme hatası: %s", e)
                self._rf_model = None

    # ------------------------------------------------------------------
    # 6. İstatistikler
    # ------------------------------------------------------------------
    def _compute_stats(self) -> Dict[str, Any]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT outcome, pnl_pct FROM indicator_performance"
            ).fetchall()

        if not rows:
            return {"win_rate": 0, "avg_pnl": 0, "total": 0}

        wins = sum(1 for r in rows if r["outcome"] == "win")
        total = len(rows)
        avg_pnl = sum(r["pnl_pct"] for r in rows) / total

        return {
            "win_rate": round(wins / total * 100, 1),
            "avg_pnl": round(avg_pnl, 2),
            "total": total,
        }

    def get_learning_summary(self) -> Dict[str, Any]:
        """Dashboard için öğrenme özeti."""
        weights = self.get_learned_weights()
        stats = self._compute_stats()
        symbol_stats = self.get_all_symbol_stats()

        # RF model durumu
        rf_status = "active" if self._rf_model is not None else "waiting_data"

        with self._conn() as con:
            row = con.execute("SELECT total_updates FROM learned_weights WHERE id=1").fetchone()
            total_updates = row["total_updates"] if row else 0

        return {
            "current_weights": weights,
            "default_weights": DEFAULT_WEIGHTS,
            "weight_changes": {
                k: round(weights.get(k, 0) - DEFAULT_WEIGHTS.get(k, 0), 4)
                for k in DEFAULT_WEIGHTS
            },
            "stats": stats,
            "total_weight_updates": total_updates,
            "rf_model_status": rf_status,
            "symbol_performance": symbol_stats,
        }

    # ------------------------------------------------------------------
    # 7. Sinyal Filtreleme (strateji entegrasyonu)
    # ------------------------------------------------------------------
    def should_trade(
        self,
        symbol: str,
        indicator_scores: Dict[str, float],
        composite_score: float,
    ) -> Tuple[bool, str]:
        """
        ML modeline göre bu sinyale güvenilmeli mi?
        Returns: (trade_allowed, reason)
        """
        # 1. Sembol performans kontrolü: çok kötü geçmişi olan sembolü filtrele
        sym_stats = self.get_symbol_stats(symbol)
        if sym_stats and sym_stats["total_trades"] >= 5:
            win_rate = sym_stats["wins"] / sym_stats["total_trades"]
            if win_rate < 0.2 and sym_stats["total_pnl"] < 0:
                return False, f"ml_symbol_blacklist (win_rate={win_rate:.0%}, pnl={sym_stats['total_pnl']:.2f})"

        # 2. RF model tahmini
        rf_pred = self.predict_signal_quality(indicator_scores)
        if rf_pred and rf_pred["win_probability"] < 0.35:
            return False, f"ml_rf_reject (win_prob={rf_pred['win_probability']:.0%})"

        return True, "ml_approved"
