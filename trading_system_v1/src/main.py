from __future__ import annotations

import json
import signal
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta

# Türkiye saat dilimi (UTC+3)
_TZ_TR = timezone(timedelta(hours=3))

from core.config import AppConfig
from core.models import MarketType, OrderRequest, Side, SignalType
from data.binance_spot import BinanceSpotAdapter
from data.yahoo_bist import YahooBISTAdapter
from database.db import (
    _conn as db_conn,
    get_open_positions,
    get_portfolio,
    increment_trade_stats,
    init_db,
    init_portfolio,
    log_signal,
    record_trade,
    save_daily_snapshot,
    update_portfolio_cash,
    upsert_position,
)
from engine.scanner import MarketScanner
from execution.paper_broker import PaperBroker
from ml.learner import AdaptiveLearner
from risk.rules import RiskManager
from strategies.aggressive_momentum import AggressiveMomentumStrategy
from utils.logging_utils import get_logger


logger = get_logger("trading_system")

SEPARATOR = "=" * 70


def _print_header(title: str) -> None:
    logger.info("")
    logger.info(SEPARATOR)
    logger.info("  %s", title)
    logger.info(SEPARATOR)


def _ml_record_closure(
    learner: AdaptiveLearner,
    symbol: str,
    market: str,
    direction: str,
    pnl: float,
    pnl_pct: float,
    all_results: list,
) -> None:
    """Kapanan bir işlem için ML öğrenme kaydı oluştur."""


def _is_bist_open() -> bool:
    """BIST açık mı kontrol et (Pazartesi-Cuma, 10:00-18:00 Türkiye saati)."""
    now_tr = datetime.now(_TZ_TR)
    # Hafta sonu: Cumartesi=5, Pazar=6
    if now_tr.weekday() >= 5:
        return False
    # Saat kontrolü: 10:00 - 18:00
    if now_tr.hour < 10 or now_tr.hour >= 18:
        return False
    return True
    # Son sinyalden indikatör skorlarını bul
    indicator_scores = {}
    composite_score = 0.0
    confidence = 0.0
    for sig, _ in all_results:
        if sig.symbol == symbol:
            indicator_scores = sig.metadata.get("indicator_scores", {})
            composite_score = sig.metadata.get("composite_score", 0.0)
            confidence = sig.metadata.get("confidence_pct", 0.0)
            break

    if indicator_scores:
        learner.record_outcome(
            symbol=symbol,
            market=market,
            direction=direction,
            pnl=pnl,
            pnl_pct=pnl_pct,
            indicator_scores=indicator_scores,
            composite_score=composite_score,
            confidence=confidence,
        )


def run_cycle() -> None:
    config = AppConfig()

    # --- DB init ---
    init_db()
    init_portfolio(config.risk.capital_tl)

    # --- ML Learner başlat ve öğrenilmiş ağırlıkları yükle ---
    learner = AdaptiveLearner(db_conn)
    learned_weights = learner.get_learned_weights()
    strategy = AggressiveMomentumStrategy(learned_weights=learned_weights)
    risk = RiskManager(config.risk)

    logger.info("🧠 ML Ağırlıklar: %s", {k: f"{v:.3f}" for k, v in learned_weights.items()})

    portfolio = get_portfolio()
    cash = portfolio["cash"] if portfolio else config.risk.capital_tl

    broker = PaperBroker(cash=cash)

    # Restore open positions into broker
    db_positions = get_open_positions()
    for pos in db_positions:
        direction = pos.get("direction", "long")
        if direction == "short":
            broker.short_positions[pos["symbol"]] = {
                "quantity": pos["quantity"],
                "entry_price": pos["avg_price"],
                "market": MarketType(pos["market"]),
                "leverage": pos.get("leverage") or 1,
            }
        else:
            from core.models import Position
            broker.positions[pos["symbol"]] = Position(
                symbol=pos["symbol"],
                market=MarketType(pos["market"]),
                quantity=pos["quantity"],
                avg_price=pos["avg_price"],
            )

    # --- Market data adapters ---
    crypto_adapter = BinanceSpotAdapter(config=config)
    bist_adapter = YahooBISTAdapter()

    crypto_scanner = MarketScanner(crypto_adapter, strategy, MarketType.CRYPTO)
    bist_scanner = MarketScanner(bist_adapter, strategy, MarketType.BIST)

    _print_header("PIYASA TARAMASI BAŞLIYOR")
    total_open = len(broker.positions) + len(broker.short_positions)
    logger.info("Sermaye: %.2f TL | Pozisyon: %d (L:%d S:%d) | Strateji: %s",
                cash, total_open, len(broker.positions), len(broker.short_positions), strategy.name)

    # --- Scan ---
    crypto_results = crypto_scanner.scan(
        symbols=config.scanner.crypto_symbols,
        interval=config.scanner.interval,
        limit=config.scanner.lookback_bars,
    )
    bist_results = bist_scanner.scan(
        symbols=config.scanner.bist_symbols,
        interval=config.scanner.bist_interval,
        limit=config.scanner.lookback_bars,
    )

    # --- Print & log signals ---
    all_results = []
    for label, results in [("CRYPTO (Binance)", crypto_results), ("BIST (Borsa İstanbul)", bist_results)]:
        _print_header(f"{label} TARAMA SONUÇLARI")
        for signal, df in results:
            last_close = float(df.iloc[-1]["close"])
            meta = signal.metadata
            composite = meta.get("composite_score", 0.0)
            confidence = meta.get("confidence_pct", 0.0)
            bull_count = meta.get("bullish_indicators", 0)
            bear_count = meta.get("bearish_indicators", 0)

            signal_emoji = "🟢" if signal.signal == SignalType.LONG else (
                "🔴" if signal.signal == SignalType.SHORT else "⚪"
            )

            logger.info(
                "%s %s | Sinyal: %s | Skor: %.1f | Güven: %.0f%% | "
                "Boğa: %d | Ayı: %d | Fiyat: %.4f",
                signal_emoji, signal.symbol, signal.signal.value.upper(),
                composite, confidence, bull_count, bear_count, last_close,
            )

            if signal.signal != SignalType.FLAT:
                indicator_scores = meta.get("indicator_scores", {})
                for ind_name, ind_score in indicator_scores.items():
                    direction = "↑" if ind_score > 0 else ("↓" if ind_score < 0 else "→")
                    logger.info("    %s %s: %.1f", direction, ind_name, ind_score)

            # Log signal to DB
            log_signal(
                symbol=signal.symbol,
                market=signal.market.value,
                signal_type=signal.signal.value,
                score=signal.score,
                strategy=signal.strategy_name,
                composite_score=composite,
                confidence=confidence,
                details=json.dumps(meta, default=str),
            )

            all_results.append((signal, df))

    # --- Execute trades ---
    _print_header("İŞLEM KARARLARI")

    # Sinyal haritası: mevcut döngüdeki sinyal yönlerini hızlı bul
    signal_map = {}
    for signal, df in all_results:
        signal_map[signal.symbol] = signal

    trailing_pct = config.risk.trailing_stop_pct  # %1.5

    # ── LONG POZİSYON YÖNETİMİ ──
    for sym, pos in list(broker.positions.items()):
        db_pos = next((p for p in db_positions if p["symbol"] == sym), None)
        if not db_pos:
            continue
        current_price = None
        for signal, df in all_results:
            if signal.symbol == sym:
                current_price = float(df.iloc[-1]["close"])
                break
        if current_price is None:
            continue

        pnl_pct = (current_price - pos.avg_price) / pos.avg_price if pos.avg_price > 0 else 0.0
        sl = db_pos.get("stop_loss") or 0
        tp = db_pos.get("take_profit") or 0
        highest = db_pos.get("highest_price") or pos.avg_price
        partial_taken = db_pos.get("partial_taken") or 0

        # ─ Trailing stop: en yüksek fiyatı güncelle ve stop'u yukarı çek ─
        if current_price > highest:
            highest = current_price
        trailing_stop = highest * (1 - trailing_pct)
        # Trailing stop, orijinal stop-loss'tan yüksekse onu kullan
        effective_sl = max(sl, trailing_stop) if sl else trailing_stop

        # ─ 1. Stop-loss / Trailing stop hit ─
        if effective_sl and current_price <= effective_sl:
            loss = (current_price - pos.avg_price) * pos.quantity
            loss_pct = pnl_pct * 100
            reason = "trailing_stop" if trailing_stop >= sl else "stop_loss"
            logger.info("🛑 %s: %s | Giriş: %.4f | Çıkış: %.4f | P&L: %+.2f TL (%+.2f%%)",
                        reason.upper().replace("_", " "), sym, pos.avg_price, current_price, loss, loss_pct)
            order = OrderRequest(symbol=sym, market=pos.market, side=Side.SELL, quantity=pos.quantity)
            fill = broker.submit_market_order(order=order, mark_price=current_price)
            record_trade(sym, pos.market.value, "sell", fill.quantity, fill.price, fill.fee,
                         strategy=strategy.name, metadata=json.dumps({"reason": reason}))
            increment_trade_stats(won=loss >= 0, pnl=loss)
            _ml_record_closure(learner, sym, pos.market.value, "long", loss, loss_pct, all_results)
            upsert_position(sym, pos.market.value, 0, 0)
            continue

        # ─ 2. Take-profit hit ─
        if tp and current_price >= tp:
            profit = (current_price - pos.avg_price) * pos.quantity
            profit_pct = pnl_pct * 100
            logger.info("🎯 TAKE-PROFIT: %s | Giriş: %.4f | Çıkış: %.4f | Kâr: %+.2f TL (%+.2f%%)",
                        sym, pos.avg_price, current_price, profit, profit_pct)
            order = OrderRequest(symbol=sym, market=pos.market, side=Side.SELL, quantity=pos.quantity)
            fill = broker.submit_market_order(order=order, mark_price=current_price)
            record_trade(sym, pos.market.value, "sell", fill.quantity, fill.price, fill.fee,
                         strategy=strategy.name, metadata=json.dumps({"reason": "take_profit"}))
            increment_trade_stats(won=True, pnl=profit)
            _ml_record_closure(learner, sym, pos.market.value, "long", profit, profit_pct, all_results)
            upsert_position(sym, pos.market.value, 0, 0)
            continue

        # ─ 3. Kısmi kâr alma: TP'nin yarısına ulaştıysa pozisyonun %50'sini kapat ─
        if not partial_taken and tp and pos.avg_price > 0:
            half_tp = pos.avg_price + (tp - pos.avg_price) * 0.5
            if current_price >= half_tp:
                sell_qty = round(pos.quantity * 0.5, 8)
                if sell_qty > 0:
                    partial_profit = (current_price - pos.avg_price) * sell_qty
                    logger.info("💰 KISMİ KÂR: %s | %d%% pozisyon kapatıldı | Kâr: %+.2f TL",
                                sym, 50, partial_profit)
                    order = OrderRequest(symbol=sym, market=pos.market, side=Side.SELL, quantity=sell_qty)
                    fill = broker.submit_market_order(order=order, mark_price=current_price)
                    record_trade(sym, pos.market.value, "sell", fill.quantity, fill.price, fill.fee,
                                 strategy=strategy.name, metadata=json.dumps({"reason": "partial_profit"}))
                    # Kalan pozisyonu güncelle, stop'u giriş fiyatına çek (breakeven)
                    remaining_qty = pos.quantity  # update_from_fill zaten güncelledi
                    new_sl = max(effective_sl, pos.avg_price)  # breakeven stop
                    upsert_position(sym, pos.market.value, remaining_qty, pos.avg_price,
                                    stop_loss=new_sl, take_profit=tp, direction="long",
                                    leverage=db_pos.get("leverage") or 1,
                                    highest_price=highest, partial_taken=1)
                    continue

        # ─ 4. Sinyal tersi kapanış: LONG'dayken SHORT sinyali gelirse kapat ─
        cur_signal = signal_map.get(sym)
        if cur_signal and cur_signal.signal == SignalType.SHORT and abs(cur_signal.score) >= 45:
            pnl_val = (current_price - pos.avg_price) * pos.quantity
            logger.info("🔄 SİNYAL TERSİ: %s | LONG → SHORT sinyal (skor: %.1f) | P&L: %+.2f TL",
                        sym, cur_signal.score, pnl_val)
            order = OrderRequest(symbol=sym, market=pos.market, side=Side.SELL, quantity=pos.quantity)
            fill = broker.submit_market_order(order=order, mark_price=current_price)
            record_trade(sym, pos.market.value, "sell", fill.quantity, fill.price, fill.fee,
                         strategy=strategy.name, metadata=json.dumps({"reason": "signal_reversal", "new_signal": "short", "score": cur_signal.score}))
            increment_trade_stats(won=pnl_val >= 0, pnl=pnl_val)
            _ml_record_closure(learner, sym, pos.market.value, "long", pnl_val, pnl_pct * 100, all_results)
            upsert_position(sym, pos.market.value, 0, 0)
            continue

        # ─ 5. Zaman bazlı çıkış: 30 döngüden (5 gün) fazla açık kalan pozisyonu kapat ─
        entry_time_str = db_pos.get("entry_time")
        if entry_time_str:
            try:
                entry_dt = datetime.fromisoformat(entry_time_str)
                hours_open = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
                if hours_open > 120 and pnl_pct < 0.01:  # 5 gün+ ve neredeyse kârsız
                    pnl_val = (current_price - pos.avg_price) * pos.quantity
                    logger.info("⏰ ZAMAN AŞIMI: %s | %.0f saat açık | P&L: %+.2f TL — kapatılıyor",
                                sym, hours_open, pnl_val)
                    order = OrderRequest(symbol=sym, market=pos.market, side=Side.SELL, quantity=pos.quantity)
                    fill = broker.submit_market_order(order=order, mark_price=current_price)
                    record_trade(sym, pos.market.value, "sell", fill.quantity, fill.price, fill.fee,
                                 strategy=strategy.name, metadata=json.dumps({"reason": "time_exit", "hours_open": round(hours_open, 1)}))
                    increment_trade_stats(won=pnl_val >= 0, pnl=pnl_val)
                    _ml_record_closure(learner, sym, pos.market.value, "long", pnl_val, pnl_pct * 100, all_results)
                    upsert_position(sym, pos.market.value, 0, 0)
                    continue
            except (ValueError, TypeError):
                pass

        # Hiçbir çıkış tetiklenmedi — trailing bilgisini güncelle
        upsert_position(sym, pos.market.value, pos.quantity, pos.avg_price,
                        stop_loss=round(effective_sl, 6), take_profit=tp, direction="long",
                        leverage=db_pos.get("leverage") or 1,
                        highest_price=highest, partial_taken=partial_taken)

        logger.info("📊 AÇIK POZİSYON: %s | Giriş: %.4f | Şimdi: %.4f | En Yüksek: %.4f | "
                    "Trailing SL: %.4f | P&L: %+.2f%%",
                    sym, pos.avg_price, current_price, highest, effective_sl, pnl_pct * 100)

    # ── SHORT POZİSYON YÖNETİMİ ──
    for sym, short_data in list(broker.short_positions.items()):
        db_pos = next((p for p in db_positions if p["symbol"] == sym), None)
        if not db_pos:
            continue
        current_price = None
        for signal, df in all_results:
            if signal.symbol == sym:
                current_price = float(df.iloc[-1]["close"])
                break
        if current_price is None:
            continue

        entry = short_data["entry_price"]
        qty = short_data["quantity"]
        pnl_pct = (entry - current_price) / entry if entry > 0 else 0.0
        sl = db_pos.get("stop_loss") or 0
        tp = db_pos.get("take_profit") or 0
        lowest = db_pos.get("lowest_price") or entry
        partial_taken = db_pos.get("partial_taken") or 0

        # Trailing stop SHORT: en düşük fiyatı güncelle ve stop'u aşağı çek
        if current_price < lowest:
            lowest = current_price
        trailing_stop = lowest * (1 + trailing_pct)
        effective_sl = min(sl, trailing_stop) if sl else trailing_stop

        # ─ 1. SHORT stop-loss / trailing stop: price went UP beyond stop ─
        if effective_sl and current_price >= effective_sl:
            loss = (entry - current_price) * qty
            loss_pct = pnl_pct * 100
            reason = "trailing_stop" if trailing_stop <= sl else "short_stop_loss"
            logger.info("🛑 %s: %s | Giriş: %.4f | Çıkış: %.4f | P&L: %+.2f TL",
                        reason.upper().replace("_", " "), sym, entry, current_price, loss)
            order = OrderRequest(symbol=sym, market=short_data["market"], side=Side.BUY, quantity=qty)
            fill = broker.submit_market_order(order=order, mark_price=current_price)
            record_trade(sym, short_data["market"].value, "buy_to_cover", fill.quantity, fill.price, fill.fee,
                         strategy=strategy.name, metadata=json.dumps({"reason": reason}))
            increment_trade_stats(won=loss >= 0, pnl=loss)
            _ml_record_closure(learner, sym, short_data["market"].value, "short", loss, loss_pct, all_results)
            upsert_position(sym, short_data["market"].value, 0, 0)
            continue

        # ─ 2. SHORT take-profit: price went DOWN to target ─
        if tp and current_price <= tp:
            profit = (entry - current_price) * qty
            profit_pct = pnl_pct * 100
            logger.info("🎯 SHORT TAKE-PROFIT: %s | Giriş: %.4f | Çıkış: %.4f | Kâr: %+.2f TL",
                        sym, entry, current_price, profit)
            order = OrderRequest(symbol=sym, market=short_data["market"], side=Side.BUY, quantity=qty)
            fill = broker.submit_market_order(order=order, mark_price=current_price)
            record_trade(sym, short_data["market"].value, "buy_to_cover", fill.quantity, fill.price, fill.fee,
                         strategy=strategy.name, metadata=json.dumps({"reason": "short_take_profit"}))
            increment_trade_stats(won=True, pnl=profit)
            _ml_record_closure(learner, sym, short_data["market"].value, "short", profit, profit_pct, all_results)
            upsert_position(sym, short_data["market"].value, 0, 0)
            continue

        # ─ 3. Kısmi kâr alma (SHORT) ─
        if not partial_taken and tp and entry > 0:
            half_tp = entry - (entry - tp) * 0.5
            if current_price <= half_tp:
                cover_qty = round(qty * 0.5, 8)
                if cover_qty > 0:
                    partial_profit = (entry - current_price) * cover_qty
                    logger.info("💰 SHORT KISMİ KÂR: %s | %d%% kapatıldı | Kâr: %+.2f TL",
                                sym, 50, partial_profit)
                    order = OrderRequest(symbol=sym, market=short_data["market"], side=Side.BUY, quantity=cover_qty)
                    fill = broker.submit_market_order(order=order, mark_price=current_price)
                    record_trade(sym, short_data["market"].value, "buy_to_cover", fill.quantity, fill.price, fill.fee,
                                 strategy=strategy.name, metadata=json.dumps({"reason": "short_partial_profit"}))
                    short_data["quantity"] -= cover_qty
                    new_sl = min(effective_sl, entry)  # breakeven stop
                    upsert_position(sym, short_data["market"].value, short_data["quantity"], entry,
                                    stop_loss=new_sl, take_profit=tp, direction="short",
                                    leverage=short_data.get("leverage") or 1,
                                    lowest_price=lowest, partial_taken=1)
                    continue

        # ─ 4. Sinyal tersi kapanış (SHORT → LONG sinyal) ─
        cur_signal = signal_map.get(sym)
        if cur_signal and cur_signal.signal == SignalType.LONG and abs(cur_signal.score) >= 45:
            pnl_val = (entry - current_price) * qty
            logger.info("🔄 SİNYAL TERSİ: %s | SHORT → LONG sinyal (skor: %.1f) | P&L: %+.2f TL",
                        sym, cur_signal.score, pnl_val)
            order = OrderRequest(symbol=sym, market=short_data["market"], side=Side.BUY, quantity=qty)
            fill = broker.submit_market_order(order=order, mark_price=current_price)
            record_trade(sym, short_data["market"].value, "buy_to_cover", fill.quantity, fill.price, fill.fee,
                         strategy=strategy.name, metadata=json.dumps({"reason": "signal_reversal", "new_signal": "long", "score": cur_signal.score}))
            increment_trade_stats(won=pnl_val >= 0, pnl=pnl_val)
            _ml_record_closure(learner, sym, short_data["market"].value, "short", pnl_val, pnl_pct * 100, all_results)
            upsert_position(sym, short_data["market"].value, 0, 0)
            continue

        # ─ 5. Zaman bazlı çıkış (SHORT) ─
        entry_time_str = db_pos.get("entry_time")
        if entry_time_str:
            try:
                entry_dt = datetime.fromisoformat(entry_time_str)
                hours_open = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
                if hours_open > 120 and pnl_pct < 0.01:
                    pnl_val = (entry - current_price) * qty
                    logger.info("⏰ SHORT ZAMAN AŞIMI: %s | %.0f saat açık | P&L: %+.2f TL — kapatılıyor",
                                sym, hours_open, pnl_val)
                    order = OrderRequest(symbol=sym, market=short_data["market"], side=Side.BUY, quantity=qty)
                    fill = broker.submit_market_order(order=order, mark_price=current_price)
                    record_trade(sym, short_data["market"].value, "buy_to_cover", fill.quantity, fill.price, fill.fee,
                                 strategy=strategy.name, metadata=json.dumps({"reason": "time_exit", "hours_open": round(hours_open, 1)}))
                    increment_trade_stats(won=pnl_val >= 0, pnl=pnl_val)
                    _ml_record_closure(learner, sym, short_data["market"].value, "short", pnl_val, pnl_pct * 100, all_results)
                    upsert_position(sym, short_data["market"].value, 0, 0)
                    continue
            except (ValueError, TypeError):
                pass

        # Trailing bilgisini güncelle
        upsert_position(sym, short_data["market"].value, qty, entry,
                        stop_loss=round(effective_sl, 6), take_profit=tp, direction="short",
                        leverage=short_data.get("leverage") or 1,
                        lowest_price=lowest, partial_taken=partial_taken)

        logger.info("📊 AÇIK SHORT: %s | Giriş: %.4f | Şimdi: %.4f | En Düşük: %.4f | "
                    "Trailing SL: %.4f | P&L: %+.2f%%",
                    sym, entry, current_price, lowest, effective_sl, pnl_pct * 100)

    # New trade signals (LONG + SHORT)
    top_candidates = []
    for signal, df in all_results:
        if signal.signal in (SignalType.LONG, SignalType.SHORT):
            if signal.symbol not in broker.positions and signal.symbol not in broker.short_positions:
                top_candidates.append((signal, df))

    top_candidates = sorted(top_candidates, key=lambda item: abs(item[0].score), reverse=True)

    for signal, df in top_candidates[:3]:  # max 3 new trades per cycle
        total_open = len(broker.positions) + len(broker.short_positions)

        # BIST piyasa saati kontrolü: borsa kapalıyken BIST işlemi açma
        if signal.market == MarketType.BIST and not _is_bist_open():
            now_tr = datetime.now(_TZ_TR)
            logger.info("🕐 BIST KAPALI: %s — Borsa İstanbul şu an kapalı (%s). İşlem açılmadı.",
                        signal.symbol, now_tr.strftime("%A %H:%M"))
            continue

        decision = risk.evaluate(signal, current_open_positions=total_open)
        last_close = float(df.iloc[-1]["close"])

        if not decision.allowed:
            logger.info("❌ RED: %s — %s", signal.symbol, decision.reason)
            continue

        # ML filtreleme: modelin onayı var mı?
        indicator_scores = signal.metadata.get("indicator_scores", {})
        ml_allowed, ml_reason = learner.should_trade(
            signal.symbol, indicator_scores, signal.score,
        )
        if not ml_allowed:
            logger.info("🧠 ML RED: %s — %s", signal.symbol, ml_reason)
            continue

        # RF tahmin bilgisi
        rf_pred = learner.predict_signal_quality(indicator_scores)
        if rf_pred:
            logger.info("🧠 ML TAHMİN: %s | Kazanma: %.0f%% | Güven: %.0f%%",
                        signal.symbol, rf_pred["win_probability"] * 100, rf_pred["model_confidence"])

        quantity = decision.max_position_notional / last_close if last_close else 0.0
        if quantity <= 0:
            continue

        is_short = signal.signal == SignalType.SHORT
        side = Side.SELL if is_short else Side.BUY

        # Kripto kaldıraç
        leverage = config.risk.crypto_leverage if signal.market == MarketType.CRYPTO else 1

        # Mevcut nakite göre pozisyon boyutunu ayarla
        notional = quantity * last_close
        margin = notional / leverage
        fee_est = notional * broker.fee_rate
        if (margin + fee_est) > broker.cash:
            # Nakitin yettiği kadar pozisyon aç: margin + fee = notional*(1/lev + fee_rate) <= cash
            max_notional = broker.cash * 0.995 / (1.0 / leverage + broker.fee_rate)
            quantity = max_notional / last_close
            if quantity <= 0:
                logger.info("❌ YETERSIZ BAKİYE: %s — nakit yetersiz", signal.symbol)
                continue
            logger.info("⚠️ POZİSYON KÜÇÜLTÜLDÜ: %s — mevcut nakite göre ayarlandı (₺%.2f)", signal.symbol, max_notional)

        order = OrderRequest(
            symbol=signal.symbol,
            market=signal.market,
            side=side,
            quantity=round(quantity, 8),
            metadata={
                "strategy": signal.strategy_name,
                "score": signal.score,
                "stop_loss_pct": decision.stop_loss_pct,
                "take_profit_pct": decision.take_profit_pct,
                "direction": "SHORT" if is_short else "LONG",
            },
        )

        try:
            fill = broker.submit_market_order(order=order, mark_price=last_close, leverage=leverage)
        except ValueError as e:
            logger.info("❌ YETERSIZ BAKİYE: %s — %s", signal.symbol, e)
            continue

        if is_short:
            stop_price = round(last_close * (1 + decision.stop_loss_pct), 4)
            tp_price = round(last_close * (1 - decision.take_profit_pct), 4)
        else:
            stop_price = round(last_close * (1 - decision.stop_loss_pct), 4)
            tp_price = round(last_close * (1 + decision.take_profit_pct), 4)

        direction_label = "SHORT 🔴" if is_short else "LONG 🟢"
        logger.info("")
        logger.info("✅ İŞLEM AÇILDI — %s", direction_label)
        logger.info("   Sembol     : %s", fill.symbol)
        logger.info("   Yön        : %s", "SHORT (SATIŞ)" if is_short else "LONG (ALIŞ)")
        logger.info("   Fiyat      : %.4f", fill.price)
        logger.info("   Miktar     : %.8f", fill.quantity)
        logger.info("   Tutar      : %.2f TL", fill.quantity * fill.price)
        logger.info("   Komisyon   : %.4f TL", fill.fee)
        logger.info("   Stop-Loss  : %.4f (%%%.1f)", stop_price, decision.stop_loss_pct * 100)
        logger.info("   Take-Profit: %.4f (%%%.1f)", tp_price, decision.take_profit_pct * 100)
        logger.info("   Skor       : %.1f | Güven: %.0f%%",
                    signal.score, signal.metadata.get("confidence_pct", 0))

        # Record to DB
        record_trade(
            symbol=fill.symbol,
            market=signal.market.value,
            side="short" if is_short else "buy",
            quantity=fill.quantity,
            price=fill.price,
            fee=fill.fee,
            strategy=signal.strategy_name,
            signal_score=signal.score,
            stop_loss_pct=decision.stop_loss_pct,
            take_profit_pct=decision.take_profit_pct,
            metadata=json.dumps(signal.metadata, default=str),
        )
        upsert_position(
            symbol=fill.symbol,
            market=signal.market.value,
            quantity=fill.quantity,
            avg_price=fill.price,
            stop_loss=stop_price,
            take_profit=tp_price,
            direction="short" if is_short else "long",
            leverage=leverage,
            highest_price=None if is_short else fill.price,
            lowest_price=fill.price if is_short else None,
            partial_taken=0,
        )

    # --- Portfolio summary ---
    update_portfolio_cash(broker.cash)

    # Yeni açılan pozisyonlar dahil güncel DB verisini yükle
    db_positions_fresh = get_open_positions()

    positions_value = 0.0  # margin + unrealized P&L
    # SHORT positions: margin + unrealized P&L
    short_unrealized = 0.0
    for sym, short_data in broker.short_positions.items():
        for signal, df in all_results:
            if signal.symbol == sym:
                current_price = float(df.iloc[-1]["close"])
                lev = short_data.get("leverage", 1)
                entry_notional = short_data["entry_price"] * short_data["quantity"]
                margin = entry_notional / lev if lev > 1 else entry_notional
                pnl = (short_data["entry_price"] - current_price) * short_data["quantity"]
                short_unrealized += margin + pnl
                break

    # LONG positions: margin + unrealized P&L
    for sym, pos in broker.positions.items():
        for signal, df in all_results:
            if signal.symbol == sym:
                current_price = float(df.iloc[-1]["close"])
                # Kaldıraç bilgisi: güncel DB'den
                db_p = next((p for p in db_positions_fresh if p["symbol"] == sym), None)
                lev = (db_p.get("leverage") or 1) if db_p else 1
                entry_notional = pos.avg_price * pos.quantity
                margin = entry_notional / lev if lev > 1 else entry_notional
                pnl = (current_price - pos.avg_price) * pos.quantity
                positions_value += margin + pnl
                break

    total_equity = broker.cash + positions_value + short_unrealized
    portfolio = get_portfolio()
    initial = portfolio["initial_capital"] if portfolio else config.risk.capital_tl
    total_pnl = total_equity - initial

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    save_daily_snapshot(
        date_str=today_str,
        cash=broker.cash,
        positions_value=positions_value + short_unrealized,
        total_equity=total_equity,
        daily_pnl=0.0,
        total_pnl=total_pnl,
    )

    total_open = len(broker.positions) + len(broker.short_positions)
    _print_header("PORTFÖY ÖZETİ")
    logger.info("  Başlangıç Sermayesi : %.2f TL", initial)
    logger.info("  Nakit               : %.2f TL", broker.cash)
    logger.info("  LONG Pozisyon Değeri: %.2f TL", positions_value)
    logger.info("  SHORT Gerçekleşmemiş: %+.2f TL", short_unrealized)
    logger.info("  Toplam Varlık       : %.2f TL", total_equity)
    pnl_pct = (total_pnl / initial * 100) if initial else 0
    pnl_symbol = "📈" if total_pnl >= 0 else "📉"
    logger.info("  Toplam P&L          : %s %+.2f TL (%+.2f%%)", pnl_symbol, total_pnl, pnl_pct)
    logger.info("  Açık Pozisyon Sayısı: %d (LONG: %d, SHORT: %d)",
                total_open, len(broker.positions), len(broker.short_positions))

    if broker.positions:
        logger.info("")
        logger.info("  LONG Pozisyonlar:")
        for sym, pos in broker.positions.items():
            for signal, df in all_results:
                if signal.symbol == sym:
                    cp = float(df.iloc[-1]["close"])
                    unreal = (cp - pos.avg_price) * pos.quantity
                    unreal_pct = ((cp - pos.avg_price) / pos.avg_price * 100) if pos.avg_price else 0
                    logger.info("    🟢 %s | Giriş: %.4f | Şimdi: %.4f | P&L: %+.2f TL (%+.2f%%)",
                                sym, pos.avg_price, cp, unreal, unreal_pct)
                    break

    if broker.short_positions:
        logger.info("")
        logger.info("  SHORT Pozisyonlar:")
        for sym, short_data in broker.short_positions.items():
            for signal, df in all_results:
                if signal.symbol == sym:
                    cp = float(df.iloc[-1]["close"])
                    entry = short_data["entry_price"]
                    qty = short_data["quantity"]
                    unreal = (entry - cp) * qty
                    unreal_pct = ((entry - cp) / entry * 100) if entry else 0
                    logger.info("    🔴 %s | Giriş: %.4f | Şimdi: %.4f | P&L: %+.2f TL (%+.2f%%)",
                                sym, entry, cp, unreal, unreal_pct)
                    break

    logger.info("")
    logger.info(SEPARATOR)
    logger.info("  Tarama tamamlandı — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(SEPARATOR)


# ---------------------------------------------------------------------------
# MONITOR MODU — Sadece açık pozisyonları kontrol et (hafif, hızlı)
# Her 30 dakikada GitHub Actions ile çalışır
# ---------------------------------------------------------------------------

def _fetch_binance_prices(symbols: list[str]) -> dict[str, float]:
    """Binance'den anlık fiyatları toplu çek — çok hızlı, tek API çağrısı."""
    import requests
    prices: dict[str, float] = {}
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            timeout=10,
        )
        resp.raise_for_status()
        all_prices = {item["symbol"]: float(item["price"]) for item in resp.json()}
        for sym in symbols:
            if sym in all_prices:
                prices[sym] = all_prices[sym]
    except Exception as e:
        logger.warning("Binance fiyat çekme hatası: %s", e)
        # Fallback: tek tek çek
        for sym in symbols:
            try:
                r = requests.get(
                    f"https://api.binance.com/api/v3/ticker/price?symbol={sym}",
                    timeout=5,
                )
                if r.ok:
                    prices[sym] = float(r.json()["price"])
            except Exception:
                pass
    return prices


def _fetch_bist_prices(symbols: list[str]) -> dict[str, float]:
    """Yahoo Finance'den BIST anlık fiyatları çek."""
    import yfinance as yf
    prices: dict[str, float] = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="1d")
            if not hist.empty:
                prices[sym] = float(hist.iloc[-1]["Close"])
        except Exception:
            pass
    return prices


def run_monitor() -> None:
    """
    Hafif pozisyon monitörü — sadece açık pozisyonları kontrol eder.
    Yeni işlem AÇMAZ, sadece mevcut pozisyonları yönetir:
    - Trailing stop güncelle
    - Stop-loss / Take-profit kontrol
    - Kısmi kâr al
    - Zaman aşımı kontrolü
    ~20-30 saniyede tamamlanır.
    """
    config = AppConfig()
    init_db()
    init_portfolio(config.risk.capital_tl)

    learner = AdaptiveLearner(db_conn)

    portfolio = get_portfolio()
    cash = portfolio["cash"] if portfolio else config.risk.capital_tl
    broker = PaperBroker(cash=cash)

    db_positions = get_open_positions()
    if not db_positions:
        logger.info("⚡ MONİTÖR: Açık pozisyon yok, çıkılıyor.")
        return

    # Pozisyonları broker'a yükle
    for pos in db_positions:
        direction = pos.get("direction", "long")
        if direction == "short":
            broker.short_positions[pos["symbol"]] = {
                "quantity": pos["quantity"],
                "entry_price": pos["avg_price"],
                "market": MarketType(pos["market"]),
                "leverage": pos.get("leverage") or 1,
            }
        else:
            from core.models import Position
            broker.positions[pos["symbol"]] = Position(
                symbol=pos["symbol"],
                market=MarketType(pos["market"]),
                quantity=pos["quantity"],
                avg_price=pos["avg_price"],
            )

    # Açık pozisyonların sembollerini topla
    crypto_symbols = [p["symbol"] for p in db_positions if p["market"] == "crypto"]
    bist_symbols = [p["symbol"] for p in db_positions if p["market"] == "bist"]

    _print_header("⚡ HIZLI MONİTÖR — Pozisyon Kontrolü")
    logger.info("Açık pozisyon: %d (Kripto: %d, BIST: %d)",
                len(db_positions), len(crypto_symbols), len(bist_symbols))

    # Anlık fiyatları çek (tek API çağrısı — çok hızlı)
    prices: dict[str, float] = {}
    if crypto_symbols:
        prices.update(_fetch_binance_prices(crypto_symbols))
    if bist_symbols:
        prices.update(_fetch_bist_prices(bist_symbols))

    if not prices:
        logger.warning("⚠️ Hiçbir fiyat alınamadı, monitor çıkıyor.")
        return

    trailing_pct = config.risk.trailing_stop_pct
    strategy = AggressiveMomentumStrategy()
    closed_count = 0

    # ── LONG POZİSYON KONTROLÜ ──
    for sym, pos in list(broker.positions.items()):
        current_price = prices.get(sym)
        if current_price is None:
            logger.info("⚠️ %s fiyat alınamadı, atlanıyor", sym)
            continue

        db_pos = next((p for p in db_positions if p["symbol"] == sym), None)
        if not db_pos:
            continue

        pnl_pct = (current_price - pos.avg_price) / pos.avg_price if pos.avg_price > 0 else 0.0
        sl = db_pos.get("stop_loss") or 0
        tp = db_pos.get("take_profit") or 0
        highest = db_pos.get("highest_price") or pos.avg_price
        partial_taken = db_pos.get("partial_taken") or 0

        # Trailing stop güncelle
        if current_price > highest:
            highest = current_price
        trailing_stop = highest * (1 - trailing_pct)
        effective_sl = max(sl, trailing_stop) if sl else trailing_stop

        # 1. Stop-loss / Trailing stop
        if effective_sl and current_price <= effective_sl:
            loss = (current_price - pos.avg_price) * pos.quantity
            loss_pct = pnl_pct * 100
            reason = "trailing_stop" if trailing_stop >= sl else "stop_loss"
            logger.info("🛑 MONİTÖR %s: %s | Giriş: %.4f | Çıkış: %.4f | P&L: %+.2f TL (%+.2f%%)",
                        reason.upper(), sym, pos.avg_price, current_price, loss, loss_pct)
            order = OrderRequest(symbol=sym, market=pos.market, side=Side.SELL, quantity=pos.quantity)
            fill = broker.submit_market_order(order=order, mark_price=current_price)
            record_trade(sym, pos.market.value, "sell", fill.quantity, fill.price, fill.fee,
                         strategy=strategy.name, metadata=json.dumps({"reason": reason, "source": "monitor"}))
            increment_trade_stats(won=loss >= 0, pnl=loss)
            upsert_position(sym, pos.market.value, 0, 0)
            closed_count += 1
            continue

        # 2. Take-profit
        if tp and current_price >= tp:
            profit = (current_price - pos.avg_price) * pos.quantity
            logger.info("🎯 MONİTÖR TP: %s | Giriş: %.4f | Çıkış: %.4f | Kâr: %+.2f TL",
                        sym, pos.avg_price, current_price, profit)
            order = OrderRequest(symbol=sym, market=pos.market, side=Side.SELL, quantity=pos.quantity)
            fill = broker.submit_market_order(order=order, mark_price=current_price)
            record_trade(sym, pos.market.value, "sell", fill.quantity, fill.price, fill.fee,
                         strategy=strategy.name, metadata=json.dumps({"reason": "take_profit", "source": "monitor"}))
            increment_trade_stats(won=True, pnl=profit)
            upsert_position(sym, pos.market.value, 0, 0)
            closed_count += 1
            continue

        # 3. Kısmi kâr alma
        if not partial_taken and tp and pos.avg_price > 0:
            half_tp = pos.avg_price + (tp - pos.avg_price) * 0.5
            if current_price >= half_tp:
                sell_qty = round(pos.quantity * 0.5, 8)
                if sell_qty > 0:
                    partial_profit = (current_price - pos.avg_price) * sell_qty
                    logger.info("💰 MONİTÖR KISMİ KÂR: %s | %d%% kapatıldı | Kâr: %+.2f TL",
                                sym, 50, partial_profit)
                    order = OrderRequest(symbol=sym, market=pos.market, side=Side.SELL, quantity=sell_qty)
                    fill = broker.submit_market_order(order=order, mark_price=current_price)
                    record_trade(sym, pos.market.value, "sell", fill.quantity, fill.price, fill.fee,
                                 strategy=strategy.name, metadata=json.dumps({"reason": "partial_profit", "source": "monitor"}))
                    remaining_qty = pos.quantity
                    new_sl = max(effective_sl, pos.avg_price)
                    upsert_position(sym, pos.market.value, remaining_qty, pos.avg_price,
                                    stop_loss=new_sl, take_profit=tp, direction="long",
                                    leverage=db_pos.get("leverage") or 1,
                                    highest_price=highest, partial_taken=1)
                    continue

        # 4. Zaman aşımı
        entry_time_str = db_pos.get("entry_time")
        if entry_time_str:
            try:
                entry_dt = datetime.fromisoformat(entry_time_str)
                hours_open = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
                if hours_open > 120 and pnl_pct < 0.01:
                    pnl_val = (current_price - pos.avg_price) * pos.quantity
                    logger.info("⏰ MONİTÖR ZAMAN: %s | %.0f saat | P&L: %+.2f TL — kapatıldı",
                                sym, hours_open, pnl_val)
                    order = OrderRequest(symbol=sym, market=pos.market, side=Side.SELL, quantity=pos.quantity)
                    fill = broker.submit_market_order(order=order, mark_price=current_price)
                    record_trade(sym, pos.market.value, "sell", fill.quantity, fill.price, fill.fee,
                                 strategy=strategy.name, metadata=json.dumps({"reason": "time_exit", "source": "monitor"}))
                    increment_trade_stats(won=pnl_val >= 0, pnl=pnl_val)
                    upsert_position(sym, pos.market.value, 0, 0)
                    closed_count += 1
                    continue
            except (ValueError, TypeError):
                pass

        # Trailing bilgisi güncelle
        upsert_position(sym, pos.market.value, pos.quantity, pos.avg_price,
                        stop_loss=round(effective_sl, 6), take_profit=tp, direction="long",
                        leverage=db_pos.get("leverage") or 1,
                        highest_price=highest, partial_taken=partial_taken)

        logger.info("📊 %s | %.4f → %.4f | TSL: %.4f | P&L: %+.2f%%",
                    sym, pos.avg_price, current_price, effective_sl, pnl_pct * 100)

    # ── SHORT POZİSYON KONTROLÜ ──
    for sym, short_data in list(broker.short_positions.items()):
        current_price = prices.get(sym)
        if current_price is None:
            continue

        db_pos = next((p for p in db_positions if p["symbol"] == sym), None)
        if not db_pos:
            continue

        entry = short_data["entry_price"]
        qty = short_data["quantity"]
        pnl_pct = (entry - current_price) / entry if entry > 0 else 0.0
        sl = db_pos.get("stop_loss") or 0
        tp = db_pos.get("take_profit") or 0
        lowest = db_pos.get("lowest_price") or entry
        partial_taken = db_pos.get("partial_taken") or 0

        if current_price < lowest:
            lowest = current_price
        trailing_stop = lowest * (1 + trailing_pct)
        effective_sl = min(sl, trailing_stop) if sl else trailing_stop

        # 1. Stop-loss / Trailing stop
        if effective_sl and current_price >= effective_sl:
            loss = (entry - current_price) * qty
            reason = "trailing_stop" if trailing_stop <= sl else "short_stop_loss"
            logger.info("🛑 MONİTÖR %s: %s | P&L: %+.2f TL", reason.upper(), sym, loss)
            order = OrderRequest(symbol=sym, market=short_data["market"], side=Side.BUY, quantity=qty)
            fill = broker.submit_market_order(order=order, mark_price=current_price)
            record_trade(sym, short_data["market"].value, "buy_to_cover", fill.quantity, fill.price, fill.fee,
                         strategy=strategy.name, metadata=json.dumps({"reason": reason, "source": "monitor"}))
            increment_trade_stats(won=loss >= 0, pnl=loss)
            upsert_position(sym, short_data["market"].value, 0, 0)
            closed_count += 1
            continue

        # 2. Take-profit
        if tp and current_price <= tp:
            profit = (entry - current_price) * qty
            logger.info("🎯 MONİTÖR SHORT TP: %s | Kâr: %+.2f TL", sym, profit)
            order = OrderRequest(symbol=sym, market=short_data["market"], side=Side.BUY, quantity=qty)
            fill = broker.submit_market_order(order=order, mark_price=current_price)
            record_trade(sym, short_data["market"].value, "buy_to_cover", fill.quantity, fill.price, fill.fee,
                         strategy=strategy.name, metadata=json.dumps({"reason": "short_take_profit", "source": "monitor"}))
            increment_trade_stats(won=True, pnl=profit)
            upsert_position(sym, short_data["market"].value, 0, 0)
            closed_count += 1
            continue

        # 3. Kısmi kâr
        if not partial_taken and tp and entry > 0:
            half_tp = entry - (entry - tp) * 0.5
            if current_price <= half_tp:
                cover_qty = round(qty * 0.5, 8)
                if cover_qty > 0:
                    logger.info("💰 MONİTÖR SHORT KISMİ: %s | %d%% kapatıldı", sym, 50)
                    order = OrderRequest(symbol=sym, market=short_data["market"], side=Side.BUY, quantity=cover_qty)
                    fill = broker.submit_market_order(order=order, mark_price=current_price)
                    record_trade(sym, short_data["market"].value, "buy_to_cover", fill.quantity, fill.price, fill.fee,
                                 strategy=strategy.name, metadata=json.dumps({"reason": "short_partial_profit", "source": "monitor"}))
                    short_data["quantity"] -= cover_qty
                    new_sl = min(effective_sl, entry)
                    upsert_position(sym, short_data["market"].value, short_data["quantity"], entry,
                                    stop_loss=new_sl, take_profit=tp, direction="short",
                                    leverage=short_data.get("leverage") or 1,
                                    lowest_price=lowest, partial_taken=1)
                    continue

        # Trailing güncelle
        upsert_position(sym, short_data["market"].value, qty, entry,
                        stop_loss=round(effective_sl, 6), take_profit=tp, direction="short",
                        leverage=short_data.get("leverage") or 1,
                        lowest_price=lowest, partial_taken=partial_taken)

        logger.info("📊 SHORT %s | %.4f → %.4f | TSL: %.4f | P&L: %+.2f%%",
                    sym, entry, current_price, effective_sl, pnl_pct * 100)

    # Nakit güncelle
    update_portfolio_cash(broker.cash)

    logger.info("")
    logger.info("⚡ MONİTÖR TAMAMLANDI — %s | Kapatılan: %d | Kalan: %d",
                datetime.now().strftime("%H:%M:%S"), closed_count,
                len(db_positions) - closed_count)


# ---------------------------------------------------------------------------
# 7/24 sürekli çalışma döngüsü
# ---------------------------------------------------------------------------
_shutdown_requested = False


def _handle_shutdown(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("⚠️  Kapatma sinyali alındı (sig=%s). Mevcut döngü bittikten sonra duracak...", signum)


def run_forever() -> None:
    """Ana trading döngüsü — 7/24 sürekli çalışır."""
    global _shutdown_requested

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    config = AppConfig()
    cycle_seconds = config.scanner.cycle_interval_minutes * 60
    cycle_count = 0
    max_consecutive_errors = 5
    consecutive_errors = 0

    logger.info(SEPARATOR)
    logger.info("  🚀 TRADİNG SİSTEMİ 7/24 MODDA BAŞLATILDI")
    logger.info("  Döngü Aralığı : %d dakika", config.scanner.cycle_interval_minutes)
    logger.info("  Başlangıç     : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(SEPARATOR)

    while not _shutdown_requested:
        cycle_count += 1
        cycle_start = time.time()

        logger.info("")
        logger.info("🔄 DÖNGÜ #%d başlıyor — %s", cycle_count, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        try:
            run_cycle()
            consecutive_errors = 0
        except KeyboardInterrupt:
            logger.info("⚠️  Kullanıcı tarafından durduruldu (Ctrl+C).")
            break
        except Exception:
            consecutive_errors += 1
            logger.error("❌ DÖNGÜ #%d HATA (%d/%d):\n%s",
                         cycle_count, consecutive_errors, max_consecutive_errors,
                         traceback.format_exc())
            if consecutive_errors >= max_consecutive_errors:
                logger.critical("🛑 Art arda %d hata! Sistem durduruluyor.", max_consecutive_errors)
                sys.exit(1)
            # Hata sonrası kısa bekleme (exponential backoff, max 10 dk)
            backoff = min(60 * consecutive_errors, 600)
            logger.info("⏳ %d saniye sonra tekrar denenecek...", backoff)
            _sleep_interruptible(backoff)
            continue

        if _shutdown_requested:
            break

        elapsed = time.time() - cycle_start
        wait = max(0, cycle_seconds - elapsed)

        if wait > 0:
            next_run = datetime.now(timezone.utc).timestamp() + wait
            next_run_str = datetime.fromtimestamp(next_run).strftime("%H:%M:%S")
            logger.info("⏳ Sonraki döngü: %s (%.0f dk sonra)", next_run_str, wait / 60)
            _sleep_interruptible(wait)

    logger.info("")
    logger.info(SEPARATOR)
    logger.info("  🛑 Trading sistemi kapatıldı — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("  Toplam döngü: %d", cycle_count)
    logger.info(SEPARATOR)


def _sleep_interruptible(seconds: float) -> None:
    """Kapatma sinyali gelene kadar bekle (1s parçalarla)."""
    end = time.time() + seconds
    while time.time() < end and not _shutdown_requested:
        time.sleep(min(1.0, end - time.time()))


if __name__ == "__main__":
    if "--once" in sys.argv:
        run_cycle()
    elif "--monitor" in sys.argv:
        run_monitor()
    else:
        run_forever()
