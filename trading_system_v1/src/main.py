from __future__ import annotations

import json
import signal
import sys
import time
import traceback
from datetime import datetime, timezone

from core.config import AppConfig
from core.models import MarketType, OrderRequest, Side, SignalType
from data.binance_spot import BinanceSpotAdapter
from data.yahoo_bist import YahooBISTAdapter
from database.db import (
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


def run_cycle() -> None:
    config = AppConfig()
    strategy = AggressiveMomentumStrategy()
    risk = RiskManager(config.risk)

    # --- DB init ---
    init_db()
    init_portfolio(config.risk.capital_tl)
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

    # Check existing positions for stop-loss / take-profit
    for sym, pos in list(broker.positions.items()):
        db_pos = next((p for p in db_positions if p["symbol"] == sym), None)
        if not db_pos:
            continue
        # Find current price from scan results
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

        # Stop-loss hit
        if sl and current_price <= sl:
            loss = (current_price - pos.avg_price) * pos.quantity
            logger.info("🛑 STOP-LOSS: %s | Giriş: %.4f | Çıkış: %.4f | Zarar: %.2f TL",
                        sym, pos.avg_price, current_price, loss)
            order = OrderRequest(symbol=sym, market=pos.market, side=Side.SELL, quantity=pos.quantity)
            fill = broker.submit_market_order(order=order, mark_price=current_price)
            record_trade(sym, pos.market.value, "sell", fill.quantity, fill.price, fill.fee,
                         strategy=strategy.name, metadata=json.dumps({"reason": "stop_loss"}))
            increment_trade_stats(won=loss >= 0, pnl=loss)
            upsert_position(sym, pos.market.value, 0, 0)
            continue

        # Take-profit hit
        if tp and current_price >= tp:
            profit = (current_price - pos.avg_price) * pos.quantity
            logger.info("🎯 TAKE-PROFIT: %s | Giriş: %.4f | Çıkış: %.4f | Kâr: %.2f TL",
                        sym, pos.avg_price, current_price, profit)
            order = OrderRequest(symbol=sym, market=pos.market, side=Side.SELL, quantity=pos.quantity)
            fill = broker.submit_market_order(order=order, mark_price=current_price)
            record_trade(sym, pos.market.value, "sell", fill.quantity, fill.price, fill.fee,
                         strategy=strategy.name, metadata=json.dumps({"reason": "take_profit"}))
            increment_trade_stats(won=True, pnl=profit)
            upsert_position(sym, pos.market.value, 0, 0)
            continue

        logger.info("📊 AÇIK POZİSYON: %s | Giriş: %.4f | Şimdi: %.4f | P&L: %+.2f%%",
                    sym, pos.avg_price, current_price, pnl_pct * 100)

    # Check SHORT positions for stop-loss / take-profit
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

        # SHORT stop-loss: price went UP beyond stop
        if sl and current_price >= sl:
            loss = (entry - current_price) * qty
            logger.info("🛑 SHORT STOP-LOSS: %s | Giriş: %.4f | Çıkış: %.4f | Zarar: %.2f TL",
                        sym, entry, current_price, loss)
            order = OrderRequest(symbol=sym, market=short_data["market"], side=Side.BUY, quantity=qty)
            fill = broker.submit_market_order(order=order, mark_price=current_price)
            record_trade(sym, short_data["market"].value, "buy_to_cover", fill.quantity, fill.price, fill.fee,
                         strategy=strategy.name, metadata=json.dumps({"reason": "short_stop_loss"}))
            increment_trade_stats(won=loss >= 0, pnl=loss)
            upsert_position(sym, short_data["market"].value, 0, 0)
            continue

        # SHORT take-profit: price went DOWN to target
        if tp and current_price <= tp:
            profit = (entry - current_price) * qty
            logger.info("🎯 SHORT TAKE-PROFIT: %s | Giriş: %.4f | Çıkış: %.4f | Kâr: %.2f TL",
                        sym, entry, current_price, profit)
            order = OrderRequest(symbol=sym, market=short_data["market"], side=Side.BUY, quantity=qty)
            fill = broker.submit_market_order(order=order, mark_price=current_price)
            record_trade(sym, short_data["market"].value, "buy_to_cover", fill.quantity, fill.price, fill.fee,
                         strategy=strategy.name, metadata=json.dumps({"reason": "short_take_profit"}))
            increment_trade_stats(won=True, pnl=profit)
            upsert_position(sym, short_data["market"].value, 0, 0)
            continue

        logger.info("📊 AÇIK SHORT: %s | Giriş: %.4f | Şimdi: %.4f | P&L: %+.2f%%",
                    sym, entry, current_price, pnl_pct * 100)

    # New trade signals (LONG + SHORT)
    top_candidates = []
    for signal, df in all_results:
        if signal.signal in (SignalType.LONG, SignalType.SHORT):
            if signal.symbol not in broker.positions and signal.symbol not in broker.short_positions:
                top_candidates.append((signal, df))

    top_candidates = sorted(top_candidates, key=lambda item: abs(item[0].score), reverse=True)

    for signal, df in top_candidates[:3]:  # max 3 new trades per cycle
        total_open = len(broker.positions) + len(broker.short_positions)
        decision = risk.evaluate(signal, current_open_positions=total_open)
        last_close = float(df.iloc[-1]["close"])

        if not decision.allowed:
            logger.info("❌ RED: %s — %s", signal.symbol, decision.reason)
            continue

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
        )

    # --- Portfolio summary ---
    update_portfolio_cash(broker.cash)

    positions_value = 0.0
    # SHORT positions value
    short_unrealized = 0.0
    for sym, short_data in broker.short_positions.items():
        for signal, df in all_results:
            if signal.symbol == sym:
                current_price = float(df.iloc[-1]["close"])
                short_unrealized += (short_data["entry_price"] - current_price) * short_data["quantity"]
                break

    for sym, pos in broker.positions.items():
        for signal, df in all_results:
            if signal.symbol == sym:
                current_price = float(df.iloc[-1]["close"])
                positions_value += pos.quantity * current_price
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
    else:
        run_forever()
