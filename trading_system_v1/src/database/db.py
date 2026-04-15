from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "trades.db"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT NOT NULL,
            market        TEXT NOT NULL,
            side          TEXT NOT NULL,
            quantity      REAL NOT NULL,
            price         REAL NOT NULL,
            fee           REAL NOT NULL DEFAULT 0,
            notional      REAL NOT NULL,
            strategy      TEXT,
            signal_score  REAL,
            stop_loss_pct REAL,
            take_profit_pct REAL,
            metadata      TEXT,
            created_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS positions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT NOT NULL UNIQUE,
            market        TEXT NOT NULL,
            quantity      REAL NOT NULL DEFAULT 0,
            avg_price     REAL NOT NULL DEFAULT 0,
            entry_time    TEXT,
            stop_loss     REAL,
            take_profit   REAL,
            realized_pnl  REAL NOT NULL DEFAULT 0,
            updated_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS portfolio (
            id            INTEGER PRIMARY KEY CHECK (id = 1),
            initial_capital REAL NOT NULL,
            cash          REAL NOT NULL,
            total_trades  INTEGER NOT NULL DEFAULT 0,
            winning_trades INTEGER NOT NULL DEFAULT 0,
            losing_trades INTEGER NOT NULL DEFAULT 0,
            total_pnl     REAL NOT NULL DEFAULT 0,
            updated_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS signals_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT NOT NULL,
            market        TEXT NOT NULL,
            signal_type   TEXT NOT NULL,
            score         REAL NOT NULL,
            strategy      TEXT,
            composite_score REAL,
            confidence    REAL,
            details       TEXT,
            created_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS daily_snapshots (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            date          TEXT NOT NULL UNIQUE,
            cash          REAL NOT NULL,
            positions_value REAL NOT NULL DEFAULT 0,
            total_equity  REAL NOT NULL,
            daily_pnl     REAL NOT NULL DEFAULT 0,
            total_pnl     REAL NOT NULL DEFAULT 0,
            created_at    TEXT NOT NULL
        );
        """)
        # Migration: add direction column to positions (SHORT/LONG tracking)
        try:
            con.execute("ALTER TABLE positions ADD COLUMN direction TEXT NOT NULL DEFAULT 'long'")
        except Exception:
            pass  # column already exists


# ---------- Portfolio ----------

def init_portfolio(capital: float) -> None:
    with _conn() as con:
        row = con.execute("SELECT id FROM portfolio WHERE id=1").fetchone()
        if row is None:
            con.execute(
                "INSERT INTO portfolio (id, initial_capital, cash, updated_at) VALUES (1, ?, ?, ?)",
                (capital, capital, _now_iso()),
            )


def get_portfolio() -> Optional[Dict[str, Any]]:
    with _conn() as con:
        row = con.execute("SELECT * FROM portfolio WHERE id=1").fetchone()
        return dict(row) if row else None


def update_portfolio_cash(cash: float) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE portfolio SET cash=?, updated_at=? WHERE id=1",
            (cash, _now_iso()),
        )


def increment_trade_stats(won: bool, pnl: float) -> None:
    with _conn() as con:
        if won:
            con.execute(
                "UPDATE portfolio SET total_trades=total_trades+1, winning_trades=winning_trades+1, total_pnl=total_pnl+?, updated_at=? WHERE id=1",
                (pnl, _now_iso()),
            )
        else:
            con.execute(
                "UPDATE portfolio SET total_trades=total_trades+1, losing_trades=losing_trades+1, total_pnl=total_pnl+?, updated_at=? WHERE id=1",
                (pnl, _now_iso()),
            )


# ---------- Trades ----------

def record_trade(
    symbol: str,
    market: str,
    side: str,
    quantity: float,
    price: float,
    fee: float,
    strategy: str = "",
    signal_score: float = 0.0,
    stop_loss_pct: float = 0.0,
    take_profit_pct: float = 0.0,
    metadata: str = "",
) -> int:
    notional = quantity * price
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO trades
               (symbol, market, side, quantity, price, fee, notional, strategy,
                signal_score, stop_loss_pct, take_profit_pct, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, market, side, quantity, price, fee, notional, strategy,
             signal_score, stop_loss_pct, take_profit_pct, metadata, _now_iso()),
        )
        return cur.lastrowid  # type: ignore[return-value]


def get_all_trades() -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM trades ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_trades_for_symbol(symbol: str) -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM trades WHERE symbol=? ORDER BY created_at DESC", (symbol,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------- Positions ----------

def upsert_position(
    symbol: str,
    market: str,
    quantity: float,
    avg_price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    realized_pnl: float = 0.0,
    direction: str = "long",
) -> None:
    now = _now_iso()
    with _conn() as con:
        existing = con.execute("SELECT id FROM positions WHERE symbol=?", (symbol,)).fetchone()
        if existing:
            if quantity <= 0:
                con.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
            else:
                con.execute(
                    """UPDATE positions SET quantity=?, avg_price=?, stop_loss=?,
                       take_profit=?, realized_pnl=realized_pnl+?, direction=?, updated_at=? WHERE symbol=?""",
                    (quantity, avg_price, stop_loss, take_profit, realized_pnl, direction, now, symbol),
                )
        else:
            if quantity > 0:
                con.execute(
                    """INSERT INTO positions
                       (symbol, market, quantity, avg_price, entry_time, stop_loss, take_profit, realized_pnl, direction, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (symbol, market, quantity, avg_price, now, stop_loss, take_profit, realized_pnl, direction, now),
                )


def get_open_positions() -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM positions WHERE quantity > 0 ORDER BY symbol").fetchall()
        return [dict(r) for r in rows]


def get_position(symbol: str) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        row = con.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
        return dict(row) if row else None


# ---------- Signals Log ----------

def log_signal(
    symbol: str,
    market: str,
    signal_type: str,
    score: float,
    strategy: str = "",
    composite_score: float = 0.0,
    confidence: float = 0.0,
    details: str = "",
) -> None:
    with _conn() as con:
        con.execute(
            """INSERT INTO signals_log
               (symbol, market, signal_type, score, strategy, composite_score, confidence, details, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, market, signal_type, score, strategy, composite_score, confidence, details, _now_iso()),
        )


def get_recent_signals(limit: int = 50) -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM signals_log ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------- Daily Snapshots ----------

def save_daily_snapshot(
    date_str: str, cash: float, positions_value: float, total_equity: float,
    daily_pnl: float, total_pnl: float,
) -> None:
    with _conn() as con:
        con.execute(
            """INSERT OR REPLACE INTO daily_snapshots
               (date, cash, positions_value, total_equity, daily_pnl, total_pnl, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (date_str, cash, positions_value, total_equity, daily_pnl, total_pnl, _now_iso()),
        )


def get_daily_snapshots() -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM daily_snapshots ORDER BY date ASC").fetchall()
        return [dict(r) for r in rows]
