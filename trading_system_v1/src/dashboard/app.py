"""
Trading Platform v4 — Canlı Kar/Zarar Takip Paneli
====================================================
Her pozisyon için Binance/Yahoo'dan anlık fiyat çeker,
gerçek zamanlı kar/zarar hesaplar ve görselleştirir.

Çalıştırmak için:  streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# Auto-refresh: 60 saniyede bir sayfa otomatik yenilenir
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, limit=None, key="live_refresh")
except ImportError:
    pass  # fallback: manuel yenile butonu

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database.db import (
    get_all_trades,
    get_daily_snapshots,
    get_open_positions,
    get_portfolio,
    get_recent_signals,
    init_db,
)

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(page_title="Trading Platform", page_icon="⚡", layout="wide")

# ─── CSS ─────────────────────────────────────────────────────────
st.markdown(
    """
<style>
:root {
    --bg-dark: #0a0e1a;
    --bg-card: #111827;
    --bg-card-hover: #1a2237;
    --border: #1e293b;
    --green: #10b981;
    --red: #ef4444;
    --yellow: #f59e0b;
    --purple: #8b5cf6;
    --text: #e2e8f0;
    --text-dim: #94a3b8;
}
.main .block-container { padding: 0.5rem 1rem; max-width: 1500px; }
div[data-testid="stSidebar"] { background: var(--bg-dark); }

/* Üst özet kartları */
.top-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.top-card .label {
    font-size: 0.72rem; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 1px; margin: 0;
}
.top-card .value { font-size: 1.6rem; font-weight: 800; margin: 4px 0 0; }
.top-card .sub   { font-size: 0.82rem; margin: 2px 0 0; }
.g { color: var(--green); }
.r { color: var(--red); }
.w { color: var(--text); }
.d { color: var(--text-dim); }
.y { color: var(--yellow); }
.p { color: var(--purple); }

/* Pozisyon satır kartları */
.pos-row {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}
.pos-row:hover { background: var(--bg-card-hover); }
.pos-symbol { font-size: 1.1rem; font-weight: 700; min-width: 120px; }
.pos-detail { font-size: 0.82rem; color: var(--text-dim); }
.pos-pnl    { font-size: 1.15rem; font-weight: 700; text-align: right; min-width: 140px; }
.pos-badge  {
    display: inline-block; padding: 2px 10px; border-radius: 5px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.5px;
}
.badge-long  { background: rgba(16,185,129,0.15); color: var(--green); }
.badge-short { background: rgba(239,68,68,0.15); color: var(--red); }

/* İşlem geçmişi satır kartları */
.trade-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
}
.trade-card:hover { background: var(--bg-card-hover); }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    background: transparent; color: var(--text-dim);
    padding: 10px 18px; font-weight: 500; border-radius: 8px 8px 0 0;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--bg-card); color: white;
    border: 1px solid var(--border); border-bottom: none;
}
</style>
""",
    unsafe_allow_html=True,
)

init_db()

# ─── Anlık Fiyat Çekme ──────────────────────────────────────────
BINANCE_PRICE_URLS = [
    "https://api.binance.com/api/v3/ticker/price",
    "https://data-api.binance.vision/api/v3/ticker/price",
    "https://api1.binance.com/api/v3/ticker/price",
]


@st.cache_data(ttl=30)
def fetch_live_prices(symbols: tuple[str, ...]) -> dict[str, float]:
    """Binance ve Yahoo Finance'den anlık fiyat çek (30 sn cache)."""
    prices: dict[str, float] = {}

    # Binance — kripto semboller (fallback destekli)
    crypto_syms = [s for s in symbols if not s.endswith(".IS")]
    if crypto_syms:
        for price_url in BINANCE_PRICE_URLS:
            try:
                resp = requests.get(price_url, timeout=5)
                if resp.status_code == 451:
                    continue
                resp.raise_for_status()
                all_prices = {item["symbol"]: float(item["price"]) for item in resp.json()}
                for s in crypto_syms:
                    if s in all_prices:
                        prices[s] = all_prices[s]
                break
            except Exception:
                continue

    # Yahoo Finance — BIST semboller
    bist_syms = [s for s in symbols if s.endswith(".IS")]
    for s in bist_syms:
        try:
            tick = yf.Ticker(s)
            prices[s] = float(tick.fast_info["lastPrice"])
        except Exception:
            pass

    return prices


def _pnl_color(val: float) -> str:
    return "g" if val > 0 else ("r" if val < 0 else "w")


def _pnl_arrow(val: float) -> str:
    return "▲" if val > 0 else ("▼" if val < 0 else "●")


def _compute_position_details(positions: list[dict], live_prices: dict[str, float]):
    """Her pozisyon için giriş/anlık fiyat, P&L hesapla."""
    details = []
    long_value = 0.0
    short_unrealized = 0.0

    for p in positions:
        sym = p["symbol"]
        entry = p["avg_price"]
        qty = p["quantity"]
        direction = p.get("direction", "long")
        current = live_prices.get(sym, entry)

        if direction == "short":
            pnl_val = (entry - current) * qty
            pnl_pct = ((entry - current) / entry * 100) if entry else 0
            short_unrealized += pnl_val
        else:
            pnl_val = (current - entry) * qty
            pnl_pct = ((current - entry) / entry * 100) if entry else 0
            long_value += current * qty

        details.append({
            "symbol": sym,
            "direction": direction,
            "entry": entry,
            "current": current,
            "qty": qty,
            "pnl_val": pnl_val,
            "pnl_pct": pnl_pct,
            "stop_loss": p.get("stop_loss") or 0,
            "take_profit": p.get("take_profit") or 0,
            "entry_time": p.get("entry_time", ""),
            "market": p.get("market", ""),
        })

    return details, long_value, short_unrealized


# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Trading Platform")
    st.caption("Agresif Momentum · Paper Trading")
    st.divider()

    portfolio = get_portfolio()
    snapshots = get_daily_snapshots()
    if portfolio:
        initial = portfolio["initial_capital"]
        if snapshots:
            eq = snapshots[-1]["total_equity"]
            pnl = snapshots[-1]["total_pnl"]
        else:
            eq = portfolio["cash"]
            pnl = 0.0
        pnl_pct = (pnl / initial * 100) if initial else 0

        st.metric("Sermaye", f"₺{initial:,.2f}")
        st.metric(
            "Toplam Varlık",
            f"₺{eq:,.2f}",
            delta=f"{pnl:+,.2f} TL ({pnl_pct:+.1f}%)",
            delta_color="normal" if pnl >= 0 else "inverse",
        )
        st.metric("Nakit", f"₺{portfolio['cash']:,.2f}")
    else:
        st.info("Veri yok — `python src/main.py` çalıştırın.")

    st.divider()
    st.caption(f"🕐 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    if st.button("🔄 Yenile", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─── Tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Anlık Durum", "💼 Pozisyonlar", "📜 İşlem Geçmişi", "📈 Performans"]
)

# ══════════════════════════════════════════════════════════════════
# TAB 1 — ANLIK DURUM
# ══════════════════════════════════════════════════════════════════
with tab1:
    portfolio = get_portfolio()
    positions = get_open_positions()
    snapshots = get_daily_snapshots()

    if not portfolio:
        st.info("Henüz veri yok. Terminalde taramayı çalıştırın.")
    else:
        initial = portfolio["initial_capital"]
        cash = portfolio["cash"]

        # Canlı fiyatları çek
        pos_symbols = tuple(p["symbol"] for p in positions)
        live_prices = fetch_live_prices(pos_symbols) if pos_symbols else {}

        details, long_value, short_unrealized = _compute_position_details(
            positions, live_prices
        )

        total_equity = cash + long_value + short_unrealized
        total_pnl = total_equity - initial
        total_pnl_pct = (total_pnl / initial * 100) if initial else 0

        # ─── Üst Kartlar ───
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(
                f"""<div class="top-card">
                <p class="label">Toplam Varlık</p>
                <p class="value w">₺{total_equity:,.2f}</p>
            </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            cls = _pnl_color(total_pnl)
            st.markdown(
                f"""<div class="top-card">
                <p class="label">Kâr / Zarar</p>
                <p class="value {cls}">{_pnl_arrow(total_pnl)} ₺{total_pnl:+,.2f}</p>
                <p class="sub {cls}">%{total_pnl_pct:+.2f}</p>
            </div>""",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""<div class="top-card">
                <p class="label">Nakit</p>
                <p class="value w">₺{cash:,.2f}</p>
            </div>""",
                unsafe_allow_html=True,
            )
        with c4:
            longs = sum(1 for d in details if d["direction"] == "long")
            shorts = sum(1 for d in details if d["direction"] == "short")
            st.markdown(
                f"""<div class="top-card">
                <p class="label">Açık Pozisyon</p>
                <p class="value p">{len(positions)}</p>
                <p class="sub d">🟢 {longs} Long · 🔴 {shorts} Short</p>
            </div>""",
                unsafe_allow_html=True,
            )
        with c5:
            tt = portfolio["total_trades"]
            wt = portfolio["winning_trades"]
            wr = (wt / tt * 100) if tt else 0
            st.markdown(
                f"""<div class="top-card">
                <p class="label">İşlem / Win Rate</p>
                <p class="value y">{tt} işlem</p>
                <p class="sub {"g" if wr >= 50 else "r"}">%{wr:.0f} başarı</p>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ─── P&L Grafiği ───
        if snapshots:
            snap_df = pd.DataFrame(snapshots)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=snap_df["date"],
                    y=snap_df["total_pnl"],
                    mode="lines+markers+text",
                    name="Kâr/Zarar",
                    line=dict(color="#8b5cf6", width=3),
                    marker=dict(
                        size=8,
                        color=[
                            "#10b981" if v >= 0 else "#ef4444"
                            for v in snap_df["total_pnl"]
                        ],
                    ),
                    text=[f"₺{v:+,.1f}" for v in snap_df["total_pnl"]],
                    textposition="top center",
                    textfont=dict(size=11),
                    fill="tozeroy",
                    fillcolor="rgba(139,92,246,0.06)",
                )
            )
            fig.add_hline(y=0, line_dash="dot", line_color="#475569", line_width=1)
            fig.update_layout(
                title=dict(
                    text="📊 Kâr / Zarar Grafiği",
                    font=dict(size=15, color="#e2e8f0"),
                ),
                template="plotly_dark",
                height=320,
                margin=dict(l=10, r=10, t=45, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.5)",
                yaxis=dict(
                    title="TL",
                    gridcolor="#1e293b",
                    zeroline=True,
                    zerolinecolor="#475569",
                ),
                xaxis=dict(gridcolor="#1e293b"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ─── Pozisyon Listesi — Anlık K/Z ───
        if details:
            st.markdown("### 💼 Açık Pozisyonlar — Anlık Kâr/Zarar")

            for d in sorted(details, key=lambda x: abs(x["pnl_val"]), reverse=True):
                cls = _pnl_color(d["pnl_val"])
                arrow = _pnl_arrow(d["pnl_val"])
                badge = "badge-long" if d["direction"] == "long" else "badge-short"
                dir_text = "LONG" if d["direction"] == "long" else "SHORT"

                sl_txt = f"SL: {d['stop_loss']:.4f}" if d["stop_loss"] else "SL: —"
                tp_txt = f"TP: {d['take_profit']:.4f}" if d["take_profit"] else "TP: —"

                st.markdown(
                    f"""
                <div class="pos-row">
                    <div>
                        <span class="pos-badge {badge}">{dir_text}</span>
                        <span class="pos-symbol w" style="margin-left:8px;">{d['symbol']}</span>
                    </div>
                    <div style="text-align:center;">
                        <span class="pos-detail">Giriş: <b class="w">{d['entry']:.4f}</b></span>
                        <span class="pos-detail" style="margin-left:16px;">Anlık: <b class="y">{d['current']:.4f}</b></span>
                        <span class="pos-detail" style="margin-left:16px;">Miktar: <b class="w">{d['qty']:.4f}</b></span>
                    </div>
                    <div style="text-align:center;">
                        <span class="pos-detail">{sl_txt}</span>
                        <span class="pos-detail" style="margin-left:12px;">{tp_txt}</span>
                    </div>
                    <div class="pos-pnl {cls}">
                        {arrow} ₺{d['pnl_val']:+,.2f}<br>
                        <span style="font-size:0.8rem;">%{d['pnl_pct']:+.2f}</span>
                    </div>
                </div>""",
                    unsafe_allow_html=True,
                )

            # Pozisyon bazlı P&L bar chart
            pnl_df = pd.DataFrame(details).sort_values("pnl_val")
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    y=pnl_df["symbol"],
                    x=pnl_df["pnl_val"],
                    orientation="h",
                    marker_color=[
                        "#10b981" if v >= 0 else "#ef4444" for v in pnl_df["pnl_val"]
                    ],
                    text=[f"₺{v:+,.2f}" for v in pnl_df["pnl_val"]],
                    textposition="outside",
                    textfont=dict(size=12),
                )
            )
            fig.update_layout(
                title=dict(
                    text="Pozisyon Bazlı Kâr/Zarar",
                    font=dict(size=14, color="#e2e8f0"),
                ),
                template="plotly_dark",
                height=max(200, len(details) * 55 + 60),
                margin=dict(l=10, r=80, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.5)",
                xaxis=dict(
                    title="TL",
                    gridcolor="#1e293b",
                    zeroline=True,
                    zerolinecolor="#475569",
                ),
                yaxis=dict(gridcolor="#1e293b"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Açık pozisyon bulunmuyor.")


# ══════════════════════════════════════════════════════════════════
# TAB 2 — POZİSYONLAR (Detaylı Tablo + Grafikler)
# ══════════════════════════════════════════════════════════════════
with tab2:
    positions = get_open_positions()
    if not positions:
        st.info("Açık pozisyon yok.")
    else:
        pos_symbols = tuple(p["symbol"] for p in positions)
        live_prices = fetch_live_prices(pos_symbols)

        rows = []
        for p in positions:
            sym = p["symbol"]
            entry = p["avg_price"]
            qty = p["quantity"]
            direction = p.get("direction", "long")
            current = live_prices.get(sym, entry)
            notional = qty * entry

            if direction == "short":
                pnl = (entry - current) * qty
                pnl_pct = ((entry - current) / entry * 100) if entry else 0
            else:
                pnl = (current - entry) * qty
                pnl_pct = ((current - entry) / entry * 100) if entry else 0

            rows.append(
                {
                    "Yön": f"{'🟢 LONG' if direction == 'long' else '🔴 SHORT'}",
                    "Sembol": sym,
                    "Piyasa": p["market"].upper(),
                    "Miktar": qty,
                    "Giriş Fiyatı": entry,
                    "Anlık Fiyat": current,
                    "Tutar (₺)": notional,
                    "K/Z (₺)": pnl,
                    "K/Z (%)": pnl_pct,
                    "Stop-Loss": p.get("stop_loss") or 0,
                    "Take-Profit": p.get("take_profit") or 0,
                    "Giriş Zamanı": (p.get("entry_time") or "")[:19],
                }
            )

        df = pd.DataFrame(rows)

        st.markdown("### 💼 Açık Pozisyonlar — Detay Tablosu")
        st.dataframe(
            df.style.format(
                {
                    "Miktar": "{:.4f}",
                    "Giriş Fiyatı": "{:.4f}",
                    "Anlık Fiyat": "{:.4f}",
                    "Tutar (₺)": "₺{:,.2f}",
                    "K/Z (₺)": "₺{:+,.2f}",
                    "K/Z (%)": "%{:+.2f}",
                    "Stop-Loss": "{:.4f}",
                    "Take-Profit": "{:.4f}",
                }
            ).map(
                lambda v: (
                    "color: #10b981"
                    if isinstance(v, (int, float)) and v > 0
                    else (
                        "color: #ef4444"
                        if isinstance(v, (int, float)) and v < 0
                        else ""
                    )
                ),
                subset=["K/Z (₺)", "K/Z (%)"],
            ),
            use_container_width=True,
            hide_index=True,
            height=min(400, len(rows) * 45 + 60),
        )

        # Dağılım grafikleri
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(
                go.Pie(
                    labels=df["Sembol"],
                    values=df["Tutar (₺)"].abs(),
                    hole=0.45,
                    marker=dict(
                        colors=[
                            "#6366f1",
                            "#8b5cf6",
                            "#a78bfa",
                            "#c4b5fd",
                            "#818cf8",
                        ]
                    ),
                    textinfo="label+percent",
                )
            )
            fig.update_layout(
                title="Pozisyon Dağılımı",
                template="plotly_dark",
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            long_c = sum(1 for r in rows if "LONG" in r["Yön"])
            short_c = sum(1 for r in rows if "SHORT" in r["Yön"])
            fig = go.Figure(
                go.Pie(
                    labels=["LONG", "SHORT"],
                    values=[long_c, short_c],
                    hole=0.5,
                    marker=dict(colors=["#10b981", "#ef4444"]),
                    textinfo="label+value",
                )
            )
            fig.update_layout(
                title="Long / Short Dağılımı",
                template="plotly_dark",
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — İŞLEM GEÇMİŞİ
# ══════════════════════════════════════════════════════════════════
with tab3:
    trades = get_all_trades()
    if not trades:
        st.info("Henüz işlem yok.")
    else:
        st.markdown("### 📜 İşlem Geçmişi")

        trades_df = pd.DataFrame(trades)

        for _, t in trades_df.iterrows():
            side = t["side"]
            if side == "buy":
                icon, color, label = "🟢", "g", "ALIŞ (LONG)"
            elif side == "short":
                icon, color, label = "🔴", "r", "SATIŞ (SHORT)"
            elif "cover" in str(side):
                icon, color, label = "🟡", "y", "KAPANIŞ"
            else:
                icon, color, label = "🔻", "r", "SATIŞ"

            time_str = (t["created_at"] or "")[:19].replace("T", " ")

            st.markdown(
                f"""
            <div class="trade-card">
                <div style="min-width:140px;">
                    <span class="{color}" style="font-weight:700;">{icon} {label}</span>
                </div>
                <div style="min-width:100px;">
                    <span class="w" style="font-weight:700;">{t['symbol']}</span>
                    <span class="d" style="font-size:0.75rem; margin-left:4px;">
                        {t['market'].upper()}
                    </span>
                </div>
                <div style="text-align:center;">
                    <span class="d">Fiyat:</span>
                    <span class="w"><b>{t['price']:.4f}</b></span>
                    <span class="d" style="margin-left:12px;">Miktar:</span>
                    <span class="w"><b>{t['quantity']:.4f}</b></span>
                </div>
                <div style="text-align:center;">
                    <span class="d">Tutar:</span>
                    <span class="y"><b>₺{t['notional']:,.2f}</b></span>
                    <span class="d" style="margin-left:12px;">Komisyon:</span>
                    <span class="r">{t['fee']:.4f}</span>
                </div>
                <div style="text-align:right; min-width:120px;">
                    <span class="d" style="font-size:0.78rem;">{time_str}</span>
                </div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("")
        c1, c2, c3 = st.columns(3)
        with c1:
            total_buy = trades_df[trades_df["side"] == "buy"]["notional"].sum()
            st.metric("Toplam Alış Tutarı", f"₺{total_buy:,.2f}")
        with c2:
            total_short = trades_df[trades_df["side"] == "short"]["notional"].sum()
            st.metric("Toplam Short Tutarı", f"₺{total_short:,.2f}")
        with c3:
            total_fee = trades_df["fee"].sum()
            st.metric("Toplam Komisyon", f"₺{total_fee:,.4f}")


# ══════════════════════════════════════════════════════════════════
# TAB 4 — PERFORMANS
# ══════════════════════════════════════════════════════════════════
with tab4:
    snapshots = get_daily_snapshots()
    portfolio = get_portfolio()

    if not snapshots or not portfolio:
        st.info("Performans verisi yok.")
    else:
        snap_df = pd.DataFrame(snapshots)
        initial = portfolio["initial_capital"]

        st.markdown("### 📈 Performans Analizi")

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=snap_df["date"],
                    y=snap_df["total_equity"],
                    mode="lines+markers",
                    line=dict(color="#8b5cf6", width=3),
                    marker=dict(size=6, color="#a78bfa"),
                    fill="tozeroy",
                    fillcolor="rgba(139,92,246,0.06)",
                    name="Varlık",
                )
            )
            fig.add_hline(
                y=initial,
                line_dash="dot",
                line_color="#f59e0b",
                annotation_text=f"Sermaye: ₺{initial:,.0f}",
                annotation_font_color="#f59e0b",
            )
            fig.update_layout(
                title="Sermaye Eğrisi",
                template="plotly_dark",
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.5)",
                yaxis=dict(title="TL", gridcolor="#1e293b"),
                xaxis=dict(gridcolor="#1e293b"),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=snap_df["date"],
                    y=snap_df["total_pnl"],
                    marker_color=[
                        "#10b981" if v >= 0 else "#ef4444"
                        for v in snap_df["total_pnl"]
                    ],
                    text=[f"₺{v:+,.1f}" for v in snap_df["total_pnl"]],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Kümülatif P&L",
                template="plotly_dark",
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.5)",
                yaxis=dict(title="TL", gridcolor="#1e293b"),
                xaxis=dict(gridcolor="#1e293b"),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # İstatistikler
        tt = portfolio["total_trades"]
        wt = portfolio["winning_trades"]
        lt = portfolio["losing_trades"]
        wr = (wt / tt * 100) if tt else 0
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Toplam İşlem", str(tt))
        with c2:
            st.metric("Kazanan", str(wt), delta=f"%{wr:.0f}" if tt else None)
        with c3:
            st.metric("Kaybeden", str(lt))
        with c4:
            net_pnl = snapshots[-1]["total_pnl"] if snapshots else 0
            st.metric(
                "Net P&L",
                f"₺{net_pnl:+,.2f}",
                delta_color="normal" if net_pnl >= 0 else "inverse",
            )


# ─── Footer ──────────────────────────────────────────────────────

# ─── Canlı İşlem Akışı (Sidebar Alt) ────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("### 🔔 Son İşlemler")
    recent_trades = get_all_trades()[:5]
    if recent_trades:
        for t in recent_trades:
            side = t["side"]
            if side == "buy":
                icon = "🟢"
            elif side == "short":
                icon = "🔴"
            else:
                icon = "🟡"
            time_str = (t["created_at"] or "")[:16].replace("T", " ")
            st.caption(f"{icon} **{t['symbol']}** · ₺{t['notional']:,.0f} · {time_str}")
    else:
        st.caption("Henüz işlem yok")

    st.markdown("### 📡 Son Sinyaller")
    recent_sigs = get_recent_signals(limit=5)
    if recent_sigs:
        for s in recent_sigs:
            sig_type = s.get("signal_type", "flat")
            if sig_type == "long":
                icon = "🟢"
            elif sig_type == "short":
                icon = "🔴"
            else:
                icon = "⚪"
            score = s.get("composite_score", 0) or 0
            conf = s.get("confidence", 0) or 0
            st.caption(f"{icon} **{s['symbol']}** · Skor: {score:.0f} · Güven: {conf:.0f}%")
    else:
        st.caption("Henüz sinyal yok")

st.divider()
st.caption(
    f"⚡ Trading Platform v4 · Anlık Fiyat Takibi · "
    f"Son güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')} · "
    f"Bu platform yatırım tavsiyesi değildir."
)
