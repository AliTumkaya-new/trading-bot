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
import streamlit.components.v1 as components
import yfinance as yf

# Auto-refresh: 10 saniyede bir sayfa otomatik yenilenir
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=10_000, limit=None, key="live_refresh")
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

# ─── Merkezi Veri Çekimi (Tüm tablar aynı veriyi kullanır) ──────
@st.cache_data(ttl=10)
def _load_all_data():
    """Tüm dashboard verisini tek seferde çek, tutarlılık sağla."""
    portfolio = get_portfolio()
    positions = get_open_positions()
    snapshots = get_daily_snapshots()
    trades = get_all_trades()
    signals = get_recent_signals(limit=50)
    return portfolio, positions, snapshots, trades, signals

_portfolio, _positions, _snapshots, _all_trades, _all_signals = _load_all_data()

# Not: Canlı fiyat çekme ve pozisyon hesaplama fonksiyonlar tanımlandıktan sonra yapılacak


# ─── TradingView Widget Helpers ──────────────────────────────────
def _tv_symbol(symbol: str) -> str:
    """Dashboard sembolünü TradingView formatına çevir."""
    if symbol.endswith(".IS"):
        return f"BIST:{symbol.replace('.IS', '')}"
    return f"BINANCE:{symbol}"


def _tradingview_chart(symbol: str, height: int = 400) -> str:
    """TradingView Advanced Chart widget HTML'i üret."""
    tv_sym = _tv_symbol(symbol)
    return f"""
    <div class="tradingview-widget-container" style="height:{height}px;">
      <div id="tv_{symbol.replace('.','_')}" style="height:100%;"></div>
      <script type="text/javascript"
        src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "container_id": "tv_{symbol.replace('.','_')}",
          "autosize": true,
          "symbol": "{tv_sym}",
          "interval": "240",
          "timezone": "Europe/Istanbul",
          "theme": "dark",
          "style": "1",
          "locale": "tr",
          "toolbar_bg": "#0a0e1a",
          "enable_publishing": false,
          "hide_top_toolbar": false,
          "hide_legend": false,
          "save_image": false,
          "studies": ["RSI@tv-basicstudies","MACD@tv-basicstudies","BB@tv-basicstudies"],
          "show_popup_button": true,
          "popup_width": "1000",
          "popup_height": "650"
        }});
      </script>
    </div>"""


def _tradingview_ticker_tape() -> str:
    """TradingView haber bandı (ticker tape) widget HTML'i üret."""
    return """
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
        {
          "symbols": [
            {"proName": "BINANCE:BTCUSDT", "title": "BTC/USDT"},
            {"proName": "BINANCE:ETHUSDT", "title": "ETH/USDT"},
            {"proName": "BINANCE:SOLUSDT", "title": "SOL/USDT"},
            {"proName": "BINANCE:AVAXUSDT", "title": "AVAX/USDT"},
            {"proName": "BINANCE:FETUSDT", "title": "FET/USDT"},
            {"proName": "BINANCE:INJUSDT", "title": "INJ/USDT"},
            {"proName": "BINANCE:TIAUSDT", "title": "TIA/USDT"},
            {"proName": "BINANCE:APTUSDT", "title": "APT/USDT"},
            {"proName": "BINANCE:NEARUSDT", "title": "NEAR/USDT"},
            {"proName": "BINANCE:ARBUSDT", "title": "ARB/USDT"},
            {"proName": "BIST:ASELS", "title": "ASELS"},
            {"proName": "BIST:THYAO", "title": "THYAO"},
            {"proName": "BIST:TUPRS", "title": "TUPRS"},
            {"proName": "BIST:SISE", "title": "ŞİŞE"},
            {"proName": "BIST:KCHOL", "title": "KCHOL"}
          ],
          "showSymbolLogo": true,
          "colorTheme": "dark",
          "isTransparent": true,
          "displayMode": "adaptive",
          "locale": "tr"
        }
      </script>
    </div>"""
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
    """Her pozisyon için giriş/anlık fiyat, P&L hesapla. Kaldıraçlı pozisyonlarda margin bazlı."""
    details = []
    long_value = 0.0
    short_unrealized = 0.0

    for p in positions:
        sym = p["symbol"]
        entry = p["avg_price"]
        qty = p["quantity"]
        direction = p.get("direction", "long")
        lev = p.get("leverage") or 1
        current = live_prices.get(sym, entry)
        entry_notional = entry * qty
        margin = entry_notional / lev if lev > 1 else entry_notional

        if direction == "short":
            pnl_val = (entry - current) * qty
            pnl_pct = ((entry - current) / entry * 100) if entry else 0
            short_unrealized += margin + pnl_val
        else:
            pnl_val = (current - entry) * qty
            pnl_pct = ((current - entry) / entry * 100) if entry else 0
            long_value += margin + pnl_val

        details.append({
            "symbol": sym,
            "direction": direction,
            "entry": entry,
            "current": current,
            "qty": qty,
            "leverage": lev,
            "margin": margin,
            "pnl_val": pnl_val,
            "pnl_pct": pnl_pct,
            "stop_loss": p.get("stop_loss") or 0,
            "take_profit": p.get("take_profit") or 0,
            "entry_time": p.get("entry_time", ""),
            "market": p.get("market", ""),
        })

    return details, long_value, short_unrealized


# ─── Canlı Fiyat & Pozisyon Hesaplama (fonksiyonlar tanımlandıktan sonra) ──
_pos_symbols = tuple(p["symbol"] for p in _positions)
_live_prices = fetch_live_prices(_pos_symbols) if _pos_symbols else {}

_pos_details: list = []
_long_value: float = 0.0
_short_unrealized: float = 0.0

if _positions and _live_prices:
    _pos_details, _long_value, _short_unrealized = _compute_position_details(
        _positions, _live_prices
    )

_initial_capital = _portfolio["initial_capital"] if _portfolio else 0.0
_cash = _portfolio["cash"] if _portfolio else 0.0
_live_total_equity = _cash + _long_value + _short_unrealized
_live_total_pnl = _live_total_equity - _initial_capital if _initial_capital else 0.0
_live_pnl_pct = (_live_total_pnl / _initial_capital * 100) if _initial_capital else 0.0


# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Trading Platform")
    st.caption("Agresif Momentum · Paper Trading")
    st.divider()

    if _portfolio:
        st.metric("Sermaye", f"₺{_initial_capital:,.2f}")
        st.metric(
            "Toplam Varlık",
            f"₺{_live_total_equity:,.2f}",
            delta=f"{_live_total_pnl:+,.2f} TL ({_live_pnl_pct:+.1f}%)",
            delta_color="normal" if _live_total_pnl >= 0 else "inverse",
        )
        st.metric("Nakit", f"₺{_cash:,.2f}")
    else:
        st.info("Veri yok — `python src/main.py` çalıştırın.")

    st.divider()
    st.caption(f"🕐 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    if st.button("🔄 Yenile", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─── TradingView Ticker Tape (üst bant) ─────────────────────────
components.html(_tradingview_ticker_tape(), height=78, scrolling=False)

# ─── Tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Anlık Durum", "📺 Canlı Grafik", "💼 Pozisyonlar", "📜 İşlem Geçmişi", "📈 Performans", "🧠 ML Öğrenme"]
)

# ══════════════════════════════════════════════════════════════════
# TAB 1 — ANLIK DURUM
# ══════════════════════════════════════════════════════════════════
with tab1:
    if not _portfolio:
        st.info("Henüz veri yok. Terminalde taramayı çalıştırın.")
    else:
        initial = _initial_capital
        cash = _cash
        details = _pos_details
        total_equity = _live_total_equity
        total_pnl = _live_total_pnl
        total_pnl_pct = _live_pnl_pct

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
                <p class="value p">{len(_positions)}</p>
                <p class="sub d">🟢 {longs} Long · 🔴 {shorts} Short</p>
            </div>""",
                unsafe_allow_html=True,
            )
        with c5:
            tt = _portfolio["total_trades"]
            wt = _portfolio["winning_trades"]
            wr = (wt / tt * 100) if tt else 0
            wr_cls = "g" if wr >= 50 else "r"
            st.markdown(
                f"""<div class="top-card">
                <p class="label">İşlem / Win Rate</p>
                <p class="value y">{tt} işlem</p>
                <p class="sub {wr_cls}">%{wr:.0f} başarı</p>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ─── P&L Grafiği ───
        if _snapshots:
            snap_df = pd.DataFrame(_snapshots)
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
                lev_txt = f"{d['leverage']}x" if d.get("leverage", 1) > 1 else ""

                st.markdown(
                    f"""
                <div class="pos-row">
                    <div>
                        <span class="pos-badge {badge}">{dir_text}</span>
                        {f'<span class="pos-badge" style="background:rgba(139,92,246,0.15);color:#8b5cf6;margin-left:4px;">{lev_txt}</span>' if lev_txt else ''}
                        <span class="pos-symbol w" style="margin-left:8px;">{d['symbol']}</span>
                    </div>
                    <div style="text-align:center;">
                        <span class="pos-detail">Giriş: <b class="w">{d['entry']:.4f}</b></span>
                        <span class="pos-detail" style="margin-left:16px;">Anlık: <b class="y">{d['current']:.4f}</b></span>
                        <span class="pos-detail" style="margin-left:16px;">Margin: <b class="p">₺{d['margin']:,.2f}</b></span>
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
# TAB 2 — CANLI GRAFİK (TradingView)
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📺 TradingView Canlı Grafikler")

    if _positions:
        st.markdown("**Açık pozisyonlarınızın canlı grafikleri:**")

        for p in _positions:
            sym = p["symbol"]
            direction = p.get("direction", "long")
            badge = "🟢 LONG" if direction == "long" else "🔴 SHORT"
            entry = p["avg_price"]
            sl = p.get("stop_loss") or 0
            tp = p.get("take_profit") or 0

            st.markdown(f"#### {badge} — {sym}")
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.caption(f"Giriş: **{entry:.4f}**")
            with col_info2:
                st.caption(f"SL: **{sl:.4f}**" if sl else "SL: —")
            with col_info3:
                st.caption(f"TP: **{tp:.4f}**" if tp else "TP: —")

            components.html(_tradingview_chart(sym, height=450), height=470)
            st.divider()
    else:
        st.info("Açık pozisyon yok. Pozisyon açıldığında canlı grafikleri burada görebilirsiniz.")

    # Ek olarak popüler kripto/borsa grafiklerini göster
    st.markdown("### 🌐 Popüler Piyasalar")
    pop_col1, pop_col2 = st.columns(2)
    with pop_col1:
        st.markdown("**BTC/USDT**")
        components.html(_tradingview_chart("BTCUSDT", height=350), height=370)
    with pop_col2:
        st.markdown("**ETH/USDT**")
        components.html(_tradingview_chart("ETHUSDT", height=350), height=370)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — POZİSYONLAR (Detaylı Tablo + Grafikler)
# ══════════════════════════════════════════════════════════════════
with tab3:
    if not _positions:
        st.info("Açık pozisyon yok.")
    else:
        rows = []
        for p in _positions:
            sym = p["symbol"]
            entry = p["avg_price"]
            qty = p["quantity"]
            direction = p.get("direction", "long")
            lev = p.get("leverage") or 1
            current = _live_prices.get(sym, entry)
            notional = qty * entry
            margin = notional / lev if lev > 1 else notional

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
                    "Kaldıraç": f"{lev}x",
                    "Miktar": qty,
                    "Giriş Fiyatı": entry,
                    "Anlık Fiyat": current,
                    "Margin (₺)": margin,
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
                    "Margin (₺)": "₺{:,.2f}",
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
                    values=df["Margin (₺)"].abs(),
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
# TAB 4 — İŞLEM GEÇMİŞİ
# ══════════════════════════════════════════════════════════════════
with tab4:
    if not _all_trades:
        st.info("Henüz işlem yok.")
    else:
        st.markdown("### 📜 İşlem Geçmişi")

        trades_df = pd.DataFrame(_all_trades)

        for _, t in trades_df.iterrows():
            side = t["side"]
            if side == "buy":
                icon, color, label = "🟢", "g", "ALIŞ (LONG)"
            elif side == "short":
                icon, color, label = "🔴", "r", "AÇIĞA SATIŞ (SHORT)"
            elif "cover" in str(side):
                icon, color, label = "🟡", "y", "SHORT KAPANIŞ"
            elif side == "sell":
                # Sell reason: metadata'dan kontrol et
                meta_str = t.get("metadata") or ""
                if "stop_loss" in meta_str:
                    icon, color, label = "🛑", "r", "STOP-LOSS SATIŞ"
                elif "take_profit" in meta_str:
                    icon, color, label = "🎯", "g", "KÂR AL SATIŞ"
                else:
                    icon, color, label = "🔻", "r", "SATIŞ"
            else:
                icon, color, label = "⚪", "d", side.upper()

            time_str = (t["created_at"] or "")[:19].replace("T", " ")

            # SL/TP yüzdeleri
            sl_pct = t.get("stop_loss_pct") or 0
            tp_pct = t.get("take_profit_pct") or 0
            sl_tp_text = ""
            if sl_pct or tp_pct:
                sl_tp_text = (
                    f' · <span class="d">SL:</span> <span class="r"><b>%{sl_pct*100:.1f}</b></span>'
                    f' <span class="d">TP:</span> <span class="g"><b>%{tp_pct*100:.1f}</b></span>'
                )

            st.markdown(
                f"""<div class="trade-card">
                <div style="min-width:140px;">
                    <span class="{color}" style="font-weight:700;">{icon} {label}</span>
                    <span class="d" style="font-size:0.75rem; margin-left:6px;">{time_str}</span>
                </div>
                <div style="min-width:100px;">
                    <span class="w" style="font-weight:700;">{t['symbol']}</span>
                    <span class="d" style="font-size:0.75rem; margin-left:4px;">{t['market'].upper()}</span>
                </div>
                <div style="text-align:center;">
                    <span class="d">Fiyat:</span> <span class="w"><b>{t['price']:.4f}</b></span>
                    <span class="d" style="margin-left:8px;">Miktar:</span> <span class="w"><b>{t['quantity']:.4f}</b></span>
                    <span class="d" style="margin-left:8px;">Tutar:</span> <span class="y"><b>₺{t['notional']:,.2f}</b></span>{sl_tp_text}
                </div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            total_buy = trades_df[trades_df["side"] == "buy"]["notional"].sum()
            st.metric("Toplam Alış Tutarı", f"₺{total_buy:,.2f}")
        with c2:
            total_short = trades_df[trades_df["side"] == "short"]["notional"].sum()
            st.metric("Toplam Short Tutarı", f"₺{total_short:,.2f}")
        with c3:
            total_fee = trades_df["fee"].sum()
            st.metric("Toplam Komisyon", f"₺{total_fee:,.4f}")
        with c4:
            # Gerçekleşmiş P/L (portfolio'dan)
            realized = _portfolio["total_pnl"] if _portfolio else 0.0
            st.metric(
                "Gerçekleşmiş P&L",
                f"₺{realized:+,.2f}",
                delta_color="normal" if realized >= 0 else "inverse",
            )


# ══════════════════════════════════════════════════════════════════
# TAB 5 — PERFORMANS
# ══════════════════════════════════════════════════════════════════
with tab5:
    if not _snapshots or not _portfolio:
        st.info("Performans verisi yok.")
    else:
        snap_df = pd.DataFrame(_snapshots)
        initial = _initial_capital

        st.markdown("### 📈 Performans Analizi")

        # Canlı özet kartları (tüm tablarla tutarlı)
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        with pc1:
            st.metric("Sermaye", f"₺{initial:,.2f}")
        with pc2:
            st.metric(
                "Toplam Varlık (Canlı)",
                f"₺{_live_total_equity:,.2f}",
                delta=f"{_live_total_pnl:+,.2f} TL",
                delta_color="normal" if _live_total_pnl >= 0 else "inverse",
            )
        with pc3:
            realized = _portfolio["total_pnl"]
            unrealized = _live_total_pnl - realized
            st.metric("Gerçekleşmiş P&L", f"₺{realized:+,.2f}")
        with pc4:
            st.metric("Gerçekleşmemiş P&L", f"₺{unrealized:+,.2f}")
        with pc5:
            st.metric("Nakit", f"₺{_cash:,.2f}")

        st.markdown("")

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
        tt = _portfolio["total_trades"]
        wt = _portfolio["winning_trades"]
        lt = _portfolio["losing_trades"]
        wr = (wt / tt * 100) if tt else 0
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Toplam İşlem", str(tt))
        with c2:
            st.metric("Kazanan", str(wt), delta=f"%{wr:.0f}" if tt else None)
        with c3:
            st.metric("Kaybeden", str(lt))
        with c4:
            # Canlı P&L kullan (snapshot yerine)
            st.metric(
                "Net P&L",
                f"₺{_live_total_pnl:+,.2f}",
                delta=f"%{_live_pnl_pct:+.1f}",
                delta_color="normal" if _live_total_pnl >= 0 else "inverse",
            )

# ══════════════════════════════════════════════════════════════════
# TAB 6 — ML ÖĞRENME
# ══════════════════════════════════════════════════════════════════
with tab6:
    try:
        from ml.learner import AdaptiveLearner, DEFAULT_WEIGHTS
        from database.db import _conn as _db_conn

        _learner = AdaptiveLearner(_db_conn)
        ml_summary = _learner.get_learning_summary()

        st.markdown("### 🧠 Adaptif ML Öğrenme Sistemi")
        st.caption("Sistem her kapanan işlemden öğrenir ve indikatör ağırlıklarını otomatik optimize eder.")

        # --- Üst metrikler ---
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Toplam Güncelleme", str(ml_summary["total_weight_updates"]))
        with mc2:
            st.metric("Kazanma Oranı", f"%{ml_summary['stats']['win_rate']:.1f}" if ml_summary['stats']['total'] else "—")
        with mc3:
            st.metric("Ort. P&L", f"%{ml_summary['stats']['avg_pnl']:+.2f}" if ml_summary['stats']['total'] else "—")
        with mc4:
            rf_label = "🟢 Aktif" if ml_summary["rf_model_status"] == "active" else "🟡 Veri Bekliyor"
            st.metric("RF Model", rf_label)

        st.divider()

        # --- Ağırlık karşılaştırma ---
        wc1, wc2 = st.columns(2)
        with wc1:
            st.markdown("#### Güncel vs Varsayılan Ağırlıklar")
            cur_w = ml_summary["current_weights"]
            def_w = ml_summary["default_weights"]
            ind_names = sorted(def_w.keys())

            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(
                name="Varsayılan",
                x=ind_names,
                y=[def_w[k] for k in ind_names],
                marker_color="#6366f1",
                opacity=0.5,
            ))
            fig_w.add_trace(go.Bar(
                name="Öğrenilmiş",
                x=ind_names,
                y=[cur_w.get(k, 0) for k in ind_names],
                marker_color="#10b981",
            ))
            fig_w.update_layout(
                barmode="group",
                height=350,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_w, use_container_width=True)

        with wc2:
            st.markdown("#### Ağırlık Değişimleri")
            changes = ml_summary["weight_changes"]
            ch_names = sorted(changes.keys(), key=lambda k: abs(changes[k]), reverse=True)
            ch_vals = [changes[k] for k in ch_names]
            ch_colors = ["#10b981" if v >= 0 else "#ef4444" for v in ch_vals]

            fig_ch = go.Figure(go.Bar(
                x=ch_vals,
                y=ch_names,
                orientation="h",
                marker_color=ch_colors,
                text=[f"{v:+.4f}" for v in ch_vals],
                textposition="auto",
            ))
            fig_ch.update_layout(
                height=350,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Değişim",
            )
            st.plotly_chart(fig_ch, use_container_width=True)

        # --- Sembol performansı ---
        st.divider()
        st.markdown("#### 📊 Sembol Performansı")
        sym_perf = ml_summary["symbol_performance"]
        if sym_perf:
            sp_df = pd.DataFrame(sym_perf)
            sp_df = sp_df[["symbol", "market", "total_trades", "wins", "losses", "total_pnl", "avg_pnl", "best_pnl", "worst_pnl"]]
            sp_df.columns = ["Sembol", "Piyasa", "İşlem", "Kazanç", "Kayıp", "Toplam P&L", "Ort. P&L", "En İyi", "En Kötü"]
            st.dataframe(
                sp_df.style.format({
                    "Toplam P&L": "₺{:+,.2f}",
                    "Ort. P&L": "₺{:+,.2f}",
                    "En İyi": "₺{:+,.2f}",
                    "En Kötü": "₺{:+,.2f}",
                }).map(
                    lambda v: "color: #10b981" if isinstance(v, (int, float)) and v > 0 else ("color: #ef4444" if isinstance(v, (int, float)) and v < 0 else ""),
                    subset=["Toplam P&L", "Ort. P&L"],
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Henüz kapanmış işlem yok. Sistem öğrenmeye başlayınca burada sembol bazlı performans görünecek.")

        # --- Öğrenme günlüğü ---
        st.divider()
        st.markdown("#### 📝 Son Öğrenme Kayıtları")
        try:
            with _db_conn() as _con:
                _ml_rows = _con.execute(
                    "SELECT symbol, direction, outcome, pnl, pnl_pct, composite_score, confidence, created_at "
                    "FROM indicator_performance ORDER BY id DESC LIMIT 20"
                ).fetchall()
            if _ml_rows:
                ml_df = pd.DataFrame([dict(r) for r in _ml_rows])
                ml_df.columns = ["Sembol", "Yön", "Sonuç", "P&L (₺)", "P&L (%)", "Skor", "Güven", "Tarih"]
                ml_df["Sonuç"] = ml_df["Sonuç"].map({"win": "✅ Kazanç", "loss": "❌ Kayıp"})
                ml_df["Yön"] = ml_df["Yön"].str.upper()
                st.dataframe(
                    ml_df.style.format({
                        "P&L (₺)": "₺{:+,.2f}",
                        "P&L (%)": "%{:+.2f}",
                        "Skor": "{:.1f}",
                        "Güven": "{:.0f}%",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Henüz öğrenme kaydı oluşmadı.")
        except Exception:
            st.info("Öğrenme veritabanı henüz hazır değil.")

    except ImportError:
        st.warning("ML modülü yüklenemedi. `scikit-learn` kurulumunu kontrol edin.")
    except Exception as e:
        st.error(f"ML sekme hatası: {e}")


# ─── Footer ──────────────────────────────────────────────────────

# ─── Canlı İşlem Akışı (Sidebar Alt) ────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("### 🔔 Son İşlemler")
    recent_trades = _all_trades[:5]
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
    recent_sigs = _all_signals[:5]
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
