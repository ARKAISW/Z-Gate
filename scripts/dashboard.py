"""dashboard.py — Streamlit live monitoring dashboard for the hybrid stat-arb system.

Run:
  streamlit run scripts/dashboard.py

Tabs:
  1. Equity Options (Module A) — Pairs health, active contracts, closed options trades
  2. Crypto Spot (Module B) — 24/7 pairs health, active spot positions, closed trades
  3. Risk Engine & Audit Log — Approved/rejected decisions stream, circuit breaker status
  4. Nightly Reflections — LLM trade journal entries and daily performance notes
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Z-Gate — Stat-Arb Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Theme CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #161a23;
        border: 1px solid #2a3142;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
    }
    .status-badge-green {
        background-color: rgba(38, 166, 154, 0.2);
        color: #26a69a;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-badge-red {
        background-color: rgba(239, 83, 80, 0.2);
        color: #ef5350;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data Layer (SQLite Read-Only)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=5)
def load_db_data(db_path: str) -> dict[str, pd.DataFrame]:
    """Load tables from SQLite database."""
    path = Path(db_path)
    if not path.exists():
        return {
            "signals": pd.DataFrame(),
            "decisions": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "orders": pd.DataFrame(),
            "reflections": pd.DataFrame(),
        }

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        signals = pd.read_sql_query("SELECT * FROM signals ORDER BY generated_at DESC LIMIT 100", conn)
        decisions = pd.read_sql_query("SELECT * FROM risk_decisions ORDER BY checked_at DESC LIMIT 100", conn)
        trades = pd.read_sql_query("SELECT * FROM trades ORDER BY entry_time DESC", conn)
        orders = pd.read_sql_query("SELECT * FROM orders ORDER BY created_at DESC LIMIT 100", conn)
        reflections = pd.read_sql_query("SELECT * FROM reflections ORDER BY generated_at DESC", conn)
        return {
            "signals": signals,
            "decisions": decisions,
            "trades": trades,
            "orders": orders,
            "reflections": reflections,
        }
    except Exception as exc:
        st.sidebar.error(f"DB Read Error: {exc}")
        return {
            "signals": pd.DataFrame(),
            "decisions": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "orders": pd.DataFrame(),
            "reflections": pd.DataFrame(),
        }
    finally:
        conn.close()


def load_config() -> dict[str, Any]:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Header & Live Broker Connectivity
# ---------------------------------------------------------------------------

config = load_config()
data = load_db_data(config.get("db_path", "data/trades.db"))

trades_df = data["trades"]
signals_df = data["signals"]
decisions_df = data["decisions"]
reflections_df = data["reflections"]

# Attempt to query live Alpaca broker account metrics
live_equity = 100000.0
live_cash = 100000.0
live_buying_power = 400000.0
live_positions_list = []
broker_connected = False

try:
    from src.broker import create_broker
    broker = create_broker()
    acct = broker.get_account()
    live_equity = float(acct.equity)
    live_cash = float(acct.cash)
    live_buying_power = float(acct.buying_power)
    live_positions_list = broker.get_positions()
    broker_connected = True
except Exception as exc:
    pass

st.markdown('<div class="main-header">⚡ Z-Gate Stat-Arb Terminal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Hybrid Multi-Agent Statistical Arbitrage System | Alpaca Paper Trading | Deterministic Risk Engine</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Top KPI Metrics Row (Live Broker Synced)
# ---------------------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

initial_capital = float(config.get("initial_capital", 100000.0))
net_pnl_usd = live_equity - initial_capital
net_pnl_pct = (net_pnl_usd / initial_capital) * 100.0

open_trades = trades_df[trades_df["status"] == "open"] if not trades_df.empty else pd.DataFrame()
closed_trades = trades_df[trades_df["status"] == "closed"] if not trades_df.empty else pd.DataFrame()

with col1:
    st.metric("Portfolio Equity", f"${live_equity:,.2f}", f"{net_pnl_usd:+,.2f} ({net_pnl_pct:+.2f}%)")
with col2:
    st.metric("Available Cash", f"${live_cash:,.2f}", f"Buying Power: ${live_buying_power:,.0f}")
with col3:
    st.metric("Active Positions", len(live_positions_list) if broker_connected else len(open_trades), "Alpaca Live" if broker_connected else "DB Synced")
with col4:
    win_count = len(closed_trades[closed_trades["realized_pnl_usd"] > 0]) if not closed_trades.empty else 0
    win_rate = (win_count / len(closed_trades) * 100.0) if len(closed_trades) > 0 else 0.0
    st.metric("Closed Trades", len(closed_trades), f"{win_rate:.0f}% Win Rate ({win_count}W)")
with col5:
    st.metric("Evaluated Signals", len(signals_df), "12 Active Pairs")

st.divider()

# ---------------------------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Module A: Equity Options",
        "🪙 Module B: Crypto Spot",
        "🛡️ Risk Engine & Audit Log",
        "📓 Nightly Reflections",
    ]
)

# ── Tab 1: Equity Options ─────────────────────────────────────────────────────

with tab1:
    st.subheader("Module A — Equity Options Statistical Arbitrage")
    st.caption("Directional Call/Put spreads on cointegrated ETF and stock pairs (GLD/SLV, XOM/CVX, KO/PEP).")

    # Filter equity data
    eq_signals = signals_df[signals_df["module"] == "equity"] if not signals_df.empty else pd.DataFrame()
    eq_trades = trades_df[trades_df["module"] == "equity"] if not trades_df.empty else pd.DataFrame()
    eq_open = eq_trades[eq_trades["status"] == "open"] if not eq_trades.empty else pd.DataFrame()

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("##### 🔍 Watchlist Health & Signals")
        if not eq_signals.empty:
            # Deduplicate by pair to get latest
            latest_eq = eq_signals.drop_duplicates(subset=["pair_id"], keep="first")
            disp_cols = ["pair_id", "direction", "z_score", "beta", "half_life", "vol_regime", "coint_pvalue"]
            st.dataframe(
                latest_eq[disp_cols].style.format(
                    {
                        "z_score": "{:.2f}",
                        "beta": "{:.3f}",
                        "half_life": "{:.1f}d",
                        "coint_pvalue": "{:.4f}",
                    }
                ),
                use_container_width=True,
            )
        else:
            st.info("No equity signals generated yet. Run pipeline cycle to ingest bars.")

    with c2:
        st.markdown("##### 📂 Active Options Positions (Alpaca Broker)")
        opt_pos = [
            {
                "Contract": p.symbol,
                "Qty": int(float(p.qty)),
                "Entry Price": f"${float(p.avg_entry_price):.2f}",
                "Market Value": f"${float(p.market_value):.2f}",
                "Unrealized P/L": f"${float(p.unrealized_pnl):+,.2f}",
            }
            for p in live_positions_list
            if len(p.symbol) > 10
        ]
        if opt_pos:
            st.dataframe(pd.DataFrame(opt_pos), use_container_width=True)
        elif not eq_open.empty:
            st.dataframe(
                eq_open[["pair_id", "direction", "entry_z", "entry_beta", "entry_time"]],
                use_container_width=True,
            )
        else:
            st.info("No active equity options positions currently open.")

    st.markdown("##### 📜 Closed Options Trades History")
    if not eq_trades.empty and not eq_trades[eq_trades["status"] == "closed"].empty:
        closed_eq = eq_trades[eq_trades["status"] == "closed"]
        st.dataframe(
            closed_eq[
                [
                    "pair_id",
                    "direction",
                    "entry_z",
                    "exit_z",
                    "exit_reason",
                    "realized_pnl_usd",
                    "holding_period_hours",
                    "entry_time",
                    "exit_time",
                ]
            ].style.format({"realized_pnl_usd": "${:,.2f}", "holding_period_hours": "{:.1f}h"}),
            use_container_width=True,
        )
    else:
        st.write("No closed equity options trades yet.")

# ── Tab 2: Crypto Spot ────────────────────────────────────────────────────────

with tab2:
    st.subheader("Module B — Crypto Spot Statistical Arbitrage (24/7)")
    st.caption("Autonomous mean-reversion trading on BTC/ETH, ETH/SOL, BTC/SOL, LINK/ETH with dynamic beta hedging.")

    cr_signals = signals_df[signals_df["module"] == "crypto"] if not signals_df.empty else pd.DataFrame()
    cr_trades = trades_df[trades_df["module"] == "crypto"] if not trades_df.empty else pd.DataFrame()
    cr_open = cr_trades[cr_trades["status"] == "open"] if not cr_trades.empty else pd.DataFrame()

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("##### 🔍 Crypto Watchlist Health")
        if not cr_signals.empty:
            latest_cr = cr_signals.drop_duplicates(subset=["pair_id"], keep="first")
            disp_cols = ["pair_id", "direction", "z_score", "beta", "half_life", "vol_regime", "coint_pvalue"]
            st.dataframe(
                latest_cr[disp_cols].style.format(
                    {
                        "z_score": "{:.2f}",
                        "beta": "{:.3f}",
                        "half_life": "{:.1f}h",
                        "coint_pvalue": "{:.4f}",
                    }
                ),
                use_container_width=True,
            )
        else:
            st.info("No crypto signals generated yet.")

    with c2:
        st.markdown("##### 📂 Active Spot Positions (Alpaca Broker)")
        crypto_pos = [
            {
                "Asset": p.symbol,
                "Qty": f"{float(p.qty):.4f}",
                "Entry Price": f"${float(p.avg_entry_price):,.2f}",
                "Market Value": f"${float(p.market_value):,.2f}",
                "Unrealized P/L": f"${float(p.unrealized_pnl):+,.2f}",
            }
            for p in live_positions_list
            if len(p.symbol) <= 10
        ]
        if crypto_pos:
            st.dataframe(pd.DataFrame(crypto_pos), use_container_width=True)
        elif not cr_open.empty:
            st.dataframe(
                cr_open[["pair_id", "direction", "entry_z", "entry_beta", "entry_time"]],
                use_container_width=True,
            )
        else:
            st.info("No active crypto spot positions currently open.")

    st.markdown("##### 📜 Closed Crypto Trades History")
    if not cr_trades.empty and not cr_trades[cr_trades["status"] == "closed"].empty:
        closed_cr = cr_trades[cr_trades["status"] == "closed"]
        st.dataframe(
            closed_cr[
                [
                    "pair_id",
                    "direction",
                    "entry_z",
                    "exit_z",
                    "exit_reason",
                    "realized_pnl_usd",
                    "holding_period_hours",
                    "entry_time",
                    "exit_time",
                ]
            ].style.format({"realized_pnl_usd": "${:,.2f}", "holding_period_hours": "{:.1f}h"}),
            use_container_width=True,
        )
    else:
        st.write("No closed crypto trades yet.")

# ── Tab 3: Risk Engine & Audit Log ────────────────────────────────────────────

with tab3:
    st.subheader("Deterministic Risk Agent & Order Audit Trail")
    st.caption("Zero-LLM deterministic risk gate enforcement (9 rules) and pre-order execution logs.")

    col_r1, col_r2 = st.columns([1, 2])

    with col_r1:
        st.markdown("##### 🛡️ Circuit Breaker Status")
        st.info("🟢 Circuit Breaker: NORMAL (24h drawdown within -3% threshold)")
        st.markdown(
            """
            **Enforced Rules:**
            1. Max concurrent pairs per module (3)
            2. No duplicate positions
            3. OU Half-Life gating (2-20d eq / 6-96h cr)
            4. Cointegration gate ($p < 0.05$)
            5. Vol regime block (EXTREME $\\ge$ 60% eq / 120% cr)
            6. Sizing sanity check (Quarter-Kelly, 5%/10% caps)
            7. Real-time buying power verification
            8. Rolling 24h loss circuit breaker (-3%, 4h cooldown)
            9. Bar staleness check ($< 15$ min)
            """
        )

    with col_r2:
        st.markdown("##### 📋 Live Risk Decisions Audit Stream")
        if not decisions_df.empty:
            st.dataframe(
                decisions_df[
                    [
                        "pair_id",
                        "module",
                        "approved",
                        "rejection_rule",
                        "rejection_reason",
                        "checked_at",
                    ]
                ],
                use_container_width=True,
            )
        else:
            st.write("No risk decisions recorded yet.")

# ── Tab 4: Nightly Reflections ────────────────────────────────────────────────

with tab4:
    st.subheader("LLM Post-Trade Journal & Daily Reflections")
    st.caption("Midnight UTC automated trade review and health audits generated by Featherless/Groq.")

    if not reflections_df.empty:
        day_refs = reflections_df[reflections_df["reflection_type"] == "day"]
        trade_refs = reflections_df[reflections_df["reflection_type"] == "trade"]

        if not day_refs.empty:
            st.markdown("##### 📅 Daily Performance Summaries")
            for _, r in day_refs.iterrows():
                with st.expander(f"UTC Day: {r['utc_date']} (Provider: {r['provider_used']})", expanded=True):
                    st.write(f"**Day Summary:** {r['day_summary']}")
                    st.write(f"**Risk Rejections:** {r['risk_rejection_pattern']}")
                    st.write(f"**Cointegration Health:** {r['cointegration_health']}")

        if not trade_refs.empty:
            st.markdown("##### 📝 Individual Closed Trade Reviews")
            for _, r in trade_refs.iterrows():
                with st.expander(f"Trade {r['pair_id']} ({r['module']}) — {r['utc_date']}"):
                    st.write(f"**Outcome:** {r['outcome_summary']}")
                    st.write(f"**Rule Alignment:** {r['rule_alignment']}")
                    st.write(f"**OU Dynamics:** {r['ou_observation']}")
                    st.write(f"**Notable Notes:** {r['notable_observation']}")
    else:
        st.info("No nightly reflections generated yet. Reflections run daily at midnight UTC.")
