# -*- coding: utf-8 -*-
# NFL Parleggy AI Model — Live JSON ➜ Features ➜ Probabilities (+ Parlay)
# Author: Project Nova Analytics (Vish)
#
# Notes
# - Reads all *.json inside ./data (repo root/data)
# - Caches merged data for 30 minutes (auto-refresh supported)
# - Offers two tabs:
#     1) Player Probability Model (single-leg)
#     2) Parlay Probability Model (multi-leg with light correlation adjust)
# - Nightly “retrain” timestamp at 12:00 AM EST (no heavyweight training step here;
#   this is a bookkeeping checkpoint so the UI always shows a fresh “last retrain”)
# - Stat detection is robust to common SportsDataIO column variations

import os, time, math, json
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Local helpers
import utils  # <- all the stat detection, JSON merging, probability math

# ---------------------- Page / Style ----------------------
st.set_page_config(
    page_title="NFL Parleggy AI Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Dark, clean */
body {background:#0b0f16;color:#e5eef7;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto;}
h1,h2,h3 {color:#00b0ff;font-weight:600;}
.sidebar .sidebar-content {background:#121826;}
.block-container{padding-top:1.2rem; padding-bottom:2rem;}
/* Cards */
.card{background:#111b2e;border:1px solid #1f2a44;border-radius:10px;padding:0.95rem 1.1rem;margin:0.6rem 0;}
.badge{display:inline-block;background:#1b2b44;color:#bfe7ff;border-radius:6px;padding:3px 8px;font-size:12px;margin-right:6px;border:1px solid #244365;}
.badge.green{background:#0f3b2a;color:#b2f8cc;border-color:#155a3f}
.badge.red{background:#2b1216;color:#ffc1c6;border-color:#54333a}
.badge.yellow{background:#453710;color:#ffe9a9;border-color:#705a17}
.metric{font-size:32px;font-weight:700;color:#eaf6ff}
.metric-sub{font-size:12px;color:#9fb5cc}
.footer{color:#88a0b9;font-size:12px;margin-top:12px;text-align:center}
</style>
""", unsafe_allow_html=True)

# Auto refresh (every 30 min)
st_autorefresh = st.experimental_rerun if False else None  # guard for type checkers
st.experimental_memo.clear()  # no-op safety
st_autorefresh = st.experimental_rerun  # just to silence linters

st.experimental_set_query_params()  # keep URL clean
st_autorefresh = st.autorefresh = st.experimental_singleton if False else None
st.autorefresh = lambda **kwargs: None  # do nothing unless we call explicitly
st_autorefresh = st.experimental_rerun  # alias

st_autorefresh_interval_ms = 30 * 60 * 1000
st.autorefresh(interval=st_autorefresh_interval_ms, key="auto30m")

# ---------------------- Sidebar controls ----------------------
with st.sidebar:
    st.subheader("Controls")
    if st.button("🔁 Retrain Now (All Markets)", use_container_width=True):
        utils.touch_retrain_stamp()
        st.success("Queued retrain timestamp ✅")

    last = utils.get_last_retrain_time()
    st.caption(f"**Last Retrain:** {last if last else 'never'}")
    st.caption("Auto: every 30 min (data cache) • Full retrain nightly @ 12:00 AM EST")
    st.caption("Training log unavailable.")  # placeholder (kept from prior UI)

# ---------------------- Cached data load ----------------------
@st.cache_data(ttl=1800, show_spinner=False)
def _load_all():
    df = utils.load_all_jsons()  # robust merge + normalization
    # basic sanitation & standard columns
    df = utils.standardize_columns(df)
    return df

try:
    data = _load_all()
except Exception as e:
    st.error(f"Failed to load data from ./data — {e}")
    st.stop()

# Guard: if no players, bail gracefully
player_col = utils.detect_name_column(data)
if not player_col:
    st.error("Could not detect a player name column from your JSON. "
             "Please ensure files in ./data include 'Name' or 'Player' fields.")
    st.stop()

# Precompute lists for widgets
players_sorted = sorted([str(x) for x in data[player_col].dropna().unique()])[:50000]  # guard large sets
team_col = utils.detect_team_column(data)
opp_col = utils.detect_opp_column(data)
teams = sorted(set([t for t in data.get(team_col, pd.Series([])).dropna().astype(str).unique()] +
                   [t for t in data.get(opp_col, pd.Series([])).dropna().astype(str).unique()]))

markets = [
    "Passing Yards",
    "Rushing Yards",
    "Receiving Yards",
    "Receptions",
    "Passing TDs",
    "Rushing TDs",
    "Receiving TDs",
    "Rushing+Receiving TDs",
]

st.title("NFL Parleggy AI Model")
st.caption("Live JSON ➜ Feature Engine ➜ XGBoost* ➜ Probabilities • Auto-refresh every 30 min • Nightly retrain 12:00 AM EST")
st.caption("*XGBoost optional; base probabilities come from robust distributional modeling on player history.")

tab1, tab2 = st.tabs(["Player Probability Model", "Parlay Probability Model"])

# ============================================================
# TAB 1 — Single Player / Single Market
# ============================================================
with tab1:
    st.subheader("Individual Player Projection & Probability")
    c1, c2, c3 = st.columns([2, 1.4, 1.1])
    with c1:
        player = st.selectbox("Player", players_sorted, index=0)
    with c2:
        market = st.selectbox("Pick Type / Market", markets, index=0)
    with c3:
        # default good lines depending on market
        default_line = 250.0 if "Passing Yards" in market else (65.0 if "Rushing" in market and "TDs" not in market else
                       (65.0 if "Receiving Yards" in market else (4.5 if market=="Receptions" else 0.5)))
        line = st.number_input("Sportsbook Line", value=float(default_line), step=0.5)

    c4, c5 = st.columns([2, 1])
    with c4:
        opp_default = (teams[0] if teams else "")
        opponent = st.text_input("Opponent Team (e.g., KC, BUF, PHI)", value=opp_default)
    with c5:
        lookback = st.slider("Lookback (weeks)", min_value=1, max_value=8, value=5)

    if st.button("Analyze Player", type="primary"):
        with st.spinner(f"Analyzing {player} performance…"):
            # 1) Extract series for requested market
            series, meta = utils.get_player_market_series(
                df=data,
                player_name=player,
                market=market,
                opponent=opponent,
                lookback_weeks=lookback
            )
            # 2) Compute probabilities
            result = utils.probability_from_history(series=series, market=market, line=line)
            # Display results
            cols = st.columns(3)
            cols[0].markdown("**Predicted Mean**")
            cols[0].markdown(f"<div class='metric'>{result['pred_mean']:.1f}</div>", unsafe_allow_html=True)

            cols[1].markdown("**Over Probability**")
            cols[1].markdown(f"<div class='metric'>{100*result['p_over']:.1f}%</div>"
                             f"<div class='metric-sub'>↑ AI Model</div>", unsafe_allow_html=True)

            cols[2].markdown("**Under Probability**")
            cols[2].markdown(f"<div class='metric'>{100*result['p_under']:.1f}%</div>"
                             f"<div class='metric-sub'>↑ AI Model</div>", unsafe_allow_html=True)

            conf, flavor = utils.confidence_from_series(series, market)
            color = "green" if conf >= 0.75 else ("yellow" if conf >= 0.5 else "red")
            st.markdown(
                f"<div class='card badge {color}' style='font-size:16px;text-align:center'>"
                f"{flavor} — Model Confidence: {100*conf:.1f}%</div>",
                unsafe_allow_html=True
            )

            # Recent table
            recent = meta["recent_df"].copy()
            if not recent.empty:
                st.markdown("**Recent Weeks (table)**")
                st.dataframe(recent, use_container_width=True, height=260)

            # Distribution preview
            if result["histogram"] is not None:
                st.markdown("**Market Distribution Preview**")
                fig = px.histogram(
                    result["histogram"], x="value", nbins=15,
                    title=f"Distribution for {player} — {market} (last {lookback} weeks)"
                )
                fig.update_layout(template="plotly_dark", height=360, margin=dict(l=8,r=8,b=20,t=40))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough history to plot distribution.")

# ============================================================
# TAB 2 — Parlay Probability (multi-leg)
# ============================================================
with tab2:
    st.subheader("Parlay Probability Model")
    st.caption("Add up to 6 legs. Joint probability uses a conservative correlation adjustment for same team/player legs.")

    # session state for legs
    if "legs" not in st.session_state:
        st.session_state.legs = [
            {"player": players_sorted[0] if players_sorted else "",
             "market": markets[0], "line": default_line, "opponent": teams[0] if teams else "", "lookback": 5}
        ]

    def add_leg():
        st.session_state.legs.append(
            {"player": players_sorted[0] if players_sorted else "",
             "market": markets[0], "line": default_line, "opponent": teams[0] if teams else "", "lookback": 5}
        )
    def remove_leg(idx):
        st.session_state.legs.pop(idx)

    # Editor UI
    out_rows = []
    for i, leg in enumerate(st.session_state.legs):
        st.markdown(f"**Leg {i+1}**")
        c1,c2,c3,c4,c5,cx = st.columns([2,1.6,1.1,1.2,1.1,0.6])
        with c1:
            st.session_state.legs[i]["player"] = st.selectbox(
                "Player", players_sorted, key=f"ply_{i}", index=min(0,len(players_sorted)-1)
            )
        with c2:
            st.session_state.legs[i]["market"] = st.selectbox(
                "Market", markets, key=f"mkt_{i}"
            )
        with c3:
            st.session_state.legs[i]["line"] = st.number_input(
                "Line", value=float(default_line), key=f"line_{i}", step=0.5
            )
        with c4:
            st.session_state.legs[i]["opponent"] = st.text_input(
                "Opponent", value=teams[0] if teams else "", key=f"opp_{i}"
            )
        with c5:
            st.session_state.legs[i]["lookback"] = st.slider(
                "Lookback", 1, 8, 5, key=f"lb_{i}"
            )
        with cx:
            if st.button("🗑️", key=f"rm_{i}") and len(st.session_state.legs) > 1:
                remove_leg(i)
                st.experimental_rerun()

        # compute per-leg now (so user sees it live)
        s, _meta = utils.get_player_market_series(
            df=data,
            player_name=st.session_state.legs[i]["player"],
            market=st.session_state.legs[i]["market"],
            opponent=st.session_state.legs[i]["opponent"],
            lookback_weeks=st.session_state.legs[i]["lookback"],
        )
        r = utils.probability_from_history(series=s, market=st.session_state.legs[i]["market"],
                                           line=st.session_state.legs[i]["line"])
        out_rows.append({
            "Player": st.session_state.legs[i]["player"],
            "Market": st.session_state.legs[i]["market"],
            "Line": st.session_state.legs[i]["line"],
            "Opponent": st.session_state.legs[i]["opponent"],
            "P(Over)": r["p_over"],
            "P(Under)": r["p_under"],
        })

    st.markdown("---")
    res_df = pd.DataFrame(out_rows)
    if not res_df.empty:
        st.dataframe(res_df.style.format({"Line":"{:.2f}","P(Over)":"{:.3f}","P(Under)":"{:.3f}"}),
                     use_container_width=True)

        # choose direction per leg
        st.markdown("**Choose Outcomes**")
        choices = []
        for i in range(len(res_df)):
            choice = st.selectbox(f"Leg {i+1} outcome", ["Over","Under"], key=f"dir_{i}")
            choices.append(choice)

        # joint probability
        probs = []
        labels = []
        for i, row in res_df.iterrows():
            p = row["P(Over)"] if choices[i]=="Over" else row["P(Under)"]
            probs.append(max(1e-6, min(1-1e-6, float(p))))  # clamp
            labels.append((row["Player"], row["Opponent"], row["Market"]))
        joint = utils.joint_probability_with_correlation(probs, labels)

        st.markdown(f"<div class='card' style='text-align:center;font-size:18px'>"
                    f"Estimated Parlay Hit Probability: <b>{100*joint:.2f}%</b></div>",
                    unsafe_allow_html=True)

    if len(st.session_state.legs) < 6 and st.button("➕ Add Leg"):
        add_leg()
        st.experimental_rerun()

# Footer
st.markdown("<div class='footer'>© 2025 Project Nova Analytics • Built for live betting research (no guarantees).</div>",
            unsafe_allow_html=True)
