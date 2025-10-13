# -*- coding: utf-8 -*-
# NFL Parleggy AI Model – Live JSON ➜ Features ➜ Probabilities ➜ Parlays
# Author: Vish (Project Nova Analytics)
#
# Notes:
#  - Reads all *.json files in ./data (repo_root/data)
#  - Caches merged data for 30 min
#  - Two tabs:
#      (1) Player Probability Model (single leg)
#      (2) Parlay Probability Model (multi-leg)
#  - Nightly “retrain” bookkeeping @ 12:00 AM EST
#  - Robust column detection for typical SportsDataIO variations

import os, json, math, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import utils  # local helper module
import os
# ...
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="NFL Parleggy AI Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ STYLE ============
st.markdown("""
<style>
body {background:#0f1117;color:#e8eaf6;font-family:Inter,system-ui,Segoe UI;}
h1,h2,h3 {color:#00b4ff;font-weight:600;}
.card{background:#111b2e;border:1px solid #14244a;border-radius:10px;
      padding:0.8rem 1rem;margin:0.4rem 0;}
.badge{display:inline-block;border-radius:6px;padding:3px 7px;font-size:12px;margin-right:6px}
.badge.green{background:#0e4b2e;color:#b9f3d0}
.badge.red{background:#3b151a;color:#ffc9c9}
.badge.yellow{background:#45370c;color:#fff59d}
.metric{font-size:34px;font-weight:700;color:#eaffff}
.metric-sub{font-size:12px;color:#9fb0c0}
.footer{color:#889;font-size:12px;text-align:center;margin-top:18px;}
</style>
""", unsafe_allow_html=True)

# ============ SIDEBAR ============
with st.sidebar:
    st.subheader("Controls")
    if st.button("🔄 Retrain Now (All Markets)", use_container_width=True):
        st.session_state["last_retrain_utc"] = datetime.now(timezone.utc)
        st.cache_data.clear()
        st.success("Caches cleared — will reload JSON data on next run.")
    if "last_retrain_utc" in st.session_state:
        st.caption("**Last Retrain:** " +
                   st.session_state["last_retrain_utc"].strftime("%b %d %Y %H:%M UTC"))
    else:
        st.caption("**Last Retrain:** never")
    st.caption("Auto-refresh every 30 min • Full retrain nightly @ 12:00 AM EST")

# ============ DATA LOAD ============
@st.cache_data(ttl=1800, show_spinner=False)
def _load_all():
    # read + normalize
    df_raw = utils.load_all_jsons(DATA_DIR)
    df = utils.standardize_columns(df_raw)
    return df

def _load_data_ready():
    return _load_all()

# later where you currently do:
# data = _load_all_jsons()
# replace with:
data = _load_data_ready()
st.title("🏈 NFL Parleggy AI Model")
st.caption("Live JSON → Feature Engine → Probabilities → Parlay Calculator")

tab1, tab2 = st.tabs(["Player Probability Model", "Parlay Probability Model"])

# ============ TAB 1 — Player Probability ============
with tab1:
    if data.empty:
        st.error("No JSON data found — add weekly player files to /data.")
        st.stop()

    name_col = utils.detect_name_column(data)
    players = sorted(data[name_col].dropna().unique().tolist())

    colA, colB, colC, colD = st.columns([1.2, 1.0, 0.9, 1.1])
    with colA:
        player = st.selectbox("Player", players)
    with colB:
        market = st.selectbox("Pick Type / Market", list(utils.MARKET_KEYWORDS.keys()))
    with colC:
        sportsbook_line = st.number_input("Sportsbook Line", value=50.0, step=0.5)
    with colD:
        lookback = st.slider("Lookback (weeks)", 1, 8, value=5)

    opp = st.text_input("Opponent (e.g., KC, BUF, PHI)", "")
    run = st.button("Analyze Player", type="primary")

    if run:
        dfp = data[data[name_col].astype(str) == str(player)].copy()
        opp_col = utils.detect_opp_column(dfp)
        if opp and opp_col in dfp.columns:
            dfp = dfp[dfp[opp_col].astype(str).str.contains(opp, case=False, na=False)]

        week_col = utils.detect_week_column(dfp)
        if week_col: dfp = dfp.sort_values(week_col, ascending=False)
        df_recent = dfp.head(lookback).copy()

        target_col = utils.find_target_column(data, market)
        if not target_col or target_col not in df_recent.columns:
            st.error(f"Could not find stat column for {market}.")
            st.stop()

        vals = pd.to_numeric(df_recent[target_col], errors="coerce").dropna().to_numpy()
        mean_pred = np.mean(vals) if len(vals) else float("nan")
        p_over = utils.normal_over_probability(vals, sportsbook_line)
        p_under = 1 - p_over if not math.isnan(p_over) else float("nan")
        conf = utils.confidence_score(vals)

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Mean", f"{mean_pred:,.1f}")
        m2.metric("Over Probability", f"{p_over*100:.1f} %")
        m3.metric("Under Probability", f"{p_under*100:.1f} %")

        color = "red" if conf < 0.45 else "yellow" if conf < 0.75 else "green"
        level = "Low" if conf < 0.45 else "Moderate" if conf < 0.75 else "High"
        st.markdown(
            f'<div class="card"><span class="badge {color}">{level} Confidence</span>'
            f'<span class="metric-sub"> Model confidence: {conf*100:.1f}% • Samples: {len(vals)}</span></div>',
            unsafe_allow_html=True
        )

        st.subheader("Recent Weeks (table)")
        show_cols = [name_col, week_col, opp_col, target_col]
        show_cols = [c for c in show_cols if c in df_recent.columns]
        st.dataframe(df_recent[show_cols], use_container_width=True)

        st.subheader("Market Distribution Preview")
        if len(vals):
            fig = px.histogram(pd.DataFrame({"Value": vals}),
                               x="Value", nbins=20,
                               title=f"{player} – {market}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for distribution plot.")

# ============ TAB 2 — Parlay Probability ============
with tab2:
    if data.empty:
        st.info("No JSON data available yet.")
    else:
        name_col = utils.detect_name_column(data)
        players = sorted(data[name_col].dropna().unique().tolist())
        markets = list(utils.MARKET_KEYWORDS.keys())

        st.write("Build up to 4 legs (assuming independence).")
        legs = []
        for i in range(1, 5):
            with st.expander(f"Leg {i}", expanded=(i == 1)):
                c1, c2, c3, c4 = st.columns(4)
                p = c1.selectbox("Player", players, key=f"p{i}")
                m = c2.selectbox("Market", markets, key=f"m{i}")
                l = c3.number_input("Line", value=50.0, step=0.5, key=f"l{i}")
                w = c4.slider("Lookback", 1, 8, value=5, key=f"w{i}")
                legs.append((p, m, l, w))

        if st.button("Compute Parlay Probability", type="primary"):
            joint = 1.0
            leg_info = []
            for p, m, l, w in legs:
                if not p: continue
                col = utils.find_target_column(data, m)
                dfp = data[data[name_col] == p].copy()
                wk = utils.detect_week_column(dfp)
                if wk: dfp = dfp.sort_values(wk, ascending=False)
                dfp = dfp.head(w)
                vals = pd.to_numeric(dfp[col], errors="coerce").dropna().to_numpy()
                pov = utils.normal_over_probability(vals, l)
                leg_info.append((p, m, pov))
                if not math.isnan(pov): joint *= pov

            df_show = pd.DataFrame(leg_info, columns=["Player", "Market", "Over Prob"])
            st.dataframe(df_show, use_container_width=True)
            if len(leg_info):
                st.success(f"Joint Over Probability: **{joint*100:.2f}%**")

# ============ FOOTER ============
st.markdown(
    f'<div class="footer">Auto-refresh every 30 min • Last update '
    f'{datetime.now(timezone.utc).strftime("%b %d %Y %H:%M UTC")} • © 2025 Project Nova Analytics</div>',
    unsafe_allow_html=True
)
