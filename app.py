# -*- coding: utf-8 -*-
"""
NFL Parleggy AI Model – Live JSON ➜ Features ➜ Probabilities (+ Parlay)
Author: Project Nova Analytics (Vish)

- Reads every *.json inside ./data
- Caches merged data (30 min) so the app stays snappy
- Two tabs: (1) Player Probability Model (2) Parlay Probability Model
- Confidence panel + thin-data emoji
"""

import os, time, math, json
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import utils  # our helper module (kept in repo)

# ---- Page config (must be FIRST Streamlit call) ----
st.set_page_config(
    page_title="NFL Parleggy AI Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Style (dark professional) ----
st.markdown(
    """
    <style>
      body {background:#0f1117; color:#e8e8ef; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto;}
      h1,h2,h3 {color:#00b0ff; font-weight:600;}
      .stTabs [data-baseweb="tab"] { color:#e8e8ef; }
      .block-container {padding-top:1.2rem;}
      .metric-sub {font-size:12px; color:#9aa4b2;}
      .badge {display:inline-block; padding:.35rem .6rem; border-radius:8px; font-size:12px;}
      .badge.green {background:#1f6f4a; color:#c8ffe0;}
      .badge.yellow{background:#544a13; color:#ffe38f;}
      .badge.red   {background:#4b1c1c; color:#ffb3b3;}
      .panel {border:1px solid #26334d; background:#111827; border-radius:12px; padding:14px;}
      .footer {text-align:center; font-size:12px; color:#8a8f98; margin-top:20px;}
      .warn {background:#3a1d1d; color:#ffd1d1; padding:.6rem .8rem; border-radius:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = os.path.join(os.getcwd(), "data")

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Controls")
    if st.button("🔁 Refresh now (all markets)", use_container_width=True):
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception:
            pass
        st.rerun()

    last_check = utils.last_checked()
    st.caption(f"Last refresh: {last_check} UTC")
    st.caption("Auto: every 30 min (data cache) • Full retrain nightly @ 12:00 AM EST")

    # Tiny model-health sparkline (MAE timestamps we log to file)
    mae_hist = utils.load_mae_history()
    if not mae_hist.empty:
        fig_m = px.line(
            mae_hist, x="ts", y="mae", title=None, height=180, markers=True
        )
        fig_m.update_layout(
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="", showgrid=False, color="#aab3c5"),
            yaxis=dict(title="Mean Abs Error", showgrid=True, gridcolor="#2a3344", color="#aab3c5"),
        )
        st.plotly_chart(fig_m, use_container_width=True)

# ---- Data loader (cached 30 min) ----
@st.cache_data(ttl=1800, show_spinner=False)
def load_data():
    df = utils.load_all_jsons(DATA_PATH)
    df = utils.standardize_columns(df)
    return df

data = load_data()
utils.touch_last_checked()  # write 'last refreshed' timestamp

# Guard
if data.empty:
    st.error("No JSON data found in ./data. Drop your weekly files there (with API-filled content) and press Refresh.")
    st.stop()

# ---- Top banner ----
st.title("NFL Parleggy AI Model")
st.caption("Live JSON ➜ Feature Engine ➜ XGBoost* ➜ Probability • Auto-refresh every 30 min • Nightly retrain 12:00 AM EST")
st.caption("(* XGBoost optional; falls back to robust moving-average if samples are thin)")

tab1, tab2 = st.tabs(["Player Probability Model", "Parlay Probability Model"])

# ========== Tab 1: Single-leg Player Model ==========
with tab1:
    st.subheader("Individual Player Projection & Probability")

    # Build selectors
    name_col = utils.detect_name_column(data)
    opp_col  = utils.detect_opponent_column(data)
    week_col = utils.detect_week_column(data)

    players = sorted(list(pd.Series(data[name_col]).dropna().astype(str).unique()))
    markets = utils.MARKET_OPTIONS  # canonical display names

    c1, c2, c3 = st.columns([1.4, 1.1, 1.0])
    player = c1.selectbox("Player", players, index=0)
    market = c2.selectbox("Pick Type / Market", markets, index=0)
    line = c3.number_input("Sportsbook Line", value=50.0, step=0.5, format="%.2f")

    c4, c5 = st.columns([1.3, 1])
    opponent = c4.text_input("Opponent (e.g., KC, BUF, PHI)", value="")
    lookback = int(c5.slider("Lookback (weeks)", 1, 8, 5))

    run = st.button("Analyze Player", type="primary")

    # Output placeholders
    mcol1, mcol2, mcol3 = st.columns([1, 1, 1])
    panel = st.container()

    tbl = st.container()
    dist_preview = st.container()

    if run:
        try:
            # Slice player data (last N games)
            pdf = utils.slice_player(
                data=data, player=player, opponent=opponent,
                lookback_weeks=lookback, week_col=week_col, name_col=name_col
            )
            target = utils.get_market_series(pdf, market)

            if target is None or len(target.dropna()) == 0:
                st.markdown('<div class="warn">😕 No available stats for this metric in recent weeks.</div>', unsafe_allow_html=True)
                st.stop()

            # Fast projection (AI if enough data → else WMA fallback)
            proj = utils.predict_market(pdf, target, market)

            # Probability of OVER vs line (Normal approx; robust fallback)
            over_p, under_p, sigma = utils.prob_over_under(target, proj, line, market)

            # Confidence score (sample size + dispersion + recentness)
            conf, tags = utils.confidence_score(target, sigma, lookback)

            # KPIs
            mcol1.metric("Predicted Mean", f"{proj:.1f}")
            mcol2.metric("Over Probability", f"{over_p*100:.1f}%", delta="AI Model")
            mcol3.metric("Under Probability", f"{under_p*100:.1f}%", delta="AI Model")

            # Confidence panel
            bar_color = "green" if conf >= 0.75 else ("yellow" if conf >= 0.5 else "red")
            label = "High" if conf >= 0.75 else ("Moderate" if conf >= 0.5 else "Low")
            panel.markdown(
                f'<div class="panel badge {bar_color}" style="width:100%;text-align:center;">'
                f'{label} Confidence — Model Accuracy: {conf*100:.1f}% '
                f'</div>',
                unsafe_allow_html=True,
            )
            if conf < 0.5:
                st.markdown('<div class="warn">🤖⚠️ Thin or noisy data — please interpret carefully.</div>', unsafe_allow_html=True)

            # Distribution preview chart
            x_min = float(min(target.min(), line, proj) - 0.25 * abs(proj if proj != 0 else 1))
            x_max = float(max(target.max(), line, proj) + 0.25 * abs(proj if proj != 0 else 1))
            chart_df = pd.DataFrame({
                "Outcome": ["Over", "Under"],
                "Probability": [over_p * 100, under_p * 100],
            })
            fig = px.bar(chart_df, x="Outcome", y="Probability", title=f"Predicted Outcome Probabilities for {player}")
            fig.update_layout(
                paper_bgcolor="#0f1117",
                plot_bgcolor="#0f1117",
                yaxis=dict(range=[0, 100], gridcolor="#223049", color="#aab3c5"),
                xaxis=dict(color="#aab3c5"),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            dist_preview.plotly_chart(fig, use_container_width=True)

            # Recent table
            show_cols = utils.columns_for_table(pdf, market, name_col, opp_col, week_col)
            tbl.dataframe(pdf[show_cols].tail(8).iloc[::-1].reset_index(drop=True), use_container_width=True)

        except Exception as e:
            st.error(f"Unable to compute probability: {e}")

# ========== Tab 2: Parlay Builder ==========
with tab2:
    st.subheader("Parlay Probability Model")

    players = sorted(list(pd.Series(data[utils.detect_name_column(data)]).dropna().astype(str).unique()))
    markets = utils.MARKET_OPTIONS

    st.caption("Add up to 6 legs. We apply a conservative correlation dampener (default 0.90).")

    corr = st.slider("Correlation Dampener (lower = more conservative)", 0.70, 1.00, 0.90, 0.01)

    legs = []
    for i in range(1, 7):
        with st.expander(f"Leg {i}", expanded=(i <= 2)):
            c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.0, 1.0])
            p = c1.selectbox("Player", ["—"] + players, index=0, key=f"p{i}")
            m = c2.selectbox("Market", markets, index=0, key=f"m{i}")
            ln = c3.number_input("Line", value=50.0, step=0.5, key=f"l{i}")
            opp = c4.text_input("Opponent (optional)", value="", key=f"o{i}")
            add = st.checkbox("Include this leg", value=(i <= 2), key=f"a{i}")

            if add and p != "—":
                legs.append((p, m, ln, opp))

    if st.button("Compute Parlay Probability", type="primary"):
        if not legs:
            st.warning("Add at least one leg.")
        else:
            probs = []
            details = []
            for (p, m, ln, opp) in legs:
                try:
                    pdf = utils.slice_player(data, p, opp, 6, utils.detect_week_column(data), utils.detect_name_column(data))
                    y = utils.get_market_series(pdf, m)
                    pred = utils.predict_market(pdf, y, m)
                    over_p, under_p, _ = utils.prob_over_under(y, pred, ln, m)
                    probs.append(over_p)
                    details.append((p, m, ln, over_p))
                except Exception as e:
                    st.error(f"Leg error ({p}, {m}): {e}")
                    probs.append(0.0)

            # Combine with dampener
            base = float(np.prod(probs))
            adjusted = base * (corr ** max(0, len(probs) - 1))

            st.write("### Leg Details")
            leg_df = pd.DataFrame([{"Player": a, "Market": b, "Line": c, "Over Probability": f"{d*100:.1f}%"} for (a, b, c, d) in details])
            st.dataframe(leg_df, use_container_width=True)

            st.write("### Parlay Result")
            st.metric("Unadjusted Win Probability", f"{base*100:.2f}%")
            st.metric("Adjusted (corr) Win Probability", f"{adjusted*100:.2f}%")

st.markdown('<div class="footer">© 2025 Project Nova Analytics</div>', unsafe_allow_html=True)
