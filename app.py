# -*- coding: utf-8 -*-
"""
NFL Parleggy AI Model — Final
Author: Vish (Project Nova Analytics)

This app:
- Loads live JSON data from /data
- Trains a local XGBoost model nightly @ 12 AM EST (and on demand)
- Auto-refreshes data every 30 minutes
- Computes over/under probabilities per player/market using learned mean + residual sigma
- Provides a Parlay Probability tab with a light correlation penalty
"""

import os, json, time, datetime, traceback
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import utils  # local AI + data helpers

# ----------------------------- PAGE CONFIG & STYLE -----------------------------
st.set_page_config(page_title="NFL Parleggy AI Model", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --bg:#0e1117; --card:#161a22; --text:#e8e8e8; --muted:#A0AEC0; --blue:#00b4ff; }
body { background-color: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif; }
h1,h2,h3 { color: var(--blue); font-weight: 700; letter-spacing: .2px; }
.stTabs [data-baseweb="tab-list"] { gap: 16px; }
.stTabs [data-baseweb="tab"] { color: #bbb; padding: 8px 16px; border: none; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: var(--blue); border-bottom: 2px solid var(--blue); }
.card { background: var(--card); border-radius: 12px; padding: 16px; box-shadow: 0 4px 10px rgba(0,0,0,.25); }
.metric { font-size: 28px; font-weight: 800; }
hr { border-color:#222; }
</style>
""", unsafe_allow_html=True)

st.title("NFL Parleggy AI Model")
st.caption("Live JSON ➜ Feature Engine ➜ XGBoost ➜ Probability • Auto-refresh every 30 min • Nightly retrain 12:00 AM EST")

# ----------------------------- AUTO REFRESH / NIGHTLY RETRAIN -----------------
# Auto-refresh UI every 30 minutes to pick up new /data files

REFRESH_SEC = 30 * 60
now_ts = time.time()
last = st.session_state.get("last_refresh_ts", 0.0)
if now_ts - last > REFRESH_SEC:
    st.session_state["last_refresh_ts"] = now_ts
    # Do not force rerun here; we will rely on cache TTL and on-demand retrain.

# Nightly retrain trigger (12:00 AM EST = 04:00 UTC)
try:
    utils.maybe_retrain_all()  # safe; returns quickly if not due
except Exception:
    pass

# ----------------------------- SIDEBAR: CONTROLS & STATUS ---------------------
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Retrain Now (All Markets)", use_container_width=True):
        with st.spinner("Retraining models..."):
            msg = utils.retrain_all_models()
        st.success(msg)

    st.write(f"**Last Retrain:** {utils.get_last_retrain_time()}")
    st.caption("Auto: every 30 min (data cache) • Full retrain nightly @ 12:00 AM EST")

    # Training log chart (if available)
    try:
        log = utils.get_training_log()
        if not log.empty:
            st.subheader("📈 Model MAE over time")
            fig = px.line(
                log, x="timestamp", y="mae", color="market",
                markers=True, labels={"mae":"Mean Abs Error", "timestamp":"UTC Timestamp", "market":"Market"},
                title=None
            )
            fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e8e8e8"), height=260)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No training log yet — will appear after first retrain.")
    except Exception:
        st.caption("Training log unavailable.")

# ----------------------------- LOAD DATA (CACHED) -----------------------------
@st.cache_data(ttl=REFRESH_SEC)
def load_all_jsons():
    return utils.load_merged_json()

df_all = load_all_jsons()
if df_all is None or df_all.empty:
    st.error("No readable JSON data found in /data. Add files like 3.json, 4.json, 5.json, 6.json.")
    st.stop()

# Detect columns robustly
name_col = utils.detect_name_column(df_all)
pos_col = utils.detect_position_column(df_all)
week_col = utils.detect_week_column(df_all)
team_col = utils.detect_team_column(df_all)
opp_col = utils.detect_opp_column(df_all)

# ----------------------------- APP TABS ---------------------------------------
tab_player, tab_parlay = st.tabs(["Player Probability Model", "Parlay Probability Model"])

# ============================= PLAYER TAB =====================================
with tab_player:
    st.subheader("Individual Player Projection & Probability")

    # Filter to skill positions if possible
    df_players = df_all.copy()
    if pos_col:
        skill_mask = df_players[pos_col].astype(str).str.upper().isin(["QB", "RB", "WR", "TE"])
        if skill_mask.any():
            df_players = df_players[skill_mask]

    # Build dropdown
    player_list = sorted(set(map(str, df_players[name_col].dropna().unique()))) if name_col else []
    if not player_list:
        st.error("Could not detect player names in data. Check JSON structure.")
        st.stop()

    colA, colB, colC = st.columns([2, 1.2, 1.2])
    with colA:
        player = st.selectbox("Player", player_list)
    with colB:
        market = st.selectbox(
            "Pick Type / Market",
            ["Passing Yards", "Rushing Yards", "Receiving Yards", "Rushing+Receiving TDs", "Passing TDs"]
        )
    with colC:
        line = st.number_input("Sportsbook Line", min_value=0.0, step=0.5, value=250.0)

    colD, colE = st.columns([1.2, 1])
    with colD:
        opponent = st.text_input("Opponent (e.g., KC, BUF, PHI)", value="")
    with colE:
        lookback = st.slider("Lookback (weeks)", min_value=1, max_value=8, value=4, step=1)

    run = st.button("Analyze Player", type="primary")

    if run:
        with st.spinner(f"Computing {market} probabilities for {player}…"):
            try:
                # Prepare player slice (recent weeks)
                hist = utils.get_player_history(
                    df_all, player_name=player, name_col=name_col, week_col=week_col, lookback_weeks=lookback
                )

                # Train (if model missing) & Predict for this market
                pred = utils.predict_player_market(
                    df_all=df_all,
                    player_name=player,
                    market=market,
                    line=line,
                    opponent=opponent,
                    name_col=name_col,
                    week_col=week_col,
                    team_col=team_col,
                    opp_col=opp_col,
                    lookback_weeks=lookback
                )

                # --- UI Output ---
                top1, top2, top3 = st.columns(3)
                top1.metric("Predicted Mean", f"{pred['pred_mean']:.1f}")
                top2.metric("Over Probability", f"{pred['p_over']*100:.1f}%")
                top3.metric("Under Probability", f"{pred['p_under']*100:.1f}%")

                # Confidence banner
                color, label = utils.confidence_color_label(pred["confidence"])
                st.markdown(
                    f"<div class='card' style='text-align:center;border-left:6px solid {color};'>"
                    f"<div class='metric' style='color:{color};'>{label}</div>"
                    f"<div style='color:#cfd8e3'>Confidence Score: {pred['confidence']:.1f}% • "
                    f"Model MAE: {pred['mae']:.2f} • Samples: {pred['samples']}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # Table of recent weeks
                st.markdown("### Recent Weeks (table)")
                if hist is None or hist.empty:
                    st.info("No recent-week table available for this player.")
                else:
                    st.dataframe(hist, use_container_width=True, hide_index=True)

                # Small bar chart (optional, kept elegant)
                st.markdown("### Market Distribution Preview")
                fig = px.histogram(
                    pred["dist_samples"], x="value", nbins=20,
                    title=f"{player} — simulated distribution vs line ({line})",
                )
                fig.add_vline(x=line, line_color="red")
                fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e8e8e8"))
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error("Analysis failed:")
                st.code(traceback.format_exc())

# ============================= PARLAY TAB =====================================
with tab_parlay:
    st.subheader("Multi-Leg Parlay Probability")

    num_legs = st.number_input("Number of legs", min_value=2, max_value=10, value=3, step=1)
    legs = []

    for i in range(int(num_legs)):
        st.markdown(f"#### Leg {i+1}")
        c1, c2, c3, c4 = st.columns([2, 1.6, 1.2, 1.2])
        with c1:
            pl = st.selectbox(f"Player (Leg {i+1})", player_list, key=f"pl_{i}")
        with c2:
            mk = st.selectbox(
                f"Market (Leg {i+1})",
                ["Passing Yards", "Rushing Yards", "Receiving Yards", "Rushing+Receiving TDs", "Passing TDs"],
                key=f"mk_{i}"
            )
        with c3:
            ln = st.number_input(f"Line (Leg {i+1})", min_value=0.0, step=0.5, value=50.0, key=f"ln_{i}")
        with c4:
            opp = st.text_input(f"Opponent (Leg {i+1})", value="", key=f"opp_{i}")

        legs.append(dict(player=pl, market=mk, line=ln, opp=opp))

    if st.button("Compute Parlay Probability", type="primary"):
        with st.spinner("Computing parlay…"):
            try:
                # Compute per-leg probability via the same model
                results = []
                for lg in legs:
                    res = utils.predict_player_market(
                        df_all=df_all,
                        player_name=lg["player"],
                        market=lg["market"],
                        line=lg["line"],
                        opponent=lg["opp"],
                        name_col=name_col,
                        week_col=week_col,
                        team_col=team_col,
                        opp_col=opp_col,
                        lookback_weeks=4
                    )
                    results.append(res)

                # Combine with light correlation penalty
                p, penalty = utils.combine_parlay_probabilities(results)
                st.success(f"Parlay Hit Probability: {p*100:.2f}% (correlation adj: −{penalty*100:.1f}%)")

                # Show leg table
                out = pd.DataFrame([{
                    "Player": legs[i]["player"],
                    "Market": legs[i]["market"],
                    "Line": legs[i]["line"],
                    "Opponent": legs[i]["opp"],
                    "Over %": f"{results[i]['p_over']*100:.1f}",
                    "Conf %": f"{results[i]['confidence']:.1f}"
                } for i in range(len(legs))])
                st.dataframe(out, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error("Parlay computation failed:")
                st.code(traceback.format_exc())

# ----------------------------- FOOTER -----------------------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#A0AEC0;'>© 2025 Project Nova Analytics • Local AI engine • "
    "Data auto-refresh 30 min • Nightly retrain at 12:00 AM EST</div>",
    unsafe_allow_html=True
)
