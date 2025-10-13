"""
NFL Parleggy AI Model v1.1
Author: Vish (Project Nova Analytics)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, pickle, time, traceback
from datetime import datetime
import plotly.express as px
import utils  # helper functions

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="NFL Parleggy AI Model", layout="wide", initial_sidebar_state="expanded")

# =============================
# STYLE (Dark Professional)
# =============================
st.markdown("""
<style>
body {background:#0F1117;color:#e8e8e8;font-family:'Inter',sans-serif;}
h1,h2,h3 {color:#00aeff;font-weight:600;}
.stTabs [data-baseweb="tab-list"] {gap:16px;}
.stTabs [data-baseweb="tab"] {color:#aaa;padding:6px 20px;border:none;}
.stTabs [data-baseweb="tab"][aria-selected="true"] {color:#00aeff;border-bottom:2px solid #00aeff;}
.card {background:#181B22;padding:1.25rem;border-radius:12px;box-shadow:0 2px 5px rgba(0,0,0,0.4);}
</style>
""", unsafe_allow_html=True)

# =============================
# TITLE
# =============================
st.title("🏈 NFL Parleggy AI Model")
st.caption("AI-driven probability engine • Auto-refresh every 30 min and re-trains nightly 12 AM EST")

# =============================
# AUTO RETRAIN CHECK
# =============================
try:
    retrained = utils.maybe_retrain()
    if retrained:
        st.success("♻️ Model retrained successfully and updated with latest data!")
    else:
        st.info(f"✅ AI Engine ready. Last retrained: {utils.get_last_retrain_time()}")
except Exception as e:
    st.error(f"⚠️ Auto-retrain check failed:\n{traceback.format_exc()}")

# =============================
# APP TABS
# =============================
tab1, tab2 = st.tabs(["Player Probability Model", "Parlay Probability Model"])

# =============================
# TAB 1 – INDIVIDUAL PLAYER MODEL
# =============================
with tab1:
    st.subheader("Individual Player Analysis")
    player = st.selectbox("Select Player", utils.get_player_dropdown())
    stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
    sportsbook_line = st.number_input("Sportsbook Line", min_value=0.0, step=0.5, value=250.0)
    opponent = st.text_input("Opponent Team (e.g., KC, BUF, PHI)", "")
    city = st.text_input("Weather City (optional, e.g., Detroit)", "")

    if st.button("Analyze Player"):
        with st.spinner(f"Analyzing {player} performance..."):
            try:
                df = utils.fetch_player_data(player, stat_type)
                if df is None or df.empty:
                    st.warning("⚠️ Unable to fetch player data. Using fallback simulation.")
                    avg, over, under = utils.simulate_fallback(sportsbook_line)
                else:
                    avg, over, under = utils.calculate_probabilities(df, sportsbook_line)
                st.metric("Average", f"{avg:.1f}")
                st.metric("Over Probability", f"{over:.1f}%")
                st.metric("Under Probability", f"{under:.1f}%")
            except Exception as e:
                st.error(f"Error analyzing player: {e}")

# =============================
# TAB 2 – PARLAY MODEL
# =============================
with tab2:
    st.subheader("Multi-Leg Parlay Probability")
    st.caption("Combine multiple player legs to compute overall parlay hit probability.")

    legs = st.number_input("Number of Legs", min_value=2, max_value=10, value=3, step=1)
    parlay_probs = []
    for i in range(int(legs)):
        st.write(f"### Leg {i+1}")
        p = st.number_input(f"Enter Probability (%) for Leg {i+1}", min_value=0.0, max_value=100.0, step=0.1)
        parlay_probs.append(p / 100)

    if st.button("Calculate Parlay Probability"):
        combined = utils.calculate_parlay_probability(parlay_probs)
        st.success(f"🏆 Parlay Hit Probability: {combined*100:.2f}%")

# =============================
# FOOTER
# =============================
st.markdown(
    "<p style='text-align:center;color:#777;margin-top:3rem;'>© 2025 Project Nova Analytics | Auto-updates every 30 min | High-accuracy AI model</p>",
    unsafe_allow_html=True,
)
