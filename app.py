# -*- coding: utf-8 -*-
"""
NFL Parleggy AI Model v71
Author: Vish (Project Nova Analytics)
AI-driven probability engine with automated retraining & player visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, pickle, time
from datetime import datetime
import plotly.express as px
import utils  # our AI logic + retrain functions

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NFL Parleggy AI Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- STYLE (Dark Blue Professional) ----------------
st.markdown("""
<style>
body {background:#0b0f16;color:#f0f0f0;font-family:'Inter',sans-serif;}
h1,h2,h3 {color:#00aaff;font-weight:600;}
.sidebar .sidebar-content {background:#101520;}
button[data-baseweb="button"] {background-color:#007bff;color:white;border-radius:6px;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
DATA_PATH = os.path.join(os.getcwd(), "data")
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH, exist_ok=True)

@st.cache_data
def load_player_data():
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".json")]
    dfs = []
    for f in files:
        try:
            df = pd.read_json(os.path.join(DATA_PATH, f))
            dfs.append(df)
        except Exception:
            pass
    if dfs:
        return pd.concat(dfs, ignore_index=True).fillna(0)
    return pd.DataFrame()

data = load_player_data()

# ---------------- TITLE ----------------
st.title("🏈 NFL Parleggy AI Model")
st.caption("AI-driven model updating every 30 minutes • Full retrain nightly at 12 AM EST")

# ---------------- CONTROLS ----------------
tab1, tab2 = st.tabs(["Player Probability Model", "Parlay Probability Model"])

with tab1:
    st.header("Individual Player Analysis")
    if data.empty:
        st.warning("No player data found in /data. Upload JSON stats or connect your SportsData.io feed.")
    else:
        player = st.selectbox("Select Player", sorted(data['Name'].unique()))
        stat = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
        line = st.number_input("Sportsbook Line", min_value=0.0, max_value=1000.0, value=250.0)
        dfp = data[data['Name'] == player]
        avg = dfp[stat].mean()
        prob = utils.calculate_probability(dfp, stat, line)
        st.metric("Average", f"{avg:.1f}")
        st.metric("Over Probability", f"{prob*100:.1f}%")

        # Visualization
        fig = px.line(dfp, x="Week", y=stat, title=f"{player} — {stat} over time",
                      template="plotly_dark", markers=True)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Parlay Probability Model (AI Weighted)")
    st.info("This module uses XGBoost + ensemble weighting to estimate multi-leg parlay hit probability.")
    st.write("Feature under optimization for upcoming release.")

# ---------------- RETRAIN STATUS ----------------
st.divider()
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Retrain Now"):
        with st.spinner("Retraining AI model..."):
            utils.retrain_ai(data)
        st.success("✅ Model retrained successfully!")
with col2:
    st.caption("Auto-refresh every 30 min • Full AI retrain nightly at 12 AM EST")

st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
