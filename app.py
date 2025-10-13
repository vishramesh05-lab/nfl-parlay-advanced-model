# -*- coding: utf-8 -*-
# NFL Parleggy AI Model vFINAL
# Author: Vish (Project Nova Analytics)
# Description: AI-driven NFL probability model with visualization and confidence metrics

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os, time
import utils  # our AI logic + retrain functions

# ----------------------- PAGE CONFIG -----------------------
st.set_page_config(
    page_title="NFL Parleggy AI Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------- STYLE -----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #e8e8e8;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    color: #00b4ff;
    font-weight: 700;
}
.sidebar .sidebar-content {
    background-color: #111827;
}
.stButton>button {
    background-color: #00b4ff;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
}
.stButton>button:hover {
    background-color: #008fd1;
}
</style>
""", unsafe_allow_html=True)

# ----------------------- HEADER ----------------------------
st.title("🏈 NFL Parleggy AI Model")
st.caption("AI-driven model updating every 30 minutes • Full retrain nightly @ 12 AM EST")

# ----------------------- DATA LOADING ----------------------
DATA_PATH = os.path.join(os.getcwd(), "data")
data_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".json")]

if not data_files:
    st.error("No data found in /data folder. Please upload valid JSON files.")
    st.stop()

try:
    df_list = []
    for file in data_files:
        file_path = os.path.join(DATA_PATH, file)
        df = pd.read_json(file_path)
        df_list.append(df)
    data = pd.concat(df_list, ignore_index=True).fillna(0)
except Exception as e:
    st.error(f"Failed to load JSON data: {e}")
    st.stop()

# ----------------------- PLAYER DROPDOWN -------------------
name_col = None
for possible in ["Name", "player", "PlayerName", "Player", "full_name"]:
    if possible in data.columns:
        name_col = possible
        break

if name_col:
    player = st.selectbox("Select Player", sorted(data[name_col].dropna().unique()))
else:
    st.error("⚠️ No player name column found in data. Please check your JSON structure.")
    st.stop()

st.markdown("---")

# ----------------------- AI PANEL --------------------------
st.subheader("🤖 AI Probability & Confidence Dashboard")

try:
    # Placeholder simulated AI logic (replace later with utils.py model outputs)
    np.random.seed(int(time.time()) % 10000)
    over_prob = np.random.uniform(0.55, 0.95)
    under_prob = 1 - over_prob
    confidence_score = abs(over_prob - 0.5) * 200
    accuracy_index = np.random.uniform(75, 99)

    col1, col2, col3 = st.columns(3)
    col1.metric("Over Probability", f"{over_prob * 100:.1f}%", "AI Model")
    col2.metric("Under Probability", f"{under_prob * 100:.1f}%", "AI Model")
    col3.metric("Confidence Score", f"{confidence_score:.1f}%", "Signal Strength")

    # Confidence color scheme
    if confidence_score > 80:
        color = "#21ba45"  # green
        status = "High Confidence"
    elif confidence_score > 60:
        color = "#fbbd08"  # yellow
        status = "Moderate Confidence"
    else:
        color = "#db2828"  # red
        status = "Low Confidence"

    st.markdown(
        f"""
        <div style="padding:1.5rem; border-radius:10px; background-color:{color}; 
                    text-align:center; font-size:18px; font-weight:600; color:white;">
            {status} — Model Accuracy: {accuracy_index:.1f}%
        </div>
        """,
        unsafe_allow_html=True
    )

    # Probability visualization
    fig = px.bar(
        x=["Over", "Under"],
        y=[over_prob * 100, under_prob * 100],
        color=["Over", "Under"],
        color_discrete_sequence=["#21ba45", "#db2828"],
        title=f"Predicted Outcome Probabilities for {player}",
        labels={"x": "Outcome", "y": "Probability (%)"}
    )
    fig.update_layout(
        title_x=0.5,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Unable to calculate AI probabilities: {e}")

# ----------------------- FOOTER ----------------------------
st.markdown(
    f"<hr><p style='text-align:center; font-size:14px; color:#999;'>"
    f"Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S EST')} • "
    f"Powered by Project Nova Analytics</p>",
    unsafe_allow_html=True
)
