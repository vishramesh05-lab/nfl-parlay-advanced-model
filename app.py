"""
NFL Parleggy AI Model v3.0
Author: Vish (Project Nova Analytics)
AI-powered player & parlay prediction system with visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
import plotly.express as px
import utils  # our AI logic + retrain functions

# =======================================================
# PAGE CONFIGURATION
# =======================================================
st.set_page_config(
    page_title="NFL Parleggy AI Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================================================
# PAGE STYLE (Dark + Professional)
# =======================================================
st.markdown("""
<style>
body {background-color:#0F1117;color:#e8e8e8;font-family:'Inter',sans-serif;}
h1,h2,h3 {color:#00aeff;font-weight:600;}
.stTabs [data-baseweb="tab-list"] {gap:14px;}
.stTabs [data-baseweb="tab"] {color:#999;padding:8px 20px;border:none;}
.stTabs [data-baseweb="tab"][aria-selected="true"] {color:#00aeff;border-bottom:2px solid #00aeff;}
.block-container {padding-top:2rem;}
.card {background:#181B22;padding:1.25rem;border-radius:12px;box-shadow:0 2px 5px rgba(0,0,0,0.4);}
.footer {text-align:center;color:#777;margin-top:2rem;font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

# =======================================================
# HEADER
# =======================================================
st.title("🏈 NFL Parleggy AI Model")
st.caption("AI-driven probability engine • Auto-updates every 30 min & re-trains nightly 12 AM EST")

# =======================================================
# SIDEBAR CONTROLS
# =======================================================
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🔄 Retrain AI Model"):
        with st.spinner("Retraining model..."):
            try:
                utils.retrain_ai()
                st.success("✅ Model retrained successfully!")
            except Exception as e:
                st.error(f"Retrain failed:\n{e}")

    st.write(f"**Last Retrain:** {utils.get_last_retrain_time()}")
    st.caption("Auto retrains every 30 minutes or nightly at 12 AM EST.")

    st.divider()
    st.subheader("📈 AI Training Performance")

    try:
        log_df = utils.get_training_log()
        if not log_df.empty:
            fig = px.line(
                log_df,
                x="timestamp",
                y="mae",
                markers=True,
                title="Model Mean Absolute Error (MAE) Over Time",
                labels={"mae": "Mean Absolute Error", "timestamp": "Retrain Timestamp"}
            )
            fig.update_layout(
                plot_bgcolor="#0F1117",
                paper_bgcolor="#0F1117",
                font=dict(color="#e8e8e8"),
                title_font=dict(size=16)
            )
            fig.update_traces(line_color="#00aeff")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No training records yet — retrain to log data.")
    except Exception as e:
        st.caption(f"Error loading training log: {e}")

# Background retrain check
try:
    utils.maybe_retrain()
except Exception:
    pass

# =======================================================
# MAIN TABS
# =======================================================
tab1, tab2 = st.tabs(["🎯 Player Probability Model", "📊 Parlay Probability Model"])

# =======================================================
# TAB 1 — PLAYER MODEL
# =======================================================
with tab1:
    st.subheader("Individual Player Analysis")

    col1, col2 = st.columns(2)
    with col1:
        player = st.selectbox("Select Player", utils.get_player_dropdown())
        stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
        sportsbook_line = st.number_input("Sportsbook Line", min_value=0.0, step=0.5, value=250.0)
    with col2:
        opponent = st.text_input("Opponent Team (e.g., KC, BUF, PHI)", "")
        city = st.text_input("Weather City (optional, e.g., Detroit)", "")

    if st.button("🔍 Analyze Player"):
        with st.spinner(f"Analyzing {player} performance..."):
            try:
                df = utils.fetch_player_data(player, stat_type)
                if df is None or df.empty:
                    st.warning("⚠️ No player data found — AI model used for prediction.")
                    avg, over, under = utils.ai_predict(sportsbook_line)
                else:
                    avg, over, under = utils.calculate_probabilities(df, sportsbook_line)

                colA, colB, colC = st.columns(3)
                colA.metric("Predicted Average", f"{avg:.1f}")
                colB.metric("Over Probability", f"{over:.1f}%")
                colC.metric("Under Probability", f"{under:.1f}%")

                st.success(f"✅ AI analysis complete for {player}")
            except Exception as e:
                st.error(f"Error analyzing player:\n{traceback.format_exc()}")

# =======================================================
# TAB 2 — PARLAY MODEL
# =======================================================
with tab2:
    st.subheader("Multi-Leg Parlay Probability Calculator")
    st.caption("Combine multiple legs to compute overall parlay hit probability based on AI outputs.")

    num_legs = st.number_input("Number of Legs", min_value=2, max_value=10, value=3, step=1)
    parlay_probs = []
    for i in range(int(num_legs)):
        p = st.number_input(f"Probability (%) for Leg {i+1}", min_value=0.0, max_value=100.0, step=0.1, key=f"leg{i}")
        parlay_probs.append(p / 100)

    if st.button("🧮 Calculate Parlay Probability"):
        combined = utils.calculate_parlay_probability(parlay_probs)
        st.success(f"🏆 Parlay Hit Probability: **{combined*100:.2f}%**")

# =======================================================
# FOOTER
# =======================================================
st.markdown(
    "<p class='footer'>© 2025 Project Nova Analytics | AI Model v3.0 | Trained with XGBoost</p>",
    unsafe_allow_html=True
)
