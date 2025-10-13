# ======================================================
# NFL Parlay Helper v62 (SportsData.io + OpenWeather)
# Author: Vishvin Ramesh
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="NFL Parlay Helper (2025)", page_icon="🏈", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>🏈 NFL Parlay Helper — Live 2025 Edition</h1>",
    unsafe_allow_html=True,
)
st.caption("Live player stats powered by SportsData.io + Weather adjustments via OpenWeather")
st.caption("Build v62 • by Vishvin Ramesh")

# -------------------------------
# API KEYS (Secrets)
# -------------------------------
SPORTSDATA_KEY = st.secrets.get("SPORTSDATA_KEY")
WEATHER_KEY = st.secrets.get("OPENWEATHER_KEY")

if not SPORTSDATA_KEY:
    st.error("⚠️ Please add your SportsData.io API key under Streamlit → Secrets → SPORTSDATA_KEY")
    st.stop()

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("⚙️ Filters")
week = st.sidebar.slider("Current Week", 1, 18, 6)
lookback = st.sidebar.slider("Lookback Weeks", 1, 10, 5)
debug = st.sidebar.checkbox("Enable Debug Mode")

# -------------------------------
# INPUTS
# -------------------------------
player_name = st.text_input("Player Name", placeholder="e.g., Josh Allen")
stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
sportsbook_line = st.number_input("Sportsbook Line", value=100.0, step=1.0)
opponent_team = st.text_input("Opponent Team (e.g., KC, BUF, PHI)", "")
weather_city = st.text_input("Weather City (optional, e.g., Detroit)", "")

st.divider()

# -------------------------------
# FETCH FUNCTIONS
# -------------------------------

@st.cache_data(ttl=900)
def fetch_week_stats(week_num):
    """Fetch weekly player stats safely, skipping bad weeks"""
    url = f"https://api.sportsdata.io/v3/nfl/stats/json/PlayerGameStatsByWeek/2025REG/{week_num}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_KEY}
    r = requests.get(url, headers=headers, timeout=10)

    if r.status_code != 200:
        raise Exception(f"SportsData.io Error {r.status_code}")

    data = r.json()
    if not isinstance(data, list) or len(data) == 0:
        raise Exception(f"No data for week {week_num}")

    df = pd.DataFrame(data)

    # Ensure safe column defaults
    for col in ["FirstName", "LastName", "Name", "Team", "PassingYards", "RushingYards", "ReceivingYards"]:
        if col not in df.columns:
            df[col] = np.nan

    # Normalized full name
    df["full_name"] = (
        df["FirstName"].fillna("") + " " + df["LastName"].fillna("")
    ).str.strip().str.lower()

    # fallback if First/Last empty
    df.loc[df["full_name"].eq(""), "full_name"] = df["Name"].astype(str).str.lower()

    return df


def fetch_weather(city):
    """Optional weather lookup"""
    if not city or not WEATHER_KEY:
        return None
    try:
        r = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=imperial",
            timeout=5,
        )
        if r.status_code == 200:
            return r.json()["main"]["temp"]
    except Exception:
        return None
    return None

# -------------------------------
# MAIN ANALYSIS
# -------------------------------

if st.button("📊 Analyze Player"):
    if not player_name.strip():
        st.warning("Enter a player name first.")
        st.stop()

    player_key = player_name.lower().strip()
    st.info(f"Fetching last {lookback} weeks for {player_name}…")

    valid_weeks = []
    all_data = []

    for wk in range(max(1, week - lookback + 1), week + 1):
        try:
            df = fetch_week_stats(wk)
            df["Week"] = wk
            all_data.append(df)
            valid_weeks.append(wk)
        except Exception as e:
            st.warning(f"⚠️ Skipping week {wk} ({e})")

    if not all_data:
        st.error("No valid data found. Check API key or try another week.")
        st.stop()

    data = pd.concat(all_data, ignore_index=True)
    data = data[data["full_name"].str.contains(player_key, na=False)]

    if data.empty:
        st.error(f"No data found for {player_name}. Try another name format or check debug mode.")
        if debug:
            st.dataframe(all_data[-1].head(15))
        st.stop()

    # Determine column to analyze
    col_map = {
        "Passing Yards": "PassingYards",
        "Rushing Yards": "RushingYards",
        "Receiving Yards": "ReceivingYards",
    }
    col = col_map[stat_type]

    if col not in data.columns:
        st.error(f"Stat column '{col}' missing for {player_name}.")
        st.stop()

    stat_values = data[col].fillna(0).astype(float)
    if len(stat_values) == 0:
        st.error("No valid stat entries found for this player.")
        st.stop()

    mean_val = np.mean(stat_values)
    median_val = np.median(stat_values)
    prob_over = np.mean(stat_values > sportsbook_line) * 100
    prob_under = 100 - prob_over

    # Weather effect
    temp = fetch_weather(weather_city)
    if temp:
        if temp < 40:
            prob_over -= 5
        elif temp > 85:
            prob_over -= 3
        prob_over = np.clip(prob_over, 0, 100)
        prob_under = 100 - prob_over

    # -------------------------------
    # DISPLAY
    # -------------------------------
    st.success(f"✅ {player_name} ({stat_type}) — Weeks {min(valid_weeks)}–{max(valid_weeks)}")

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Average", f"{mean_val:.1f} yards")
    col2.metric("📈 Over Prob.", f"{prob_over:.1f}%")
    col3.metric("📉 Under Prob.", f"{prob_under:.1f}%")

    if temp:
        st.caption(f"🌡️ Weather adjustment applied ({temp:.0f}°F in {weather_city})")

    # Distribution chart
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(range(len(stat_values))),
            y=stat_values,
            marker_color="#00cc96",
            name=stat_type,
        )
    )
    fig.update_layout(
        title=f"{player_name} — {stat_type} History (Weeks {min(valid_weeks)}–{max(valid_weeks)})",
        yaxis_title="Yards",
        xaxis_title="Week Index",
        template="plotly_dark",
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    if debug:
        st.subheader("Raw Data (Debug Mode)")
        st.dataframe(data[["Week", "Team", col]])
