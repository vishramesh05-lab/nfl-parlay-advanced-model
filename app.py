# ======================================================
# NFL Parlay Helper v61 (SportsData.io + OpenWeather)
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
st.caption("Real-time player stats via SportsData.io + Weather Adjustments via OpenWeather (Free Tier)")
st.caption("Build v61 • by Vishvin Ramesh")

# -------------------------------
# KEYS
# -------------------------------
SPORTSDATA_KEY = st.secrets.get("SPORTSDATA_KEY")
WEATHER_KEY = st.secrets.get("OPENWEATHER_KEY")

if not SPORTSDATA_KEY:
    st.error("⚠️ Add SPORTSDATA_KEY to Streamlit Secrets before continuing.")
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
    """Fetch weekly player stats"""
    url = f"https://api.sportsdata.io/v3/nfl/stats/json/PlayerGameStatsByWeek/2025REG/{week_num}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        raise Exception(f"SportsData.io Error {r.status_code}: {r.text}")
    df = pd.DataFrame(r.json())
    df["full_name"] = (
        df["FirstName"].fillna("") + " " + df["LastName"].fillna("")
    ).str.lower()
    return df


def fetch_weather(city):
    if not city or not WEATHER_KEY:
        return None
    try:
        r = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=imperial"
        )
        if r.status_code == 200:
            return r.json()["main"]["temp"]
    except:
        return None
    return None


# -------------------------------
# ANALYSIS LOGIC
# -------------------------------
if st.button("📊 Analyze Player"):
    if not player_name.strip():
        st.warning("Enter a player name first.")
        st.stop()

    player_key = player_name.lower().strip()

    st.info(f"Fetching last {lookback} weeks for {player_name}...")

    all_data = []
    for wk in range(max(1, week - lookback + 1), week + 1):
        try:
            df = fetch_week_stats(wk)
            df["Week"] = wk
            all_data.append(df)
        except Exception as e:
            st.error(f"Error fetching week {wk}: {e}")
            st.stop()

    data = pd.concat(all_data, ignore_index=True)
    data = data[data["full_name"].str.contains(player_key, na=False)]

    if data.empty:
        st.error(f"No data found for {player_name}. Try enabling Debug Mode.")
        if debug:
            st.dataframe(all_data[-1][["FirstName", "LastName", "Team", "Position"]].head(20))
        st.stop()

    # Select stat
    col_map = {
        "Passing Yards": "PassingYards",
        "Rushing Yards": "RushingYards",
        "Receiving Yards": "ReceivingYards",
    }
    col = col_map[stat_type]
    stat_values = data[col].astype(float)

    mean_val = np.mean(stat_values)
    median_val = np.median(stat_values)
    prob_over = np.mean(stat_values > sportsbook_line) * 100
    prob_under = 100 - prob_over

    # Weather adjustment
    temp = fetch_weather(weather_city)
    if temp:
        if temp < 40:
            prob_over -= 5
        elif temp > 85:
            prob_over -= 3
        prob_over = np.clip(prob_over, 0, 100)
        prob_under = 100 - prob_over

    # -------------------------------
    # DISPLAY RESULTS
    # -------------------------------
    st.success(f"✅ {player_name} ({stat_type}) — Weeks {week - lookback + 1}–{week}")

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Average", f"{mean_val:.1f} yards")
    col2.metric("📈 Probability Over", f"{prob_over:.1f}%")
    col3.metric("📉 Probability Under", f"{prob_under:.1f}%")

    if temp:
        st.caption(f"🌡️ Weather Adjustment: {temp:.0f}°F in {weather_city}")

    # Plot better graph
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=stat_values,
            name=f"{stat_type}",
            boxpoints="all",
            jitter=0.3,
            marker_color="#00cc96",
            line_color="#004d40",
        )
    )
    fig.update_layout(
        title=f"{player_name} — {stat_type} (Weeks {week - lookback + 1}–{week})",
        yaxis_title="Yards",
        template="plotly_dark",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    if debug:
        st.subheader("Raw Data (Debug Mode)")
        st.dataframe(data[["Week", "Team", col]])
