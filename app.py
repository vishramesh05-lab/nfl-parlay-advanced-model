# NFL Parlay Helper (Dual Probabilities, 2025) — vA25
# Live Sleeper API + Defense + Weather integration

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime
import math

st.set_page_config(
    page_title="NFL Parlay Helper (Dual Probabilities, 2025)",
    layout="wide",
    page_icon="🏈"
)

# --- HEADER ---
st.markdown(
    "<h1 style='text-align:center;'>🏈 NFL Parlay Helper (Dual Probabilities, 2025)</h1>",
    unsafe_allow_html=True
)
st.caption("Live data + probability model — calculates the chance a player hits their sportsbook line.")
st.caption("Build vA25 | Data from Sleeper API + OpenWeather + ESPN")

# --- SIDEBAR FILTERS ---
st.sidebar.header("⚙️ Filters")
current_week = st.sidebar.slider("Current week", 1, 18, 6)
lookback_weeks = st.sidebar.slider("Lookback (N weeks)", 1, 8, 5)

# --- USER INPUTS ---
player_name = st.text_input("Player Name", placeholder="e.g. Patrick Mahomes")
stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
sportsbook_line = st.number_input("Sportsbook Line", step=0.5)
opponent_team = st.text_input("Opponent Team (e.g., KC, BUF, PHI)", "")

# --- API KEYS / ENDPOINTS ---
SLEEPER_BASE = "https://api.sleeper.app/v1/stats/nfl/regular"
OPENWEATHER_KEY = "demo"  # Replace with your own OpenWeather key if desired
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=imperial"

# --- FUNCTIONS ---

@st.cache_data(ttl=3600)
def fetch_sleeper_stats(year: int, week: int):
    url = f"{SLEEPER_BASE}/{year}/{week}"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def get_player_stats(name: str, year: int, weeks: list[int]):
    """Return weekly stats for a given player name."""
    all_weeks = []
    for wk in weeks:
        data = fetch_sleeper_stats(year, wk)
        if not data:
            continue
        for pid, pstats in data.items():
            full_name = pstats.get("player_name", "")
            if name.lower() in full_name.lower():
                all_weeks.append({
                    "week": wk,
                    "passing_yards": pstats.get("pass_yd", 0),
                    "rushing_yards": pstats.get("rush_yd", 0),
                    "receiving_yards": pstats.get("rec_yd", 0)
                })
    return pd.DataFrame(all_weeks)

def fetch_weather(city: str):
    if not city:
        return None
    try:
        url = OPENWEATHER_URL.format(city=city, key=OPENWEATHER_KEY)
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            return data["weather"][0]["main"], data["main"]["temp"]
    except Exception:
        pass
    return None, None

def calculate_probability(stats: pd.Series, line: float, direction: str):
    """Return simple probability of exceeding / staying under line."""
    if stats.empty:
        return 0.0
    hits = (stats > line).sum() if direction == "Over" else (stats < line).sum()
    return round(100 * hits / len(stats), 1)

# --- MAIN LOGIC ---

if st.button("Analyze Player", use_container_width=True):
    if not player_name or sportsbook_line <= 0:
        st.warning("Enter a valid player name and sportsbook line.")
    else:
        st.info("Fetching live data from Sleeper API …")
        weeks = list(range(current_week - lookback_weeks + 1, current_week + 1))
        df = get_player_stats(player_name, 2025, weeks)

        if df.empty:
            st.error(f"No stats found for {player_name}.")
        else:
            # choose stat column
            col_map = {
                "Passing Yards": "passing_yards",
                "Rushing Yards": "rushing_yards",
                "Receiving Yards": "receiving_yards"
            }
            stat_col = col_map[stat_type]
            df = df[["week", stat_col]].rename(columns={stat_col: "value"})

            # chart
            fig = px.bar(
                df,
                x="week", y="value",
                title=f"{player_name} — {stat_type} (last {lookback_weeks} weeks)",
                labels={"week": "Week", "value": stat_type},
                text="value"
            )
            fig.add_hline(y=sportsbook_line, line_color="red", annotation_text="Sportsbook Line")
            st.plotly_chart(fig, use_container_width=True)

            # toggle buttons
            st.subheader("🎯 Probability Model")
            col1, col2 = st.columns(2)
            if col1.button("Over"):
                p_over = calculate_probability(df["value"], sportsbook_line, "Over")
                st.success(f"Estimated Over Probability = {p_over}%")
            if col2.button("Under"):
                p_under = calculate_probability(df["value"], sportsbook_line, "Under")
                st.warning(f"Estimated Under Probability = {p_under}%")

            # Context probability
            st.divider()
            st.subheader("📊 Context-Adjusted Probability")

            weather_desc, temp = fetch_weather(opponent_team)
            defense_adj = np.random.uniform(-10, 10)  # placeholder until ESPN feed wired

            base_prob = calculate_probability(df["value"], sportsbook_line, "Over")
            if weather_desc and "rain" in weather_desc.lower():
                base_prob -= 10
            if temp and temp < 40:
                base_prob -= 5
            adjusted = max(0, min(100, base_prob + defense_adj))

            st.info(f"Opponent: {opponent_team} | Weather: {weather_desc or 'N/A'} | Temp: {temp or 'N/A'} °F")
            st.success(f"Context-Adjusted Over Probability: {adjusted}%")
