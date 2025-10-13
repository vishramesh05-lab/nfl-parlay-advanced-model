# NFL Parlay Helper (Dual Probabilities, 2025) — vA46
# Live Kaggle (NFL Big Data Bowl 2025) + Over/Under probability model
# Author: Vishvin Ramesh | 2025

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
import os
import subprocess
import requests

# ---------------------------------------------------------------
# Streamlit Page Configuration
st.set_page_config(
    page_title="NFL Parlay Helper (2025)",
    layout="wide",
    page_icon="🏈"
)

st.markdown("<h1 style='text-align:center;'>🏈 NFL Parlay Helper (Dual Probabilities, 2025)</h1>",
            unsafe_allow_html=True)
st.caption("Live data + probability model — Kaggle NFL Big Data Bowl 2025 + OpenWeather")
st.caption("Build vA46 | by Vishvin Ramesh")

# ---------------------------------------------------------------
# 🔐 Load Kaggle Credentials (from Streamlit Secrets)
if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
else:
    st.error("⚠️ Kaggle credentials not found. Add them under 'App Settings → Secrets' in Streamlit Cloud.")
    st.stop()

# ---------------------------------------------------------------
# Sidebar Filters
st.sidebar.header("⚙️ Filters")
current_week = st.sidebar.slider("Current week", 1, 18, 6)
lookback_weeks = st.sidebar.slider("Lookback (weeks)", 1, 8, 5)

# Inputs
player_name = st.text_input("Player Name", placeholder="e.g. Patrick Mahomes")
stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
sportsbook_line = st.number_input("Sportsbook Line", step=0.5, value=250.0)
opponent_team = st.text_input("Opponent Team (e.g., KC, BUF, PHI)", "")
weather_city = st.text_input("Weather City (optional, e.g., Detroit)", "")

OPENWEATHER_KEY = "demo"  # Replace with your key if you want live temps
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=imperial"

# ---------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_kaggle_data():
    """
    Fetches and loads the NFL Big Data Bowl 2025 dataset via Kaggle API.
    """
    dataset = "nfl-big-data-bowl-2025"
    csv_path = "player_weekly_stats.csv"

    try:
        if not os.path.exists(csv_path):
            st.info("📦 Downloading NFL 2025 data from Kaggle … please wait ⏳")
            subprocess.run(["kaggle", "competitions", "download", "-c", dataset, "-p", "."], check=True)
            subprocess.run(["unzip", "-o", "*.zip"], check=False)
    except Exception as e:
        raise RuntimeError(f"Kaggle data download failed: {e}")

    try:
        df = pd.read_csv(csv_path)
        st.success("✅ NFL 2025 data loaded successfully.")
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading CSV file: {e}")

# ---------------------------------------------------------------
def fetch_weather(city):
    """Fetch current weather and temperature."""
    if not city.strip():
        return None, None
    try:
        r = requests.get(OPENWEATHER_URL.format(city=city, key=OPENWEATHER_KEY), timeout=10)
        if r.status_code == 200:
            j = r.json()
            return j["weather"][0]["main"], j["main"]["temp"]
    except Exception:
        pass
    return None, None


def calc_prob(series, line, direction):
    """Calculate simple over/under probability."""
    if len(series) == 0:
        return 0
    hits = (series > line).sum() if direction == "Over" else (series < line).sum()
    return round(100 * hits / len(series), 1)

# ---------------------------------------------------------------
# Reload data manually
if st.button("🔄 Reload Latest Kaggle Data", use_container_width=True):
    st.cache_data.clear()
    st.success("Cache cleared — new data will load on next analysis.")

# ---------------------------------------------------------------
# Analyze player logic
if st.button("Analyze Player", use_container_width=True):
    st.info("Fetching 2025 NFL player stats from Kaggle …")
    try:
        df = fetch_kaggle_data()
    except Exception as e:
        st.error(str(e))
        st.stop()

    df.columns = [c.lower().strip() for c in df.columns]

    # Ensure consistent stat column mapping
    stat_map = {
        "Passing Yards": ["pass_yards", "passingyards", "passing_yards"],
        "Rushing Yards": ["rush_yards", "rushingyards", "rushing_yards"],
        "Receiving Yards": ["rec_yards", "receivingyards", "receiving_yards"]
    }

    found_col = None
    for alias in stat_map[stat_type]:
        if alias in df.columns:
            found_col = alias
            break

    if not found_col:
        st.error(f"Could not find column for {stat_type}. Please verify Kaggle data file.")
        st.stop()

    if "player" not in df.columns:
        st.error("Dataset missing 'player' column.")
        st.stop()

    # Search player
    match = df[df["player"].str.lower().str.contains(player_name.lower().strip(), na=False)]
    if match.empty:
        st.error(f"No player found for '{player_name}'. Try partial name (e.g., 'Mahomes').")
        st.stop()

    # Recent weeks
    recent_games = match.tail(lookback_weeks)
    values = recent_games[found_col].astype(float).dropna()

    st.success(f"✅ Found {player_name} — Analyzing last {len(values)} games.")

    # Chart
    fig = px.bar(
        recent_games, x="week" if "week" in recent_games.columns else recent_games.index,
        y=found_col, text=found_col,
        title=f"{player_name} — {stat_type} (Last {lookback_weeks} Games)"
    )
    fig.add_hline(y=sportsbook_line, line_color="red", annotation_text="Sportsbook Line")
    st.plotly_chart(fig, use_container_width=True)

    # Baseline probabilities
    st.subheader("🎯 Baseline Probability")
    col1, col2 = st.columns(2)
    if col1.button("Over"):
        st.success(f"Over Probability: {calc_prob(values, sportsbook_line, 'Over')}%")
    if col2.button("Under"):
        st.warning(f"Under Probability: {calc_prob(values, sportsbook_line, 'Under')}%")

    # Context adjustments
    st.divider()
    st.subheader("📊 Context-Adjusted Probability")
    weather, temp = fetch_weather(weather_city)
    adj = calc_prob(values, sportsbook_line, "Over")
    if weather and "rain" in weather.lower():
        adj -= 8
    if temp and temp < 40:
        adj -= 5
    adj = max(0, min(100, adj))

    st.info(f"Opponent: {opponent_team or 'N/A'} | Weather: {weather or 'N/A'} | Temp: {('%.0f' % temp) if temp else 'N/A'} °F")
    st.success(f"Adjusted Over Probability: {adj}%")

    # Summary table
    st.divider()
    st.subheader("📈 Player Summary (Last Games)")
    summary = {
        "Games Analyzed": len(values),
        "Average Yards": round(values.mean(), 1),
        "Max Yards": int(values.max()) if len(values) > 0 else 0,
        "Min Yards": int(values.min()) if len(values) > 0 else 0
    }
    st.table(pd.DataFrame([summary]))
    st.caption(f"Last refresh: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------------------------------------------
st.markdown("---")
st.caption("Data: Kaggle NFL Big Data Bowl 2025 • OpenWeather • Build vA46 (2025)")
