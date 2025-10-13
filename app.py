# 🏈 NFL Parlay Helper (Dual Probabilities, 2025)
# Data: Kaggle Big Data Bowl 2025 + Optional OpenWeather Adjustments
# Build vA53 | Vishvin Ramesh

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import chardet
from datetime import datetime

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="NFL Parlay Helper (2025 - Kaggle Edition)",
    layout="wide",
    page_icon="🏈"
)
st.markdown("<h1 style='text-align:center;'>🏈 NFL Parlay Helper (2025 - Kaggle Edition)</h1>", unsafe_allow_html=True)
st.caption("Data: Kaggle Big Data Bowl 2025 (Weekly Updates) + OpenWeather Adjustments")
st.caption("Build vA53 | by Vishvin Ramesh")

# ----------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------
st.sidebar.header("⚙️ Filters")
current_week = st.sidebar.slider("Current week", 1, 18, 6)
lookback_weeks = st.sidebar.slider("Lookback (weeks)", 1, 8, 5)

# ----------------------------------------------------
# INPUT FIELDS
# ----------------------------------------------------
player_name = st.text_input("Player Name", placeholder="e.g. Patrick Mahomes")
stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
sportsbook_line = st.number_input("Sportsbook Line", step=0.5, format="%.1f")
opponent_team = st.text_input("Opponent Team (e.g., KC, BUF, PHI)")
weather_city = st.text_input("Weather City (optional, e.g., Detroit)")

# ----------------------------------------------------
# CONSTANTS
# ----------------------------------------------------
DATA_FILE = "nfl_2025_player_stats.csv"
OPENWEATHER_KEY = st.secrets.get("OPENWEATHER_KEY", "demo")  # Add in Streamlit Secrets if available
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=imperial"

# ----------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------
def detect_encoding(file_path):
    """Detects encoding for robust CSV reading."""
    with open(file_path, "rb") as f:
        raw_data = f.read(2048)
        result = chardet.detect(raw_data)
        return result["encoding"]

@st.cache_data(ttl=3600)
def load_data(uploaded_file=None):
    """Load Kaggle data either from upload or repo, auto-detect encoding."""
    try:
        if uploaded_file is not None:
            raw_bytes = uploaded_file.read()
            detected = chardet.detect(raw_bytes)
            encoding_used = detected["encoding"] or "utf-8"
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding_used)
        else:
            encoding_used = detect_encoding(DATA_FILE)
            df = pd.read_csv(DATA_FILE, encoding=encoding_used)
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    except FileNotFoundError:
        st.error("⚠️ Could not find the dataset file. Please upload the CSV below.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

def fetch_weather(city):
    """Fetch weather and temperature (optional)."""
    if not city:
        return None, None
    try:
        r = requests.get(OPENWEATHER_URL.format(city=city, key=OPENWEATHER_KEY))
        if r.status_code == 200:
            j = r.json()
            return j["weather"][0]["main"], j["main"]["temp"]
    except Exception:
        pass
    return None, None

def calc_prob(series, line, direction):
    """Calculate over/under probability."""
    if len(series) == 0:
        return 0.0
    hits = (series > line).sum() if direction == "Over" else (series < line).sum()
    return round(100 * hits / len(series), 1)

# ----------------------------------------------------
# DATA UPLOAD / RELOAD
# ----------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload new Kaggle 2025 CSV (optional)", type=["csv"])

if st.button("🔁 Reload / Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.success("✅ Cache cleared. New data will reload automatically!")

# ----------------------------------------------------
# MAIN ANALYSIS
# ----------------------------------------------------
if st.button("Analyze Player", use_container_width=True):
    df = load_data(uploaded_file)
    if df.empty:
        st.stop()

    # Match player by name fragment
    matches = df[df["player"].str.contains(player_name, case=False, na=False)]
    if matches.empty:
        st.warning(f"No data found for '{player_name}'. Try partial name (e.g. 'Mahomes').")
        st.stop()

    # Map stat type to column
    stat_col_map = {
        "Passing Yards": "passing_yards",
        "Rushing Yards": "rushing_yards",
        "Receiving Yards": "receiving_yards"
    }
    stat_col = stat_col_map[stat_type]
    if stat_col not in df.columns:
        st.error(f"Column '{stat_col}' not found in dataset.")
        st.stop()

    # Subset player’s last N weeks
    player_df = matches[["week", stat_col]].sort_values("week").tail(lookback_weeks)
    player_df.rename(columns={stat_col: "value"}, inplace=True)

    # Plot trend
    fig = px.bar(
        player_df, x="week", y="value", text="value",
        title=f"{player_name} — {stat_type} (Last {lookback_weeks} weeks)"
    )
    fig.add_hline(y=sportsbook_line, line_color="red", annotation_text="Sportsbook Line")
    st.plotly_chart(fig, use_container_width=True)

    # Probabilities
    st.subheader("🎯 Probability Model")
    over_prob = calc_prob(player_df["value"], sportsbook_line, "Over")
    under_prob = calc_prob(player_df["value"], sportsbook_line, "Under")

    col1, col2 = st.columns(2)
    col1.success(f"Over Probability: {over_prob}%")
    col2.warning(f"Under Probability: {under_prob}%")

    # Weather adjustment
    weather, temp = fetch_weather(weather_city)
    adj = over_prob
    if weather and "rain" in weather.lower():
        adj -= 8
    if temp and temp < 40:
        adj -= 5
    adj = max(0, min(100, adj))

    st.divider()
    st.info(f"Opponent: {opponent_team or 'N/A'} | Weather: {weather or 'N/A'} | Temp: {temp or 'N/A'}°F")
    st.success(f"Adjusted Over Probability: {adj}%")

st.caption("Data: Kaggle Big Data Bowl 2025 | Build vA53 | by Vishvin Ramesh")
