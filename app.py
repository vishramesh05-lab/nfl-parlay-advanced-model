# 🏈 NFL Parlay Helper (Dual Probabilities, 2025)
# Build vA60 | Vishvin Ramesh
# Compatible with Kaggle KAFFLW NFL Big Data Bowl format
# Auto-detects player stats, weather integration, weekly updates

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io, zipfile, chardet
from datetime import datetime
import plotly.express as px

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="NFL Parlay Helper (2025)", layout="wide", page_icon="🏈")
st.markdown("<h1 style='text-align:center;'>🏈 NFL Parlay Helper (2025 - KAFFLW Edition)</h1>", unsafe_allow_html=True)
st.caption("Data: Kaggle Big Data Bowl 2025 (KAFFLW) + Optional OpenWeather Adjustments")
st.caption("Build vA60 | by Vishvin Ramesh")

# ----------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------
st.sidebar.header("⚙️ Filters")
current_week = st.sidebar.slider("Current Week", 1, 18, 6)
lookback_weeks = st.sidebar.slider("Lookback (weeks)", 1, 8, 5)

# ----------------------------------------------------
# INPUTS
# ----------------------------------------------------
player_name = st.text_input("Player Name", placeholder="e.g. Patrick Mahomes")
stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
sportsbook_line = st.number_input("Sportsbook Line", step=0.5, format="%.1f", value=100.0)
opponent_team = st.text_input("Opponent Team (e.g., KC, BUF, PHI)")
weather_city = st.text_input("Weather City (optional, e.g., Detroit)")

# ----------------------------------------------------
# CONSTANTS
# ----------------------------------------------------
DATA_FILE = "nfl_2025_player_stats.csv"
OPENWEATHER_KEY = st.secrets.get("OPENWEATHER_KEY", "demo")
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=imperial"

# ----------------------------------------------------
# UNIVERSAL FILE READER
# ----------------------------------------------------
def read_any_file(file_like):
    try:
        if zipfile.is_zipfile(file_like):
            with zipfile.ZipFile(file_like, "r") as z:
                csvs = [f for f in z.namelist() if f.endswith(".csv")]
                if not csvs:
                    st.warning("⚠️ ZIP file has no CSV inside.")
                    return pd.DataFrame()
                with z.open(csvs[0]) as f:
                    raw = f.read()
                    enc = chardet.detect(raw).get("encoding", "utf-8")
                    return pd.read_csv(io.BytesIO(raw), encoding=enc)
        else:
            raw = file_like.read()
            enc = chardet.detect(raw).get("encoding", "utf-8")
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_data(uploaded_file=None):
    try:
        if uploaded_file:
            return read_any_file(uploaded_file)
        else:
            with open(DATA_FILE, "rb") as f:
                return read_any_file(f)
    except Exception as e:
        st.error(f"❌ Could not read dataset: {e}")
        return pd.DataFrame()

# ----------------------------------------------------
# WEATHER FETCH
# ----------------------------------------------------
def fetch_weather(city):
    if not city:
        return None, None
    try:
        r = requests.get(OPENWEATHER_URL.format(city=city, key=OPENWEATHER_KEY))
        if r.status_code == 200:
            j = r.json()
            return j["weather"][0]["main"], j["main"]["temp"]
    except:
        pass
    return None, None

# ----------------------------------------------------
# PROBABILITY MODEL
# ----------------------------------------------------
def calc_prob(series, line, direction):
    if len(series) == 0:
        return 0.0
    hits = (series > line).sum() if direction == "Over" else (series < line).sum()
    return round(100 * hits / len(series), 1)

# ----------------------------------------------------
# FILE UPLOAD SECTION
# ----------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Updated Kaggle 2025 CSV (optional)", type=["csv", "zip"])
if st.button("🔁 Reload / Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.success("✅ Cache cleared — data refreshed!")

df = load_data(uploaded_file)
if df.empty:
    st.warning("⚠️ No data found. Upload a valid KAFFLW or Kaggle CSV.")
    st.stop()

# Normalize column names
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

# ----------------------------------------------------
# MAIN ANALYSIS
# ----------------------------------------------------
if st.button("Analyze Player", use_container_width=True):
    # Identify key columns
    player_col = next((c for c in df.columns if "player" in c), None)
    week_col = next((c for c in df.columns if "week" in c), None)
    pass_col = next((c for c in df.columns if "pass" in c and "yard" in c), None)
    rush_col = next((c for c in df.columns if "rush" in c and "yard" in c), None)
    rec_col = next((c for c in df.columns if "rec" in c and "yard" in c), None)

    if not player_col:
        st.error("❌ No player column found in dataset.")
        st.stop()

    matches = df[df[player_col].str.contains(player_name, case=False, na=False)]
    if matches.empty:
        st.warning(f"No stats found for '{player_name}'. Try shorter name (e.g., 'Mahomes').")
        st.stop()

    stat_map = {
        "Passing Yards": pass_col,
        "Rushing Yards": rush_col,
        "Receiving Yards": rec_col
    }
    stat_col = stat_map[stat_type]
    if not stat_col:
        st.error(f"❌ No data column found for {stat_type}.")
        st.stop()

    player_df = matches[[week_col, stat_col]].tail(lookback_weeks)
    player_df.columns = ["week", "yards"]

    # Chart
    fig = px.bar(player_df, x="week", y="yards", text="yards",
                 title=f"{player_name} — {stat_type} (Last {lookback_weeks} Weeks)")
    fig.add_hline(y=sportsbook_line, line_color="red", annotation_text="Sportsbook Line")
    st.plotly_chart(fig, use_container_width=True)

    # Probabilities
    over_prob = calc_prob(player_df["yards"], sportsbook_line, "Over")
    under_prob = calc_prob(player_df["yards"], sportsbook_line, "Under")

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

st.caption("Data: Kaggle Big Data Bowl 2025 (KAFFLW) | Build vA60 | Vishvin Ramesh")
