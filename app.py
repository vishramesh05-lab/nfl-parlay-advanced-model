# 🏈 NFL Parlay Helper (Dual Probabilities, 2025)
# Data: Kaggle Big Data Bowl 2025 + OpenWeather Adjustments
# Build vA56 | Vishvin Ramesh

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import zipfile
import chardet
from datetime import datetime
import io

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
st.caption("Build vA56 | by Vishvin Ramesh")

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
OPENWEATHER_KEY = st.secrets.get("OPENWEATHER_KEY", "demo")
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=imperial"

# ----------------------------------------------------
# UNIVERSAL SAFE READER
# ----------------------------------------------------
def safe_read_file(file_like):
    """Reads CSV, ZIP, or Excel with universal encoding fallback."""
    try:
        return pd.read_csv(file_like, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        st.error("❌ File is empty.")
        return pd.DataFrame()
    except pd.errors.ParserError:
        file_like.seek(0)
        if zipfile.is_zipfile(file_like):
            with zipfile.ZipFile(file_like, "r") as z:
                csv_files = [n for n in z.namelist() if n.endswith(".csv")]
                if not csv_files:
                    st.error("⚠️ ZIP found but no CSV inside.")
                    return pd.DataFrame()
                csv_name = csv_files[0]
                st.info(f"📦 Extracting {csv_name} from ZIP...")
                with z.open(csv_name) as csv_file:
                    return pd.read_csv(csv_file, encoding="utf-8-sig")
        file_like.seek(0)
        try:
            return pd.read_excel(file_like)
        except Exception:
            st.error("⚠️ Could not parse as CSV, ZIP, or Excel.")
            return pd.DataFrame()
    except UnicodeDecodeError:
        file_like.seek(0)
        raw = file_like.read(4096)
        enc = chardet.detect(raw).get("encoding", "utf-8")
        file_like.seek(0)
        return pd.read_csv(file_like, encoding=enc)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_data(uploaded_file=None):
    """Loads dataset from upload or default file."""
    try:
        if uploaded_file is not None:
            df = safe_read_file(uploaded_file)
        else:
            with open(DATA_FILE, "rb") as f:
                df = safe_read_file(f)
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset not found. Please upload below.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

# ----------------------------------------------------
# WEATHER
# ----------------------------------------------------
def fetch_weather(city):
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

# ----------------------------------------------------
# PROBABILITY
# ----------------------------------------------------
def calc_prob(series, line, direction):
    if len(series) == 0:
        return 0.0
    hits = (series > line).sum() if direction == "Over" else (series < line).sum()
    return round(100 * hits / len(series), 1)

# ----------------------------------------------------
# UPLOAD & RELOAD
# ----------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Kaggle 2025 CSV or ZIP file", type=["csv", "zip", "xlsx"])
if st.button("🔁 Reload / Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.success("✅ Cache cleared. Data will reload automatically!")

# ----------------------------------------------------
# MAIN LOGIC
# ----------------------------------------------------
if uploaded_file:
    df = load_data(uploaded_file)
    if not df.empty:
        st.success("✅ Data loaded successfully!")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.stop()

if st.button("Analyze Player", use_container_width=True):
    df = load_data(uploaded_file)
    if df.empty:
        st.warning("Please upload a dataset first.")
        st.stop()

    if "player" not in df.columns:
        st.error("❌ No 'player' column found in dataset.")
        st.stop()

    matches = df[df["player"].str.contains(player_name, case=False, na=False)]
    if matches.empty:
        st.warning(f"No data found for '{player_name}'. Try partial name (e.g., 'Mahomes').")
        st.stop()

    stat_col_map = {
        "Passing Yards": "passing_yards",
        "Rushing Yards": "rushing_yards",
        "Receiving Yards": "receiving_yards"
    }
    stat_col = stat_col_map[stat_type]
    if stat_col not in df.columns:
        st.error(f"Column '{stat_col}' not found in dataset.")
        st.stop()

    player_df = matches[["week", stat_col]].sort_values("week").tail(lookback_weeks)
    player_df.rename(columns={stat_col: "value"}, inplace=True)

    # Plot
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

    # Weather Adjustment
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

st.caption("Data: Kaggle Big Data Bowl 2025 | Build vA56 | by Vishvin Ramesh")
