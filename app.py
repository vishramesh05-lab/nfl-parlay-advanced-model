# 🏈 NFL Parlay Helper (Dual Probabilities, 2025)
# Build vA59 | Vishvin Ramesh
# Supports: Kaggle Big Data Bowl 2025, FootballDB, or manual upload
# Formats: CSV, ZIP, JSON, Parquet (any encoding)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests, io, zipfile, chardet
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
st.caption("Data: Kaggle Big Data Bowl 2025 + FootballDB + OpenWeather adjustments")
st.caption("Build vA59 | by Vishvin Ramesh")

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
# UNIVERSAL FILE LOADER
# ----------------------------------------------------
def deep_read_csv(file_like):
    """Reads CSV, JSON, or Parquet (including inside ZIP) with automatic encoding detection."""
    try:
        if zipfile.is_zipfile(file_like):
            with zipfile.ZipFile(file_like, "r") as z:
                candidates = [n for n in z.namelist() if n.endswith((".csv", ".json", ".parquet"))]
                if not candidates:
                    st.warning("⚠️ ZIP found but no CSV/Parquet/JSON inside.")
                    return pd.DataFrame()

                file_name = candidates[0]
                st.info(f"📦 Extracting {file_name} from ZIP…")
                with z.open(file_name) as f:
                    raw = f.read()

                    # --- CSV ---
                    if file_name.endswith(".csv"):
                        enc = chardet.detect(raw).get("encoding", "utf-8")
                        for e in [enc, "utf-8-sig", "latin1", "cp1252"]:
                            try:
                                return pd.read_csv(io.BytesIO(raw), encoding=e)
                            except Exception:
                                continue
                        st.error("❌ Could not decode CSV file.")
                        return pd.DataFrame()

                    # --- JSON ---
                    elif file_name.endswith(".json"):
                        try:
                            return pd.read_json(io.BytesIO(raw))
                        except Exception as e:
                            st.error(f"❌ Failed to read JSON: {e}")
                            return pd.DataFrame()

                    # --- Parquet ---
                    elif file_name.endswith(".parquet"):
                        try:
                            import pyarrow.parquet as pq
                            table = pq.read_table(io.BytesIO(raw))
                            return table.to_pandas()
                        except Exception as e:
                            st.error(f"❌ Failed to read Parquet: {e}")
                            return pd.DataFrame()

        # If not a ZIP, try normal read
        raw = file_like.read()
        enc = chardet.detect(raw).get("encoding", "utf-8")
        for e in [enc, "utf-8-sig", "latin1", "cp1252"]:
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=e)
            except Exception:
                continue
        st.error("❌ Could not decode CSV with any fallback encoding.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_data(uploaded_file=None):
    """Loads dataset from uploaded or default file."""
    try:
        if uploaded_file:
            df = deep_read_csv(uploaded_file)
        else:
            with open(DATA_FILE, "rb") as f:
                df = deep_read_csv(f)
        if df.empty:
            st.warning("⚠️ No data found in file.")
            return pd.DataFrame()
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

# ----------------------------------------------------
# WEATHER FUNCTION
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
# PROBABILITY CALCULATOR
# ----------------------------------------------------
def calc_prob(series, line, direction):
    if len(series) == 0:
        return 0.0
    hits = (series > line).sum() if direction == "Over" else (series < line).sum()
    return round(100 * hits / len(series), 1)

# ----------------------------------------------------
# UPLOAD SECTION
# ----------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Kaggle 2025 CSV, ZIP, JSON, or Parquet", type=["csv", "zip", "json", "parquet"])
if st.button("🔁 Reload / Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.success("✅ Cache cleared. Ready for fresh data!")

# ----------------------------------------------------
# DATA PREVIEW
# ----------------------------------------------------
if uploaded_file:
    df = load_data(uploaded_file)
    if not df.empty:
        st.success("✅ Data loaded successfully!")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.stop()

# ----------------------------------------------------
# MAIN ANALYSIS
# ----------------------------------------------------
if st.button("Analyze Player", use_container_width=True):
    df = load_data(uploaded_file)
    if df.empty:
        st.warning("Please upload a dataset first.")
        st.stop()

    # Smart column detection
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    player_col = next((c for c in df.columns if "player" in c or "name" in c), None)
    week_col = next((c for c in df.columns if "week" in c), None)
    passing_col = next((c for c in df.columns if "pass" in c and "yard" in c), None)
    rushing_col = next((c for c in df.columns if "rush" in c and "yard" in c), None)
    receiving_col = next((c for c in df.columns if "rec" in c and "yard" in c), None)

    if not player_col:
        st.error("❌ No player column detected.")
        st.stop()

    matches = df[df[player_col].str.contains(player_name, case=False, na=False)]
    if matches.empty:
        st.warning(f"No data found for '{player_name}'. Try partial name.")
        st.stop()

    # Pick stat column
    stat_map = {
        "Passing Yards": passing_col,
        "Rushing Yards": rushing_col,
        "Receiving Yards": receiving_col
    }
    stat_col = stat_map[stat_type]
    if not stat_col:
        st.error(f"❌ Could not detect column for {stat_type}.")
        st.stop()

    # Prepare player subset
    week_col = week_col or "week"
    player_df = matches[[week_col, stat_col]].sort_values(week_col).tail(lookback_weeks)
    player_df.rename(columns={stat_col: "value"}, inplace=True)

    # Plot
    fig = px.bar(player_df, x=week_col, y="value", text="value",
                 title=f"{player_name} — {stat_type} (Last {lookback_weeks} Weeks)")
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

st.caption("Data: Kaggle Big Data Bowl 2025 | Build vA59 | Vishvin Ramesh")
