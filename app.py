# 🏈 NFL Parlay Helper (Dual Probabilities, 2025)
# Source: FootballDB (weekly) + OpenWeather
# Build vA49 | Vishvin Ramesh

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="NFL Parlay Helper (Dual Probabilities, 2025)",
    page_icon="🏈",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;'>🏈 NFL Parlay Helper (Dual Probabilities, 2025)</h1>",
    unsafe_allow_html=True
)
st.caption("Live data probability model — FootballDB (weekly) + OpenWeather")
st.caption("Build vA49 | by Vishvin Ramesh")

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Filters")
current_week = st.sidebar.slider("Current Week", 1, 18, 6)
lookback_weeks = st.sidebar.slider("Lookback (weeks)", 1, 8, 5)
debug_mode = st.sidebar.checkbox("Enable Debug Mode", False)

# ---------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------
player_name = st.text_input("Player Name", placeholder="e.g. Patrick Mahomes")
stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
sportsbook_line = st.number_input("Sportsbook Line", step=0.5)
opponent_team = st.text_input("Opponent Team (e.g., KC, BUF, PHI)")
weather_city = st.text_input("Weather City (optional, e.g., Detroit)")

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
OPENWEATHER_KEY = "demo"  # replace with your key
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=imperial"
CACHE_FILE = "footballdb_2025.csv"

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_footballdb_data():
    """Scrape weekly player stats from FootballDB (passing/rushing/receiving)."""
    urls = {
        "Passing Yards": "https://www.footballdb.com/stats/stats.html?lg=NFL&yr=2025&type=reg&cat=passing",
        "Rushing Yards": "https://www.footballdb.com/stats/stats.html?lg=NFL&yr=2025&type=reg&cat=rushing",
        "Receiving Yards": "https://www.footballdb.com/stats/stats.html?lg=NFL&yr=2025&type=reg&cat=receiving"
    }
    all_data = []
    for cat, url in urls.items():
        try:
            r = requests.get(url, timeout=15)
            soup = BeautifulSoup(r.text, "html.parser")
            table = soup.find("table", {"class": "statistics"})
            df = pd.read_html(str(table))[0]
            df["Category"] = cat
            all_data.append(df)
        except Exception as e:
            if debug_mode:
                st.error(f"Error loading {cat}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def fetch_weather(city):
    """Fetch weather and temperature."""
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

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
if st.button("🔁 Reload Latest FootballDB Data", use_container_width=True):
    st.cache_data.clear()
    st.success("Data cache cleared. Click 'Analyze Player' to refresh with latest stats.")

st.divider()

# ---------------------------------------------------------
# MAIN ACTION
# ---------------------------------------------------------
if st.button("Analyze Player", use_container_width=True):
    st.info("Fetching live weekly stats from FootballDB …")
    df = fetch_footballdb_data()

    if df.empty:
        st.error("⚠️ Failed to load player data from FootballDB.")
    else:
        # Normalize player name column (FootballDB uses multiple naming styles)
        df.columns = [c.strip() for c in df.columns]
        player_matches = df[df["Player"].str.contains(player_name, case=False, na=False)]

        if player_matches.empty:
            st.warning(f"No data found for '{player_name}'. Try a shorter name fragment (e.g., 'Mahomes').")
        else:
            # Select by category
            stat_df = player_matches[player_matches["Category"] == stat_type]
            if stat_df.empty:
                st.warning(f"No {stat_type} data for {player_name}.")
            else:
                # Extract numeric yard values
                if stat_type == "Passing Yards":
                    values = stat_df["Yards"].astype(float)
                elif stat_type == "Rushing Yards":
                    values = stat_df["Yards"].astype(float)
                else:
                    values = stat_df["Yards"].astype(float)

                # Visualization
                fig = px.bar(
                    x=np.arange(len(values)),
                    y=values,
                    text=values,
                    title=f"{player_name} — {stat_type} (Last {lookback_weeks} Games)"
                )
                fig.add_hline(y=sportsbook_line, line_color="red", annotation_text="Sportsbook Line")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("🎯 Baseline Probability")
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"Over Probability: {calc_prob(values, sportsbook_line, 'Over')}%")
                with col2:
                    st.warning(f"Under Probability: {calc_prob(values, sportsbook_line, 'Under')}%")

                # Weather adjustment
                weather, temp = fetch_weather(weather_city)
                adj = calc_prob(values, sportsbook_line, "Over")
                if weather and "rain" in weather.lower():
                    adj -= 8
                if temp and temp < 40:
                    adj -= 5
                adj = max(0, min(100, adj))
                st.divider()
                st.info(f"Opponent: {opponent_team or 'N/A'} | Weather: {weather or 'N/A'} | Temp: {temp or 'N/A'}°F")
                st.success(f"Adjusted Over Probability: {adj}%")

st.caption("Data: FootballDB (weekly, public) | Weather: OpenWeather | Build vA49")
