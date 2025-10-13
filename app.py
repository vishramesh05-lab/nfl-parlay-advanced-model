# NFL Parlay Helper (Dual Probabilities, 2025) — vA42
# Live 365Scores API + Over/Under probability model
# Author: Vish (2025)

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import datetime

# ---------------------------------------------------------------
# Page Setup
st.set_page_config(page_title="NFL Parlay Helper (Dual Probabilities, 2025)",
                   layout="wide", page_icon="🏈")

st.markdown("<h1 style='text-align:center;'>🏈 NFL Parlay Helper (Dual Probabilities, 2025)</h1>",
            unsafe_allow_html=True)
st.caption("Live data + probability model — 365Scores API + OpenWeather")
st.caption("Build vA42 | by Vish")

# ---------------------------------------------------------------
# Sidebar Filters
st.sidebar.header("⚙️ Filters")
current_week = st.sidebar.slider("Current week", 1, 18, 6)
lookback_weeks = st.sidebar.slider("Lookback (weeks)", 1, 8, 5)

# Inputs
player_name = st.text_input("Player Name", placeholder="e.g. Patrick Mahomes")
stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
sportsbook_line = st.number_input("Sportsbook Line", step=0.5)
opponent_team = st.text_input("Opponent Team (e.g., KC, BUF, PHI)", "")
weather_city = st.text_input("Weather City (optional, e.g., Detroit)", "")

OPENWEATHER_KEY = "demo"
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=imperial"

# ---------------------------------------------------------------
# Helper Functions

@st.cache_data(ttl=3600)
def fetch_365scores_player_stats():
    """
    Fetches live NFL 2025 stats from 365Scores public API.
    """
    base_url = "https://webapi.365scores.com/web/stats/players/?competition=352&season=2025"
    r = requests.get(base_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    data = r.json()
    records = []
    for p in data.get("players", []):
        name = p.get("name", "")
        stats = p.get("statistics", {})
        records.append({
            "player": name,
            "team": p.get("team", {}).get("shortName", ""),
            "passing_yards": stats.get("passingYards", 0),
            "rushing_yards": stats.get("rushingYards", 0),
            "receiving_yards": stats.get("receivingYards", 0),
            "games_played": stats.get("gamesPlayed", 0)
        })
    return pd.DataFrame(records)

def fetch_weather(city):
    if not city.strip():
        return None, None
    try:
        r = requests.get(OPENWEATHER_URL.format(city=city, key=OPENWEATHER_KEY), timeout=15)
        if r.status_code == 200:
            j = r.json()
            return j["weather"][0]["main"], j["main"]["temp"]
    except Exception:
        pass
    return None, None

def calc_prob(value, line, direction):
    """Simple binary probability (not historical) — extend later when time series available."""
    if direction == "Over":
        return 100 if value > line else 0
    return 100 if value < line else 0

# ---------------------------------------------------------------
# Main Logic
if st.button("Analyze Player", use_container_width=True):
    st.info("Fetching live player data from 365Scores …")
    try:
        df = fetch_365scores_player_stats()
    except Exception as e:
        st.error(f"Failed to load 365Scores data: {e}")
        st.stop()

    match = df[df["player"].str.lower().str.contains(player_name.lower().strip())]
    if match.empty:
        st.error(f"No player found for '{player_name}'. Try partial name (e.g. 'Mahomes').")
        st.stop()

    row = match.iloc[0]
    st.success(f"✅ Found {row['player']} ({row['team']})")

    # Select Stat
    stat_map = {
        "Passing Yards": "passing_yards",
        "Rushing Yards": "rushing_yards",
        "Receiving Yards": "receiving_yards"
    }
    col = stat_map[stat_type]
    value = row[col]

    # Chart (single value bar)
    view = pd.DataFrame({"Stat": [stat_type], "Value": [value]})
    fig = px.bar(view, x="Stat", y="Value", text="Value",
                 title=f"{row['player']} — {stat_type} (2025 Total)")
    fig.add_hline(y=sportsbook_line, line_color="red", annotation_text="Sportsbook Line")
    st.plotly_chart(fig, use_container_width=True)

    # Baseline Probability
    st.subheader("🎯 Baseline Probability")
    col1, col2 = st.columns(2)
    if col1.button("Over"):
        st.success(f"Over Probability: {calc_prob(value, sportsbook_line, 'Over')}%")
    if col2.button("Under"):
        st.warning(f"Under Probability: {calc_prob(value, sportsbook_line, 'Under')}%")

    # Adjusted Probability
    st.divider()
    st.subheader("📊 Context-Adjusted Probability")
    weather, temp = fetch_weather(weather_city)
    adj = calc_prob(value, sportsbook_line, "Over")
    if weather and "rain" in weather.lower():
        adj -= 8
    if temp and temp < 40:
        adj -= 5
    adj = max(0, min(100, adj))
    st.info(f"Opponent: {opponent_team or 'N/A'} | Weather: {weather or 'N/A'} | Temp: {('%.0f' % temp) if temp else 'N/A'} °F")
    st.success(f"Adjusted Over Probability: {adj}%")

    # Summary
    st.divider()
    st.subheader("📈 Player Summary")
    stats = {
        "Games Played": int(row["games_played"]),
        "Total Passing Yards": int(row["passing_yards"]),
        "Total Rushing Yards": int(row["rushing_yards"]),
        "Total Receiving Yards": int(row["receiving_yards"]),
    }
    st.table(pd.DataFrame([stats]))
    st.caption(f"Last refresh: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------------------------------------------
st.markdown("---")
st.caption("Data: 365Scores API (live) • OpenWeather • Build vA42 (2025)")
