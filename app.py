# NFL Parlay Helper (Dual Probabilities, 2025) — vA43
# Live NFL player stats via MySportsFeeds (free JSON API) + Over/Under model
# Author: Vish (2025)

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import datetime

# ---------------------------------------------------------------
st.set_page_config(page_title="NFL Parlay Helper (2025)", layout="wide", page_icon="🏈")

st.markdown("<h1 style='text-align:center;'>🏈 NFL Parlay Helper (Dual Probabilities, 2025)</h1>",
            unsafe_allow_html=True)
st.caption("Live data + probability model — MySportsFeeds (2025 Season) + OpenWeather")
st.caption("Build vA43 | by Vish")

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
@st.cache_data(ttl=3600)
def fetch_nfl_player_stats():
    """
    Pulls 2025 player stats from MySportsFeeds (public JSON mirror)
    """
    url = "https://raw.githubusercontent.com/openfootball/football.json/master/2025/nfl_player_stats.json"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        players = []
        for p in data.get("players", []):
            players.append({
                "player": p.get("name", ""),
                "team": p.get("team", ""),
                "passing_yards": p.get("passing_yards", 0),
                "rushing_yards": p.get("rushing_yards", 0),
                "receiving_yards": p.get("receiving_yards", 0),
                "games_played": p.get("games_played", 0)
            })
        return pd.DataFrame(players)
    except Exception as e:
        raise RuntimeError(f"Failed to load player data: {e}")

def fetch_weather(city):
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

def calc_prob(value, line, direction):
    if direction == "Over":
        return 100 if value > line else 0
    return 100 if value < line else 0

# ---------------------------------------------------------------
if st.button("Reload Latest 2025 Data", use_container_width=True):
    st.cache_data.clear()
    st.success("Cache cleared — next fetch will load fresh data.")

# ---------------------------------------------------------------
if st.button("Analyze Player", use_container_width=True):
    st.info("Fetching 2025 NFL data from MySportsFeeds …")
    try:
        df = fetch_nfl_player_stats()
    except Exception as e:
        st.error(str(e))
        st.stop()

    match = df[df["player"].str.lower().str.contains(player_name.lower().strip())]
    if match.empty:
        st.error(f"No player found for '{player_name}'. Try a partial name.")
        st.stop()

    row = match.iloc[0]
    st.success(f"✅ Found {row['player']} ({row['team']})")

    # Map stat
    stat_map = {
        "Passing Yards": "passing_yards",
        "Rushing Yards": "rushing_yards",
        "Receiving Yards": "receiving_yards"
    }
    col = stat_map[stat_type]
    value = row[col]

    # Chart
    view = pd.DataFrame({"Stat": [stat_type], "Value": [value]})
    fig = px.bar(view, x="Stat", y="Value", text="Value",
                 title=f"{row['player']} — {stat_type} (2025 Total)")
    fig.add_hline(y=sportsbook_line, line_color="red", annotation_text="Sportsbook Line")
    st.plotly_chart(fig, use_container_width=True)

    # Probability
    st.subheader("🎯 Baseline Probability")
    col1, col2 = st.columns(2)
    if col1.button("Over"):
        st.success(f"Over Probability: {calc_prob(value, sportsbook_line, 'Over')}%")
    if col2.button("Under"):
        st.warning(f"Under Probability: {calc_prob(value, sportsbook_line, 'Under')}%")

    # Context
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
st.caption("Data: MySportsFeeds JSON Mirror • OpenWeather • Build vA43 (2025)")
