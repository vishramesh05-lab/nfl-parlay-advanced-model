# NFL Parlay Helper (Dual Probabilities, 2025) — vA31
# Free open-source NFL FastR data + OpenWeather
# Author: Vish (2025)

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import re

st.set_page_config(page_title="NFL Parlay Helper (Dual Probabilities, 2025)",
                   layout="wide", page_icon="🏈")

st.markdown("<h1 style='text-align:center;'>🏈 NFL Parlay Helper (Dual Probabilities, 2025)</h1>",
            unsafe_allow_html=True)
st.caption("Live data + probability model — NFLfastR (free) + OpenWeather")
st.caption("Build vA31 | by Vish")

# Sidebar
st.sidebar.header("⚙️ Filters")
current_week = st.sidebar.slider("Current week", 1, 18, 6)
lookback_weeks = st.sidebar.slider("Lookback (weeks)", 1, 8, 5)
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

# Inputs
player_name = st.text_input("Player Name", placeholder="e.g. Patrick Mahomes")
stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
sportsbook_line = st.number_input("Sportsbook Line", step=0.5)
opponent_team = st.text_input("Opponent Team (e.g., KC, BUF, PHI)", "")
weather_city = st.text_input("Weather City (optional, e.g., Detroit)", "")

# --------------------------------------------------------------------
# URLs
NFLFASTR_URL = "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/player_stats.csv.gz"
OPENWEATHER_KEY = "demo"
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=imperial"

NFLFASTR_URL = "https://raw.githubusercontent.com/nflverse/nflverse-data/master/data/player_stats.csv.gz"

@st.cache_data(ttl=3600)
def load_nflfastR():
    """Load current NFL player stats from nflverse GitHub mirror (works globally)."""
    df = pd.read_csv(NFLFASTR_URL, compression="gzip")
    max_season = df["season"].max()
    df = df[df["season"] == max_season]
    return df

def fetch_weather(city):
    city = (city or "").strip()
    if not city:
        return None, None
    try:
        r = requests.get(OPENWEATHER_URL.format(city=city, key=OPENWEATHER_KEY), timeout=15)
        if r.status_code == 200:
            j = r.json()
            return j["weather"][0]["main"], j["main"]["temp"]
    except Exception:
        pass
    return None, None

def calc_prob(series, line, direction):
    if len(series) == 0: return 0.0
    hits = (series > line).sum() if direction == "Over" else (series < line).sum()
    return round(100 * hits / len(series), 1)

# --------------------------------------------------------------------
if st.button("Analyze Player", use_container_width=True):
    if not player_name or sportsbook_line <= 0:
        st.warning("Enter a valid player name and sportsbook line.")
        st.stop()

    st.info("Loading NFLfastR player dataset …")
    try:
        df = load_nflfastR()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    q = player_name.lower()
    players = df[df["player_name"].str.lower().str.contains(q)]
    if players.empty:
        st.error(f"No player found for '{player_name}'. Try last name only.")
        st.stop()

    # pick top-occurrence player_id (handles duplicates)
    pid = players["player_id"].mode()[0]
    p_df = df[df["player_id"] == pid].copy()

    stat_map = {
        "Passing Yards": "passing_yards",
        "Rushing Yards": "rushing_yards",
        "Receiving Yards": "receiving_yards"
    }
    stat_col = stat_map[stat_type]

    # keep only non-null weekly records
    p_df = p_df[p_df[stat_col].notna()]
    if p_df.empty:
        st.error(f"No {stat_type} data for {player_name}.")
        st.stop()

    # group by week
    view = p_df.groupby("week", as_index=False)[stat_col].sum().sort_values("week")

    # Visualize
    fig = px.bar(view, x="week", y=stat_col, text=stat_col,
                 title=f"{player_name} — {stat_type} ({int(p_df['season'].max())} Season)")
    fig.add_hline(y=sportsbook_line, line_color="red", annotation_text="Sportsbook Line")
    st.plotly_chart(fig, use_container_width=True)

    # Probability
    st.subheader("🎯 Baseline Probability")
    col1, col2 = st.columns(2)
    if col1.button("Over"):
        st.success(f"Over Probability: {calc_prob(view[stat_col], sportsbook_line, 'Over')}%")
    if col2.button("Under"):
        st.warning(f"Under Probability: {calc_prob(view[stat_col], sportsbook_line, 'Under')}%")

    # Context
    st.divider()
    st.subheader("📊 Context-Adjusted Probability")
    weather, temp = fetch_weather(weather_city)
    base = calc_prob(view[stat_col], sportsbook_line, "Over")
    adj = base
    if weather and "rain" in weather.lower():
        adj -= 8
    if temp and temp < 40:
        adj -= 5
    adj = max(0, min(100, adj))

    st.info(f"Weather: {weather or 'N/A'} | Temp: {('%.0f' % temp) if temp else 'N/A'} °F")
    st.success(f"Adjusted Over Probability: {adj}%")

# --------------------------------------------------------------------
st.markdown("---")
st.caption("Data: NFLfastR (GitHub mirror, free) • OpenWeather • Build vA31 (2025)")
