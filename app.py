# NFL Parlay Helper (Dual Probabilities, 2025) — vA26
# Live Sleeper + ESPN Defense + Weather integration

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="NFL Parlay Helper (Dual Probabilities, 2025)",
    layout="wide",
    page_icon="🏈"
)

st.markdown(
    "<h1 style='text-align:center;'>🏈 NFL Parlay Helper (Dual Probabilities, 2025)</h1>",
    unsafe_allow_html=True
)
st.caption("Live data + probability model — Sleeper API (2025) + ESPN Defense + OpenWeather")
st.caption("Build vA26 | by Vish")

# Sidebar
st.sidebar.header("⚙️ Filters")
current_week = st.sidebar.slider("Current week", 1, 18, 6)
lookback_weeks = st.sidebar.slider("Lookback (weeks)", 1, 8, 5)

# Inputs
player_name = st.text_input("Player Name", placeholder="e.g. Patrick Mahomes")
stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
sportsbook_line = st.number_input("Sportsbook Line", step=0.5)
opponent_team = st.text_input("Opponent Team (e.g., KC, BUF, PHI)", "")

# APIs
SLEEPER_BASE = "https://api.sleeper.app/v1/stats/nfl/regular"
OPENWEATHER_KEY = "demo"   # replace with your OpenWeather key
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=imperial"
ESPN_DEFENSE_URL = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team}/statistics"

# -------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_sleeper_week(year, week):
    url = f"{SLEEPER_BASE}/{year}/{week}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

def find_player_data(player_name, year, weeks):
    all_weeks = []
    for wk in weeks:
        data = fetch_sleeper_week(year, wk)
        for pid, pstats in data.items():
            name_fields = [
                str(pstats.get("player_name", "")),
                str(pstats.get("full_name", "")),
                str(pstats.get("display_name", "")),
            ]
            if any(player_name.lower() in n.lower() for n in name_fields):
                all_weeks.append({
                    "week": wk,
                    "passing_yards": pstats.get("pass_yd", 0),
                    "rushing_yards": pstats.get("rush_yd", 0),
                    "receiving_yards": pstats.get("rec_yd", 0)
                })
    return pd.DataFrame(all_weeks)

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

def fetch_defense_rank(team_abbr):
    """Scrape ESPN defensive rankings (yards allowed) for adjustment."""
    try:
        r = requests.get(ESPN_DEFENSE_URL.format(team=team_abbr.lower()))
        if r.status_code == 200:
            js = r.json()
            for cat in js.get("splits", []):
                if "defense" in cat.get("name", "").lower():
                    val = cat["stats"][0].get("value", None)
                    if val:
                        return float(val)
    except Exception:
        pass
    return None

def calc_prob(series, line, direction):
    if len(series) == 0:
        return 0.0
    hits = (series > line).sum() if direction == "Over" else (series < line).sum()
    return round(100 * hits / len(series), 1)

# -------------------------------------------------------------------------
if st.button("Analyze Player", use_container_width=True):
    if not player_name or sportsbook_line <= 0:
        st.warning("Enter a valid player name and sportsbook line.")
    else:
        st.info("Fetching live data from Sleeper API …")
        weeks = list(range(max(1, current_week - lookback_weeks + 1), current_week + 1))
        df = find_player_data(player_name, 2025, weeks)

        if df.empty:
            st.error(f"No stats found for {player_name}. Try a shorter name fragment (e.g. 'Mahomes').")
        else:
            stat_map = {
                "Passing Yards": "passing_yards",
                "Rushing Yards": "rushing_yards",
                "Receiving Yards": "receiving_yards"
            }
            stat_col = stat_map[stat_type]
            df = df[["week", stat_col]].rename(columns={stat_col: "value"})

            fig = px.bar(
                df, x="week", y="value",
                text="value",
                title=f"{player_name} — {stat_type} (last {lookback_weeks} weeks)"
            )
            fig.add_hline(y=sportsbook_line, line_color="red", annotation_text="Sportsbook Line")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🎯 Baseline Probability")
            col1, col2 = st.columns(2)
            if col1.button("Over"):
                st.success(f"Over Probability: {calc_prob(df['value'], sportsbook_line, 'Over')}%")
            if col2.button("Under"):
                st.warning(f"Under Probability: {calc_prob(df['value'], sportsbook_line, 'Under')}%")

            # Context
            st.divider()
            st.subheader("📊 Context-Adjusted Probability")
            weather, temp = fetch_weather(opponent_team)
            defense_rank = fetch_defense_rank(opponent_team)

            base = calc_prob(df["value"], sportsbook_line, "Over")
            adj = base
            if defense_rank:
                # Lower defensive rank = harder matchup
                adj -= min(10, max(0, (32 - defense_rank) / 3))
            if weather and "rain" in weather.lower():
                adj -= 8
            if temp and temp < 40:
                adj -= 5
            adj = max(0, min(100, adj))

            st.info(f"Opponent: {opponent_team} | Defense Rank: {defense_rank or 'N/A'} | "
                    f"Weather: {weather or 'N/A'} | Temp: {temp or 'N/A'} °F")
            st.success(f"Adjusted Over Probability: {adj}%")
