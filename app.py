# NFL Parlay Helper (Dual Probabilities, 2025) — vA30
# Free ESPN API + OpenWeather integration
# Author: Vish (2025)

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import re
from difflib import get_close_matches

# -------------------------------------------------------------------------
st.set_page_config(page_title="NFL Parlay Helper (Dual Probabilities, 2025)",
                   layout="wide", page_icon="🏈")

st.markdown(
    "<h1 style='text-align:center;'>🏈 NFL Parlay Helper (Dual Probabilities, 2025)</h1>",
    unsafe_allow_html=True
)
st.caption("Live data + probability model — ESPN + OpenWeather (Free Data Source)")
st.caption("Build vA30 | by Vish")

# -------------------------------------------------------------------------
# Sidebar Filters
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

# -------------------------------------------------------------------------
# ESPN + OpenWeather URLs
ESPN_SEARCH_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes"
ESPN_PLAYER_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/{pid}/statistics"
OPENWEATHER_KEY = "demo"  # replace with real key if desired
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=imperial"

# -------------------------------------------------------------------------
def _norm(s): return re.sub(r"[^a-z]", "", (s or "").lower())

@st.cache_data(ttl=3600)
def search_player_espm(name):
    """Search player by name via ESPN free JSON index."""
    q = _norm(name)
    resp = requests.get(ESPN_SEARCH_URL, timeout=20)
    if resp.status_code != 200:
        return None
    data = resp.json().get("items", [])
    for p in data:
        nm = _norm(p.get("displayName", ""))
        if q in nm:
            return p
    # fuzzy fallback
    names = [p.get("displayName", "") for p in data]
    best = get_close_matches(name, names, n=1, cutoff=0.75)
    if best:
        for p in data:
            if p.get("displayName") == best[0]:
                return p
    return None

def fetch_player_stats(pid):
    """Pull player stat history from ESPN (includes 2025)."""
    url = ESPN_PLAYER_URL.format(pid=pid)
    r = requests.get(url, timeout=25)
    if r.status_code != 200:
        return None
    j = r.json()
    splits = j.get("splits", [])
    weekly = []
    for s in splits:
        if "statSource" in s and "week" in s:
            wk = s.get("week")
            stats = s.get("stats", {})
            weekly.append({
                "week": wk,
                "passing_yards": float(stats.get("passingYards", 0)),
                "rushing_yards": float(stats.get("rushingYards", 0)),
                "receiving_yards": float(stats.get("receivingYards", 0))
            })
    return pd.DataFrame(weekly)

def fetch_weather(city):
    """Fetch weather from OpenWeather."""
    city = (city or "").strip()
    if not city:
        return None, None
    try:
        r = requests.get(OPENWEATHER_URL.format(city=city, key=OPENWEATHER_KEY), timeout=20)
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

# -------------------------------------------------------------------------
if st.button("Analyze Player", use_container_width=True):
    if not player_name or sportsbook_line <= 0:
        st.warning("Enter a valid player name and sportsbook line.")
    else:
        st.info("Fetching data from ESPN API …")

        # Search player on ESPN
        player = search_player_espm(player_name)
        if not player:
            st.error(f"No player found for '{player_name}'. Try full name.")
            st.stop()

        pid = player.get("id") or player.get("uid", "").split(":")[-1]
        df = fetch_player_stats(pid)

        if df is None or df.empty:
            st.error(f"No stats available for '{player_name}' (ESPN data may be limited).")
            st.stop()

        stat_map = {
            "Passing Yards": "passing_yards",
            "Rushing Yards": "rushing_yards",
            "Receiving Yards": "receiving_yards"
        }
        stat_col = stat_map[stat_type]
        df = df[df["week"].notna()].sort_values("week")

        # --------------- Visualization ---------------
        fig = px.bar(df, x="week", y=stat_col, text=stat_col,
                     title=f"{player_name} — {stat_type} (2025 Season)")
        fig.add_hline(y=sportsbook_line, line_color="red", annotation_text="Sportsbook Line")
        st.plotly_chart(fig, use_container_width=True)

        # --------------- Probability ---------------
        st.subheader("🎯 Baseline Probability")
        col1, col2 = st.columns(2)
        if col1.button("Over"):
            st.success(f"Over Probability: {calc_prob(df[stat_col], sportsbook_line, 'Over')}%")
        if col2.button("Under"):
            st.warning(f"Under Probability: {calc_prob(df[stat_col], sportsbook_line, 'Under')}%")

        # --------------- Context Adjustments ---------------
        st.divider()
        st.subheader("📊 Context-Adjusted Probability")

        weather, temp = fetch_weather(weather_city)
        base = calc_prob(df[stat_col], sportsbook_line, "Over")
        adj = base
        if weather and "rain" in weather.lower():
            adj -= 8
        if temp and temp < 40:
            adj -= 5
        adj = max(0, min(100, adj))

        st.info(f"Weather: {weather or 'N/A'} | Temp: {('%.0f' % temp) if temp else 'N/A'} °F")
        st.success(f"Adjusted Over Probability: {adj}%")

# -------------------------------------------------------------------------
st.markdown("---")
st.caption("Data: ESPN API (Free) • OpenWeather • Build vA30 (2025)")
