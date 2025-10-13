# NFL Parlay Helper (Dual Probabilities, 2025) — vA28
# Live Sleeper + ESPN Defense + Weather integration
# Author: Vish (2025)

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import re
from difflib import get_close_matches

# -------------------------------------------------------------------------
# Page Config
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
st.caption("Build vA28 | by Vish")

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
# API Endpoints
SLEEPER_BASE = "https://api.sleeper.app/v1/stats/nfl"
SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
OPENWEATHER_KEY = "demo"   # replace with your real OpenWeather key
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=imperial"
ESPN_DEFENSE_URL = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/teams"

# -------------------------------------------------------------------------
# Utility Functions

def _norm(s: str) -> str:
    """Normalize names (remove punctuation, lowercase)."""
    return re.sub(r"[^a-z]", "", (s or "").lower())

@st.cache_data(ttl=3600)
def load_sleeper_players():
    """Loads all Sleeper NFL players once and caches."""
    r = requests.get(SLEEPER_PLAYERS_URL, timeout=30)
    r.raise_for_status()
    return r.json()

def get_player_info(name: str):
    """Find player_id from Sleeper master list."""
    data = load_sleeper_players()
    q = _norm(name)
    matches = []
    for pid, p in data.items():
        full = _norm(p.get("full_name") or "")
        disp = _norm(p.get("display_name") or "")
        last = _norm(p.get("last_name") or "")
        if q in full or q in disp or q in last:
            matches.append({
                "id": pid,
                "name": p.get("full_name"),
                "team": p.get("team"),
                "pos": p.get("position")
            })
    if matches:
        return matches[0]

    # fuzzy fallback
    names = [p.get("full_name", "") for p in data.values() if p.get("full_name")]
    best = get_close_matches(name, names, n=1, cutoff=0.75)
    if best:
        for pid, p in data.items():
            if p.get("full_name") == best[0]:
                return {"id": pid, "name": p.get("full_name"), "team": p.get("team"), "pos": p.get("position")}
    return None

@st.cache_data(ttl=1200)
def fetch_sleeper_week(year: int, week: int):
    """
    Fetch weekly stats with automatic season_type fallback.
    Sleeper returns {} if data for that type/week doesn’t exist.
    """
    urls = [
        f"{SLEEPER_BASE}/regular/{year}/{week}?season_type=regular",
        f"{SLEEPER_BASE}/post/{year}/{week}?season_type=post",
        f"{SLEEPER_BASE}/pre/{year}/{week}?season_type=pre"
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=20)
            if debug_mode:
                st.write("Fetching:", url, "| Status:", r.status_code)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    data = list(data.values())
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception as e:
            if debug_mode:
                st.write("Fetch error:", e)
            continue
    return []

def find_player_weekly(name: str, year: int, weeks: list[int]) -> pd.DataFrame:
    """Retrieve player stats by resolved player_id."""
    player = get_player_info(name)
    if not player:
        st.warning("Could not match player name in Sleeper catalog.")
        return pd.DataFrame()
    pid = player["id"]
    records = []

    for wk in weeks:
        week_data = fetch_sleeper_week(year, wk)
        if not week_data:
            continue
        for row in week_data:
            if row.get("player_id") == pid:
                records.append({
                    "week": wk,
                    "passing_yards": float(row.get("pass_yd", 0) or 0),
                    "rushing_yards": float(row.get("rush_yd", 0) or 0),
                    "receiving_yards": float(row.get("rec_yd", 0) or 0),
                    "opp": row.get("opponent") or row.get("opp") or ""
                })
                break

    df = pd.DataFrame(records)
    if debug_mode:
        st.write("Fetched weeks:", [r["week"] for r in records])
        if not df.empty:
            st.dataframe(df.head())
    return df

def fetch_weather(city):
    """Get weather for specified city."""
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

def fetch_defense_rank(team_abbr):
    """Placeholder: ESPN team data (optional extension)."""
    try:
        r = requests.get(ESPN_DEFENSE_URL, timeout=20)
        if r.status_code != 200:
            return None
        js = r.json()
        teams = js.get("sports", [])[0].get("leagues", [])[0].get("teams", [])
        team_abbr = (team_abbr or "").upper().strip()
        for t in teams:
            team = t.get("team", {})
            if team.get("abbreviation", "").upper() == team_abbr:
                return None
    except Exception:
        pass
    return None

def calc_prob(series, line, direction):
    """Calculate percentage chance of over/under hitting."""
    if len(series) == 0:
        return 0.0
    hits = (series > line).sum() if direction == "Over" else (series < line).sum()
    return round(100 * hits / len(series), 1)

# -------------------------------------------------------------------------
# Main Analysis
if st.button("Analyze Player", use_container_width=True):
    if not player_name or sportsbook_line <= 0:
        st.warning("Enter a valid player name and sportsbook line.")
    else:
        st.info("Fetching live data from Sleeper API …")
        weeks = list(range(max(1, current_week - lookback_weeks + 1), current_week + 1))
        df = find_player_weekly(player_name, 2025, weeks)

        if df.empty:
            st.error(f"No stats found for '{player_name}'. Try verifying name or season/week availability.")
        else:
            stat_map = {
                "Passing Yards": "passing_yards",
                "Rushing Yards": "rushing_yards",
                "Receiving Yards": "receiving_yards"
            }
            stat_col = stat_map[stat_type]
            view = df[["week", stat_col]].rename(columns={stat_col: "value"})

            # ----- Visualization -----
            fig = px.bar(
                view, x="week", y="value", text="value",
                title=f"{player_name} — {stat_type} (Last {lookback_weeks} Weeks)"
            )
            fig.add_hline(y=sportsbook_line, line_color="red", annotation_text="Sportsbook Line")
            st.plotly_chart(fig, use_container_width=True)

            # ----- Base Probability -----
            st.subheader("🎯 Baseline Probability")
            col1, col2 = st.columns(2)
            if col1.button("Over"):
                st.success(f"Over Probability: {calc_prob(view['value'], sportsbook_line, 'Over')}%")
            if col2.button("Under"):
                st.warning(f"Under Probability: {calc_prob(view['value'], sportsbook_line, 'Under')}%")

            # ----- Context Adjustments -----
            st.divider()
            st.subheader("📊 Context-Adjusted Probability")
            weather, temp = fetch_weather(weather_city)
            defense_rank = fetch_defense_rank(opponent_team)

            base = calc_prob(view["value"], sportsbook_line, "Over")
            adj = base
            if defense_rank is not None:
                adj -= min(10, max(0, (32 - float(defense_rank)) / 3))
            if weather and "rain" in weather.lower():
                adj -= 8
            if temp is not None and temp < 40:
                adj -= 5
            adj = max(0, min(100, adj))

            st.info(f"Opponent: {opponent_team or 'N/A'} | Defense Rank: {defense_rank or 'N/A'} | "
                    f"Weather: {weather or 'N/A'} | Temp: {('%.0f' % temp) if temp else 'N/A'} °F")
            st.success(f"Adjusted Over Probability: {adj}%")

# -------------------------------------------------------------------------
st.markdown("---")
st.caption("Data: Sleeper API • ESPN • OpenWeather • Build vA28 (2025)")
