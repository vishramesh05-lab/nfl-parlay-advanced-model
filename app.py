# NFL Parlay Helper (Dual Probabilities, 2025) — vA26 (patched)
# Live Sleeper + ESPN Defense + Weather integration

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime
import re
from difflib import get_close_matches

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
st.caption("Build vA26 | by Vish (patched)")

# Sidebar
st.sidebar.header("⚙️ Filters")
current_week = st.sidebar.slider("Current week", 1, 18, 6)
lookback_weeks = st.sidebar.slider("Lookback (weeks)", 1, 8, 5)

# Inputs
player_name = st.text_input("Player Name", placeholder="e.g. Patrick Mahomes")
stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
sportsbook_line = st.number_input("Sportsbook Line", step=0.5)
opponent_team = st.text_input("Opponent Team (e.g., KC, BUF, PHI)", "")
weather_city = st.text_input("Weather City (optional, e.g., Detroit)", "")

# APIs
SLEEPER_WEEKLY_BASE = "https://api.sleeper.app/v1/stats/nfl/regular"
SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
OPENWEATHER_KEY = "demo"   # replace with your OpenWeather key
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=imperial"
# NOTE: ESPN defense endpoint usually needs numeric team IDs; keep as best-effort
ESPN_DEFENSE_URL = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/teams"

# ---------- Helpers ----------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())

@st.cache_data(ttl=3600)
def load_sleeper_players():
    """Load once; build convenient search fields."""
    r = requests.get(SLEEPER_PLAYERS_URL, timeout=30)
    r.raise_for_status()
    raw = r.json()  # dict keyed by player_id
    players = []
    for pid, p in raw.items():
        if not isinstance(p, dict):
            continue
        full = p.get("full_name") or p.get("display_name") or ""
        players.append({
            "player_id": pid,
            "full_name": full,
            "last_name": p.get("last_name") or "",
            "team": p.get("team") or "",
            "position": p.get("position") or "",
            "_n_full": _norm(full),
            "_n_last": _norm(p.get("last_name") or ""),
        })
    return players

def resolve_player_id(name: str):
    """Find best matching player_id from user-provided name/text."""
    q = _norm(name)
    players = load_sleeper_players()
    # direct contains on normalized names
    direct = [p for p in players if q and (q in p["_n_full"] or q in p["_n_last"])]
    if direct:
        # if multiple (e.g., multiple Johnsons), prefer QBs for passing stats
        return direct[0]
    # fuzzy fallback
    names = [p["full_name"] for p in players]
    best = get_close_matches(name, names, n=1, cutoff=0.75)
    if best:
        for p in players:
            if p["full_name"] == best[0]:
                return p
    return None

@st.cache_data(ttl=1800)
def fetch_sleeper_week(year: int, week: int):
    """
    Sleeper weekly endpoint returns a LIST of player stat dicts.
    """
    url = f"{SLEEPER_WEEKLY_BASE}/{year}/{week}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return []
    data = r.json()
    # Some deployments get back an object; normalize to list
    if isinstance(data, dict):
        # older mirrors occasionally key by 'player_id' -> dict
        data = list(data.values())
    return data if isinstance(data, list) else []

def find_player_weekly(player_name: str, year: int, weeks: list[int]) -> pd.DataFrame:
    """
    Resolve player_id once, then filter weekly list rows by that id.
    Map short stat keys to readable columns.
    """
    player = resolve_player_id(player_name)
    if not player:
        return pd.DataFrame()
    pid = player["player_id"]

    rows = []
    for wk in weeks:
        lst = fetch_sleeper_week(year, wk)
        if not lst:
            continue
        # rows include 'player_id' and stat keys like 'pass_yd', 'rush_yd', 'rec_yd'
        for row in lst:
            if row.get("player_id") == pid:
                rows.append({
                    "week": wk,
                    "passing_yards": float(row.get("pass_yd", 0) or 0),
                    "rushing_yards": float(row.get("rush_yd", 0) or 0),
                    "receiving_yards": float(row.get("rec_yd", 0) or 0),
                    "opp": row.get("opponent") or row.get("opp") or "",
                })
                break  # found this player's row for the week

    return pd.DataFrame(rows)

def fetch_weather(city):
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
    """
    Best-effort: pull ESPN team index and try to read a simple yards allowed value.
    Many ESPN stat endpoints require numeric IDs + category parsing; return None if unavailable.
    """
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
                # Not all payloads include defense ranks in this listing; return None gracefully.
                return None
    except Exception:
        pass
    return None

def calc_prob(series, line, direction):
    if len(series) == 0:
        return 0.0
    hits = (series > line).sum() if direction == "Over" else (series < line).sum()
    return round(100 * hits / len(series), 1)

# ---------- UI Action ----------

if st.button("Analyze Player", use_container_width=True):
    if not player_name or sportsbook_line <= 0:
        st.warning("Enter a valid player name and sportsbook line.")
    else:
        st.info("Fetching live data from Sleeper API …")
        weeks = list(range(max(1, current_week - lookback_weeks + 1), current_week + 1))

        df = find_player_weekly(player_name, 2025, weeks)

        if df.empty:
            st.error(f"No stats found for '{player_name}' in the selected window. "
                     "Try last name only (e.g., 'Mahomes') or double-check the week/season.")
        else:
            stat_map = {
                "Passing Yards": "passing_yards",
                "Rushing Yards": "rushing_yards",
                "Receiving Yards": "receiving_yards"
            }
            stat_col = stat_map[stat_type]
            view = df[["week", stat_col]].rename(columns={stat_col: "value"})

            fig = px.bar(
                view, x="week", y="value", text="value",
                title=f"{player_name} — {stat_type} (last {lookback_weeks} weeks)"
            )
            fig.add_hline(y=sportsbook_line, line_color="red", annotation_text="Sportsbook Line")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🎯 Baseline Probability")
            col1, col2 = st.columns(2)
            if col1.button("Over"):
                st.success(f"Over Probability: {calc_prob(view['value'], sportsbook_line, 'Over')}%")
            if col2.button("Under"):
                st.warning(f"Under Probability: {calc_prob(view['value'], sportsbook_line, 'Under')}%")

            # Context
            st.divider()
            st.subheader("📊 Context-Adjusted Probability")

            weather, temp = fetch_weather(weather_city)
            defense_rank = fetch_defense_rank(opponent_team)

            base = calc_prob(view["value"], sportsbook_line, "Over")
            adj = base
            if defense_rank is not None:
                # Example adjustment placeholder; tune when a numeric rank is available
                adj -= min(10, max(0, (32 - float(defense_rank)) / 3))
            if weather and "rain" in weather.lower():
                adj -= 8
            if temp is not None and temp < 40:
                adj -= 5
            adj = max(0, min(100, adj))

            st.info(f"Opponent: {opponent_team or 'N/A'} | Defense Rank: {defense_rank or 'N/A'} | "
                    f"Weather: {weather or 'N/A'} | Temp: {('%.0f' % temp) if temp else 'N/A'} °F")
            st.success(f"Adjusted Over Probability: {adj}%")
