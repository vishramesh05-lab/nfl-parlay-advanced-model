# NFL Parlay Helper (Dual Probabilities, 2025) — vA40
# Live scraper from Pro-Football-Reference (2025)
# Author: Vish

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import datetime

st.set_page_config(page_title="NFL Parlay Helper (Dual Probabilities, 2025)",
                   layout="wide", page_icon="🏈")

st.markdown("<h1 style='text-align:center;'>🏈 NFL Parlay Helper (Dual Probabilities, 2025)</h1>",
            unsafe_allow_html=True)
st.caption("Live data + probability model — Pro-Football-Reference + OpenWeather")
st.caption("Build vA40 | by Vish")

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

OPENWEATHER_KEY = "demo"
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=imperial"

# ---------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_pfr_data(player_url):
    """Scrape weekly stats for the player from Pro-Football-Reference."""
    r = requests.get(player_url, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        raise ValueError(f"HTTP {r.status_code} from PFR.")
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", id="passing" if "MahoPa00" in player_url else None)
    # If not found, use pandas fallback:
    tables = pd.read_html(r.text)
    df = pd.concat(tables, axis=0)
    # Keep only numeric week rows
    df = df[df["Week"].astype(str).str.isnumeric()]
    df["Week"] = df["Week"].astype(int)
    return df

def find_player_url(name):
    """Map player name → Pro-Football-Reference URL (simplified)."""
    base = "https://www.pro-football-reference.com/players/"
    lookup = {
        "patrick mahomes": "M/MahoPa00",
        "jayden daniels": "D/DaniJa03",
        "josh allen": "A/AlleJo02",
        "lamar jackson": "J/JackLa00",
        "joe burrow": "B/BurrJo01",
    }
    name = name.lower().strip()
    if name in lookup:
        return base + lookup[name] + "/gamelog/2025/"
    raise ValueError("Player not in lookup. Add to dictionary manually.")

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
    if len(series) == 0:
        return 0.0
    hits = (series > line).sum() if direction == "Over" else (series < line).sum()
    return round(100 * hits / len(series), 1)

# ---------------------------------------------------------------------
if st.button("Analyze Player", use_container_width=True):
    if not player_name or sportsbook_line <= 0:
        st.warning("Enter a valid player name and sportsbook line.")
        st.stop()

    try:
        player_url = find_player_url(player_name)
        st.info(f"Fetching live data from Pro-Football-Reference …")
        df = fetch_pfr_data(player_url)
    except Exception as e:
        st.error(f"Failed to load player data: {e}")
        st.stop()

    # Map stat column
    stat_map = {
        "Passing Yards": ["Yds", "PassYds", "Yds.1"],
        "Rushing Yards": ["RushYds", "Yds.2"],
        "Receiving Yards": ["RecYds", "Yds.3"]
    }
    cols = stat_map[stat_type]
    col = next((c for c in cols if c in df.columns), None)
    if not col:
        st.error("No matching stat column found.")
        st.stop()

    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    view = df[["Week", col]].rename(columns={col: "value"}).sort_values("Week")

    # Chart
    fig = px.bar(view, x="Week", y="value", text="value",
                 title=f"{player_name.title()} — {stat_type} (2025 Season)")
    fig.add_hline(y=sportsbook_line, line_color="red", annotation_text="Sportsbook Line")
    st.plotly_chart(fig, use_container_width=True)

    # Probability
    st.subheader("🎯 Baseline Probability")
    col1, col2 = st.columns(2)
    if col1.button("Over"):
        st.success(f"Over Probability: {calc_prob(view['value'], sportsbook_line, 'Over')}%")
    if col2.button("Under"):
        st.warning(f"Under Probability: {calc_prob(view['value'], sportsbook_line, 'Under')}%")

    # Adjusted Probability
    st.divider()
    st.subheader("📊 Context-Adjusted Probability")
    weather, temp = fetch_weather(weather_city)
    base = calc_prob(view["value"], sportsbook_line, "Over")
    adj = base
    if weather and "rain" in weather.lower(): adj -= 8
    if temp and temp < 40: adj -= 5
    adj = max(0, min(100, adj))
    st.info(f"Opponent: {opponent_team or 'N/A'} | Weather: {weather or 'N/A'} | Temp: {('%.0f' % temp) if temp else 'N/A'} °F")
    st.success(f"Adjusted Over Probability: {adj}%")

    # Summary
    st.divider()
    st.subheader("📈 Season Summary")
    stats = {
        "Games Played": len(view),
        "Average": round(view["value"].mean(), 1),
        "Std Dev": round(view["value"].std(), 1),
        "Max": view["value"].max(),
        "Last 3 Avg": round(view.tail(3)["value"].mean(), 1)
    }
    st.table(pd.DataFrame([stats]))
    st.caption(f"Last refresh: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------------------------------------------------
st.markdown("---")
st.caption("Data: Pro-Football-Reference (live scrape) • OpenWeather • Build vA40 (2025)")
