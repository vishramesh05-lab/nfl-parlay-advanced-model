# NFL Parlay Helper (Dual Probabilities, 2025) — vA41
# Live JSON API version using SportsDataverse (PFR mirror)
# Author: Vish (2025)

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import datetime

st.set_page_config(
    page_title="NFL Parlay Helper (Dual Probabilities, 2025)",
    layout="wide",
    page_icon="🏈"
)

st.markdown("<h1 style='text-align:center;'>🏈 NFL Parlay Helper (Dual Probabilities, 2025)</h1>",
            unsafe_allow_html=True)
st.caption("Live data + probability model — SportsDataverse NFL API + OpenWeather")
st.caption("Build vA41 | by Vish")

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
def fetch_player_data(player_name: str):
    """
    Query SportsDataverse for NFL player game logs.
    """
    base_url = "https://sportsdataverse.net/nfl/players/search"
    try:
        search = requests.get(f"{base_url}?q={player_name}", timeout=15)
        search.raise_for_status()
        results = search.json()
        if len(results) == 0:
            raise ValueError("Player not found in SportsDataverse.")
        # pick top match
        player_id = results[0]["id"]
        player_url = f"https://sportsdataverse.net/nfl/players/{player_id}_gamelog_2025.json"
        data = requests.get(player_url, timeout=20)
        data.raise_for_status()
        df = pd.DataFrame(data.json()["games"])
        df = df[df["week"].notna()]
        df["week"] = df["week"].astype(int)
        return df, results[0]["name"]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data for {player_name}: {e}")

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
        st.info("Fetching live 2025 data from SportsDataverse …")
        df, canonical_name = fetch_player_data(player_name)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Select the correct stat column
    stat_map = {
        "Passing Yards": ["passing_yards", "pass_yards", "pass_yds"],
        "Rushing Yards": ["rushing_yards", "rush_yards", "rush_yds"],
        "Receiving Yards": ["receiving_yards", "rec_yards", "rec_yds"]
    }
    cols = stat_map[stat_type]
    col = next((c for c in cols if c in df.columns), None)
    if not col:
        st.error("No matching stat column found in API data.")
        st.stop()

    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    view = df[["week", col]].rename(columns={col: "value"}).sort_values("week")

    # Chart
    fig = px.bar(
        view, x="week", y="value", text="value",
        title=f"{canonical_name} — {stat_type} (2025 Season)"
    )
    fig.add_hline(y=sportsbook_line, line_color="red", annotation_text="Sportsbook Line")
    st.plotly_chart(fig, use_container_width=True)

    # Probability outputs
    st.subheader("🎯 Baseline Probability")
    col1, col2 = st.columns(2)
    if col1.button("Over"):
        st.success(f"Over Probability: {calc_prob(view['value'], sportsbook_line, 'Over')}%")
    if col2.button("Under"):
        st.warning(f"Under Probability: {calc_prob(view['value'], sportsbook_line, 'Under')}%")

    # Weather adjustment
    st.divider()
    st.subheader("📊 Context-Adjusted Probability")
    weather, temp = fetch_weather(weather_city)
    base = calc_prob(view["value"], sportsbook_line, "Over")
    adj = base
    if weather and "rain" in weather.lower():
        adj -= 8
    if temp and temp < 40:
        adj -= 5
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
st.caption("Data: SportsDataverse NFL (mirror of Pro-Football-Reference) • OpenWeather • Build vA41 (2025)")
