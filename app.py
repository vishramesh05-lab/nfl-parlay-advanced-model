# ======================================================
# NFL Parlay Helper v63 — Dual Probability + Defense Adjustment
# Author: Vishvin Ramesh
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="NFL Parlay Helper (v63)", page_icon="🏈", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>🏈 NFL Parlay Helper — 2025 Dual Model Edition</h1>",
    unsafe_allow_html=True,
)
st.caption("Player + Defense + Injury adjusted probability model — SportsData.io + OpenWeather")
st.caption("Build v63 • by Vishvin Ramesh")

# -------------------------------
# API KEYS
# -------------------------------
SPORTSDATA_KEY = st.secrets.get("SPORTSDATA_KEY")
WEATHER_KEY = st.secrets.get("OPENWEATHER_KEY")

if not SPORTSDATA_KEY:
    st.error("⚠️ Add SPORTSDATA_KEY in Streamlit Secrets.")
    st.stop()

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("⚙️ Filters")
week = st.sidebar.slider("Current Week", 1, 18, 6)
lookback = st.sidebar.slider("Lookback Weeks", 1, 10, 5)
debug = st.sidebar.checkbox("Enable Debug Mode")

# -------------------------------
# USER INPUT
# -------------------------------
player_name = st.text_input("Player Name", placeholder="e.g., Josh Allen")
stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
sportsbook_line = st.number_input("Sportsbook Line", value=250.0, step=1.0)
opponent_team = st.text_input("Opponent Team (e.g., KC, BUF, PHI)", "")
weather_city = st.text_input("Weather City (optional)", "")

st.divider()

# -------------------------------
# FETCH HELPERS
# -------------------------------

def safe_fetch_json(url):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def fetch_week_stats(week_num):
    url = f"https://api.sportsdata.io/v3/nfl/stats/json/PlayerGameStatsByWeek/2025REG/{week_num}?key={SPORTSDATA_KEY}"
    data = safe_fetch_json(url)
    if not data:
        raise Exception(f"No data for week {week_num}")
    df = pd.DataFrame(data)
    for c in ["FirstName","LastName","Name","Team","PassingYards","RushingYards","ReceivingYards"]:
        if c not in df.columns: df[c] = np.nan
    df["full_name"] = (df["FirstName"].fillna("")+" "+df["LastName"].fillna("")).str.strip().str.lower()
    df.loc[df["full_name"].eq(""),"full_name"] = df["Name"].astype(str).str.lower()
    return df

def fetch_defense_stats(team):
    """Fetch basic defensive averages for opponent"""
    if not team: return None
    url = f"https://api.sportsdata.io/v3/nfl/scores/json/TeamSeasonStats/2025REG?key={SPORTSDATA_KEY}"
    data = safe_fetch_json(url)
    if not data: return None
    df = pd.DataFrame(data)
    team = team.upper()
    df = df[df["Team"].eq(team)]
    if df.empty: return None
    return {
        "PassYdsAllowed": df.iloc[0].get("OpponentPassingYards", 0),
        "RushYdsAllowed": df.iloc[0].get("OpponentRushingYards", 0),
        "PointsAllowed": df.iloc[0].get("PointsAllowed", 0)
    }

def fetch_injuries(team):
    url = f"https://api.sportsdata.io/v3/nfl/scores/json/Injuries/2025REG?key={SPORTSDATA_KEY}"
    data = safe_fetch_json(url)
    if not data: return 0
    df = pd.DataFrame(data)
    injured = df[df["Team"].eq(team.upper())]
    return len(injured)

def fetch_weather(city):
    if not city or not WEATHER_KEY: return None
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=imperial"
    js = safe_fetch_json(url)
    if js: return js["main"]["temp"]
    return None

# -------------------------------
# MAIN ANALYSIS
# -------------------------------
if st.button("📊 Analyze Player"):
    if not player_name.strip():
        st.warning("Enter player name.")
        st.stop()

    pkey = player_name.lower().strip()
    st.info(f"Fetching last {lookback} weeks for {player_name}…")

    all_data = []
    for wk in range(max(1, week - lookback + 1), week + 1):
        try:
            df = fetch_week_stats(wk)
            df["Week"] = wk
            all_data.append(df)
        except Exception as e:
            st.warning(f"Skipping Week {wk}: {e}")

    if not all_data:
        st.error("No data found.")
        st.stop()

    df = pd.concat(all_data, ignore_index=True)
    df = df[df["full_name"].str.contains(pkey, na=False)]
    if df.empty:
        st.error(f"No player data for {player_name}.")
        st.stop()

    col_map = {"Passing Yards":"PassingYards","Rushing Yards":"RushingYards","Receiving Yards":"ReceivingYards"}
    stat = col_map[stat_type]
    vals = df[stat].fillna(0).astype(float)

    mean = vals.mean()
    prob_over = np.mean(vals > sportsbook_line) * 100
    prob_under = 100 - prob_over

    # --- DEFENSE ADJUSTMENT ---
    defense = fetch_defense_stats(opponent_team)
    def_adj = 0
    if defense:
        if stat_type == "Passing Yards":
            def_adj = (defense["PassYdsAllowed"]/250 - 1)*10
        elif stat_type == "Rushing Yards":
            def_adj = (defense["RushYdsAllowed"]/120 - 1)*10
        prob_over += def_adj

    # --- INJURY ADJUSTMENT ---
    injuries = fetch_injuries(opponent_team)
    if injuries > 3: prob_over += 5
    elif injuries == 0: prob_over -= 2

    # --- WEATHER ADJUSTMENT ---
    temp = fetch_weather(weather_city)
    if temp:
        if temp < 40: prob_over -= 5
        elif temp > 85: prob_over -= 3

    prob_over = float(np.clip(prob_over, 0, 100))
    prob_under = 100 - prob_over

    # -------------------------------
    # DISPLAY
    # -------------------------------
    st.success(f"✅ {player_name} ({stat_type}) — Adjusted Probabilities")
    df_show = pd.DataFrame({
        "Metric": [
            "Average Yards",
            "Probability Over",
            "Probability Under",
            "Defense Adj (%)",
            "Injuries (Affected)",
            "Weather Temp (°F)"
        ],
        "Value": [
            f"{mean:.1f}",
            f"{prob_over:.1f} %",
            f"{prob_under:.1f} %",
            f"{def_adj:+.1f}" if defense else "N/A",
            injuries,
            f"{temp:.1f}" if temp else "N/A"
        ]
    })
    st.dataframe(df_show, use_container_width=True)

    if debug:
        st.subheader("Raw Weekly Data")
        st.dataframe(df[["Week","Team",stat]])
