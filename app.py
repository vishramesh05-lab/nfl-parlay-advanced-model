# ======================================================
# NFL Parlay Helper (2025 - Live SportsData.io Edition)
# Author: Vishvin Ramesh
# Version: vFinal | Live 2025 API + Weather Adjustments
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="NFL Parlay Helper (2025 - Live Edition)",
    page_icon="🏈",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;'>🏈 NFL Parlay Helper (Dual Probabilities, 2025)</h1>",
    unsafe_allow_html=True
)
st.caption("Live player data powered by SportsData.io + OpenWeather (Free Tiers)")
st.caption("Build vFinal | by Vishvin Ramesh")

# ===============================
# API KEYS
# ===============================
SPORTSDATA_KEY = st.secrets.get("SPORTSDATA_KEY", None)
WEATHER_KEY = st.secrets.get("OPENWEATHER_KEY", None)

if not SPORTSDATA_KEY:
    st.error("⚠️ Missing SportsData.io API key. Add SPORTSDATA_KEY in Streamlit secrets.")
    st.stop()

# ===============================
# SIDEBAR FILTERS
# ===============================
st.sidebar.header("⚙️ Filters")
week = st.sidebar.slider("Current Week", 1, 18, 6)
lookback = st.sidebar.slider("Lookback (weeks)", 1, 10, 5)
debug = st.sidebar.checkbox("Enable Debug Mode")

# ===============================
# MAIN INPUTS
# ===============================
player_name = st.text_input("Player Name", placeholder="e.g. Patrick Mahomes")
stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
sportsbook_line = st.number_input("Sportsbook Line", value=100.0, step=1.0)
opponent_team = st.text_input("Opponent Team (e.g., KC, BUF, PHI)", "")
weather_city = st.text_input("Weather City (optional, e.g., Detroit)", "")

st.divider()

# ===============================
# DATA FETCH FUNCTION
# ===============================
@st.cache_data(ttl=900)
def fetch_nfl_player_stats(week):
    url = f"https://api.sportsdata.io/v3/nfl/stats/json/PlayerGameStatsByWeek/2025REG/{week}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_KEY}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise ValueError(f"SportsData API error {resp.status_code}: {resp.text}")
    return pd.DataFrame(resp.json())

# ===============================
# WEATHER DATA FUNCTION
# ===============================
def fetch_weather(city):
    if not city or not WEATHER_KEY:
        return None
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=imperial"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.json()["main"]["temp"]
    except:
        return None
    return None

# ===============================
# MAIN LOGIC
# ===============================
if st.button("🔄 Reload / Refresh Data"):
    st.cache_data.clear()
    st.success("Data cache cleared — reloading latest stats.")

if st.button("📊 Analyze Player"):
    if not player_name.strip():
        st.warning("Please enter a player name.")
        st.stop()

    with st.spinner(f"Fetching 2025 NFL data for Week {week}..."):
        try:
            df = fetch_nfl_player_stats(week)
        except Exception as e:
            st.error(f"❌ Error fetching SportsData.io: {e}")
            st.stop()

    if df.empty:
        st.warning("No data returned for this week.")
        st.stop()

    # Normalize names for easier matching
    df["Name"] = df["Name"].str.lower().str.strip()
    player = player_name.lower().strip()

    # Find player data
    player_data = df[df["Name"].str.contains(player, na=False)]
    if player_data.empty:
        st.error(f"No data found for '{player_name}'. Try full name as listed on SportsData.io.")
        st.stop()

    # Stat extraction
    if stat_type == "Passing Yards":
        col = "PassingYards"
    elif stat_type == "Rushing Yards":
        col = "RushingYards"
    else:
        col = "ReceivingYards"

    player_stats = player_data[col].astype(float)
    mean_val = np.mean(player_stats)
    prob_over = np.mean(player_stats > sportsbook_line) * 100
    prob_under = 100 - prob_over

    # Weather adjustment
    temp = fetch_weather(weather_city)
    if temp:
        if temp < 40:
            prob_over -= 5
        elif temp > 85:
            prob_over -= 3
        prob_over = max(min(prob_over, 100), 0)
        prob_under = 100 - prob_over

    # ===============================
    # OUTPUT
    # ===============================
    st.success(f"✅ {player_name} ({stat_type}) — Week {week} Results")
    st.write(f"Average {stat_type}: **{mean_val:.1f} yards**")
    st.write(f"Line: **{sportsbook_line} yards**")

    st.metric("📈 Probability Over", f"{prob_over:.1f}%")
    st.metric("📉 Probability Under", f"{prob_under:.1f}%")

    # Distribution chart
    fig = px.histogram(player_stats, nbins=10, title=f"{player_name} — {stat_type} Distribution (Week {week})",
                       labels={'value': stat_type}, color_discrete_sequence=['#00cc96'])
    st.plotly_chart(fig, use_container_width=True)

    if debug:
        st.subheader("Raw Data (Debug Mode)")
        st.dataframe(player_data.head(20))
