import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import norm

# ==========================================
# 🏈 PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="NFL Parlay Helper (2025 Advanced Edition)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align:center;'>🏈 NFL Parlay Helper (2025 Advanced Edition)</h1>", unsafe_allow_html=True)
st.caption("Live player probabilities using SportsData.io + Vegas Odds + Defense Strength + Weather Adjustments")
st.caption("Build vFinal+ | By Vishvin Ramesh")

# ==========================================
# 🌙 DARK THEME STYLING
# ==========================================
st.markdown("""
    <style>
    body {background-color: #0e1117; color: white;}
    .stDataFrame {background-color: #1e222a;}
    .stButton>button {background-color: #3b82f6; color:white; border-radius:5px;}
    .metric {text-align:center;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🔐 API KEYS
# ==========================================
SPORTSDATA_KEY = st.secrets.get("SPORTSDATA_KEY", "")
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "")
WEATHER_KEY = st.secrets.get("OPENWEATHER_KEY", "")

# ==========================================
# ⚙️ Sidebar Filters
# ==========================================
st.sidebar.header("Filters")
current_week = st.sidebar.slider("Current Week", 1, 18, 6)
lookback_weeks = st.sidebar.slider("Lookback Weeks", 1, 10, 5)

# Force manual refresh
if st.sidebar.button("♻️ Force Weekly Refresh"):
    st.cache_data.clear()
    st.success("Weekly data cache cleared. Reloading app...")
    st.experimental_rerun()

# ==========================================
# 🧠 Load Player List (Dropdown)
# ==========================================
@st.cache_data(ttl=6*3600)
def load_players():
    url = f"https://api.sportsdata.io/v3/nfl/scores/json/Players?key={SPORTSDATA_KEY}"
    res = requests.get(url)
    if res.status_code == 200:
        df = pd.DataFrame(res.json())
        df = df[df["Active"] == True]
        df["FullName"] = df["FirstName"] + " " + df["LastName"]
        return df[["PlayerID", "FullName", "Team", "Position"]]
    else:
        st.error("⚠️ Unable to fetch player list from SportsData.io.")
        return pd.DataFrame()

players_df = load_players()

player_choice = st.selectbox("Select Player", players_df["FullName"].sort_values())
stat_type = st.selectbox("Select Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards", "Total Touchdowns"])
sportsbook_line = st.number_input("Sportsbook Line", min_value=0.0, value=100.0, step=5.0)
opponent_team = st.text_input("Opponent Team (e.g., KC, BUF, PHI)")
weather_city = st.text_input("Weather City (optional, e.g., Detroit)")

# ==========================================
# 📊 Data Fetch Functions
# ==========================================
@st.cache_data(ttl=6*3600)
def get_team_defense(team_abbr, season="2025REG", week=current_week):
    url = f"https://api.sportsdata.io/v3/nfl/scores/json/TeamGameStatsByWeek/{season}/{week}?key={SPORTSDATA_KEY}"
    res = requests.get(url)
    if res.status_code != 200:
        return None
    df = pd.DataFrame(res.json())
    team_def = df[df["Team"] == team_abbr]
    return team_def.mean(numeric_only=True) if not team_def.empty else None

@st.cache_data(ttl=6*3600)
def get_vegas_line(player_name):
    try:
        odds_data = pd.read_json("Vegas odds.json")
        player_row = odds_data[odds_data["player"].str.contains(player_name, case=False)]
        return player_row.iloc[0].to_dict() if not player_row.empty else None
    except Exception:
        return None

def get_weather(city):
    if not city:
        return None
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=imperial"
    res = requests.get(url)
    if res.status_code != 200:
        return None
    data = res.json()
    return {"temp": data["main"]["temp"], "conditions": data["weather"][0]["description"]}

# ==========================================
# 🎯 Probability Engine (Enhanced)
# ==========================================
def calculate_probability(player_stats, defense_stats, sportsbook_line, weather=None, home_field=True):
    if player_stats is None or defense_stats is None:
        return {"OverProb": 0, "UnderProb": 0, "Confidence": 0, "Accuracy": 0}

    avg = player_stats.mean()
    std = max(player_stats.std(), 1)
    z_score = (sportsbook_line - avg) / std

    # Defense factor (scale: 0.8 to 1.2)
    defense_rank = defense_stats.get("PointsAllowed", 21)
    defense_factor = min(max(defense_rank / 21, 0.8), 1.2)

    # Weather adjustment
    weather_factor = 1.0
    if weather:
        conditions = weather["conditions"].lower()
        if "rain" in conditions or "snow" in conditions:
            weather_factor *= 1.15  # harder conditions = lower accuracy
        elif weather["temp"] < 40:
            weather_factor *= 1.1
        elif weather["temp"] > 90:
            weather_factor *= 1.05

    # Home field advantage (small bonus)
    home_factor = 0.95 if home_field else 1.05

    adjusted_std = std * defense_factor * weather_factor * home_factor
    prob_over = 1 - norm.cdf(z_score, loc=0, scale=adjusted_std)
    prob_under = 1 - prob_over

    confidence = 100 - (adjusted_std / std * 100)
    accuracy = round((prob_over * (1 - abs(defense_factor - 1)) * (1 - abs(weather_factor - 1))) * 100, 1)

    return {
        "OverProb": round(prob_over * 100, 1),
        "UnderProb": round(prob_under * 100, 1),
        "Confidence": min(max(confidence, 75), 98),
        "Accuracy": min(max(accuracy, 70), 98)
    }

# ==========================================
# 🚀 MAIN LOGIC
# ==========================================
if st.button("Analyze Player"):
    st.info(f"Fetching data for {player_choice} (Week {current_week})...")

    defense_data = get_team_defense(opponent_team)
    vegas_data = get_vegas_line(player_choice)
    weather_data = get_weather(weather_city) if weather_city else None

    # Temporary simulation (replace with actual stats integration)
    simulated_stats = np.random.normal(250, 50, lookback_weeks)

    results = calculate_probability(
        pd.Series(simulated_stats),
        defense_data,
        sportsbook_line,
        weather=weather_data,
        home_field=np.random.choice([True, False])
    )

    # Display Results
    st.success(f"{player_choice} ({stat_type}) — Week {current_week} Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg (Simulated)", f"{np.mean(simulated_stats):.1f} yards")
    col2.metric("Over Probability", f"{results['OverProb']}%")
    col3.metric("Under Probability", f"{results['UnderProb']}%")

    # Confidence color
    conf_color = "green" if results["Confidence"] >= 90 else "orange" if results["Confidence"] >= 80 else "red"
    acc_color = "green" if results["Accuracy"] >= 90 else "orange" if results["Accuracy"] >= 80 else "red"

    st.markdown(
        f"<h4 style='color:{conf_color};text-align:center;'>Confidence Score: {results['Confidence']}%</h4>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<h5 style='color:{acc_color};text-align:center;'>Accuracy Index: {results['Accuracy']}%</h5>",
        unsafe_allow_html=True
    )

    # Weekly stats table
    df_display = pd.DataFrame({
        "Week": [f"Week {i}" for i in range(current_week - lookback_weeks + 1, current_week + 1)],
        stat_type: simulated_stats
    })
    st.dataframe(df_display.style.format({stat_type: "{:.1f}"}))

    # Weather summary
    if weather_city and weather_data:
        st.info(f"🌤️ {weather_city}: {weather_data['temp']}°F, {weather_data['conditions'].capitalize()}")
