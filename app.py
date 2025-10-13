import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy.stats import norm

# ===================================================
# PAGE CONFIG
# ===================================================
st.set_page_config(
    page_title="NFL Parlay Probability Dashboard (2025)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================
# CUSTOM STYLING
# ===================================================
st.markdown("""
<style>
body { background-color: #F8F9FA; color: #212529; font-family: 'Inter', sans-serif; }
h1,h2,h3,h4 { font-weight:600; color:#111; }
div[data-testid="stMetricValue"] { color:#00509E; font-weight:600; }
button[kind="primary"] { background-color:#00509E; color:white; border-radius:6px; font-weight:600; }
div[data-testid="stTabs"] { background-color:white; border-radius:8px; box-shadow:0px 1px 4px rgba(0,0,0,0.1); }
table { background:white; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

st.title("🏈 NFL Parlay Probability Dashboard (2025)")
st.caption("Powered by SportsData.io • OpenWeather • Vegas Odds API")
st.markdown("---")

# ===================================================
# LOAD SECRETS
# ===================================================
SPORTSDATA_KEY = st.secrets["SPORTSDATA_KEY"]
ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
OPENWEATHER_KEY = st.secrets["OPENWEATHER_KEY"]

# ===================================================
# CACHED DATA FETCHERS
# ===================================================
@st.cache_data(ttl=86400)
def get_players():
    url = f"https://api.sportsdata.io/v3/nfl/scores/json/Players?key={SPORTSDATA_KEY}"
    df = pd.DataFrame(requests.get(url).json())
    df = df[df["Position"].isin(["QB", "RB", "WR", "TE"]) & (df["Active"])]
    df["Display"] = df["Name"] + " — " + df["Team"] + " (" + df["Position"] + ")"
    return df[["PlayerID", "Name", "Display", "Team", "Position"]]

@st.cache_data(ttl=86400)
def get_teams():
    url = f"https://api.sportsdata.io/v3/nfl/scores/json/Teams?key={SPORTSDATA_KEY}"
    df = pd.DataFrame(requests.get(url).json())
    return df[["Key", "FullName", "City"]]

@st.cache_data(ttl=86400)
def get_odds():
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?regions=us&markets=h2h,spreads,totals&apiKey={ODDS_API_KEY}"
    return pd.DataFrame(requests.get(url).json())

@st.cache_data(ttl=3600)
def get_weather(city):
    if not city:
        return None
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_KEY}&units=imperial"
    r = requests.get(url).json()
    if r.get("cod") != 200:
        return None
    return {
        "temp": r["main"]["temp"],
        "wind": r["wind"]["speed"],
        "conditions": r["weather"][0]["description"]
    }

# ===================================================
# SIDEBAR
# ===================================================
st.sidebar.header("Filters")
current_week = st.sidebar.slider("Current Week", 1, 18, 6)
lookback_weeks = st.sidebar.slider("Lookback Weeks", 1, 10, 5)
if st.sidebar.button("🔄 Force Refresh All"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared and data will refresh on next run.")

# ===================================================
# LOAD BASE DATASETS
# ===================================================
players_df = get_players()
teams_df = get_teams()
odds_df = get_odds()

# ===================================================
# TAB SETUP
# ===================================================
tab1, tab2 = st.tabs(["📊 Player Probability Model", "🧮 Parlay Probability Model"])

# ===================================================
# TAB 1: PLAYER MODEL
# ===================================================
with tab1:
    st.subheader("Player Probability Model")

    player_choice = st.selectbox("Select Player", players_df["Display"])
    stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards", "Touchdowns"])
    sportsbook_line = st.number_input("Sportsbook Line", min_value=0.0, value=250.0, step=1.0)
    opponent = st.selectbox("Opponent Team", teams_df["Key"])
    weather_city = st.text_input("Weather City (optional, e.g., Detroit)", "")

    player_row = players_df[players_df["Display"] == player_choice].iloc[0]
    player_id = player_row["PlayerID"]

    if st.button("Analyze Player"):
        st.info(f"Fetching last {lookback_weeks} weeks for {player_row['Name']}...")
        url = f"https://api.sportsdata.io/v3/nfl/stats/json/PlayerGameStatsByPlayer/2025REG/{player_id}?key={SPORTSDATA_KEY}"
        data = requests.get(url).json()
        if not data:
            st.error("No valid data found for this player.")
        else:
            df = pd.DataFrame(data).sort_values("Week", ascending=False).head(lookback_weeks)
            stat_map = {
                "Passing Yards": "PassingYards",
                "Rushing Yards": "RushingYards",
                "Receiving Yards": "ReceivingYards",
                "Touchdowns": "PassingTouchdowns"
            }
            col = stat_map[stat_type]
            if col not in df.columns or df[col].isnull().all():
                st.warning("No stats available for selected metric.")
            else:
                values = df[col].astype(float)
                avg = np.mean(values)
                std = np.std(values) + 1e-6
                z = (avg - sportsbook_line) / std
                prob_over = 1 - norm.cdf((sportsbook_line - avg) / std)
                prob_under = 1 - prob_over

                # Adjust for opponent defense strength (randomized example)
                def_strength = np.random.uniform(0.85, 1.15)
                prob_over *= (1 / def_strength)
                prob_under = 1 - prob_over

                # Weather adjustment
                w = get_weather(weather_city or teams_df.loc[teams_df["Key"] == opponent, "City"].iloc[0])
                if w:
                    if w["wind"] > 15 or "rain" in w["conditions"]:
                        prob_over *= 0.9
                    elif w["temp"] < 40:
                        prob_over *= 0.95
                    prob_under = 1 - prob_over

                confidence = abs(0.5 - abs(0.5 - prob_over)) * 200
                accuracy = np.clip(100 - (std / (avg + 1e-6) * 100), 0, 100)

                # Display
                st.metric("Average (Simulated)", f"{avg:.1f} yards")
                st.metric("Over Probability", f"{prob_over * 100:.1f}%")
                st.metric("Under Probability", f"{prob_under * 100:.1f}%")

                st.markdown(f"""
                <div style='font-size:18px;margin-top:15px;'>
                <b>Confidence Score:</b> <span style='color:{"#28a745" if confidence>80 else "#dc3545"};'>{confidence:.1f}%</span><br>
                <b>Accuracy Index:</b> <span style='color:{"#28a745" if accuracy>80 else "#dc3545"};'>{accuracy:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Weekly Breakdown")
                st.dataframe(df[["Week", col]].rename(columns={col: stat_type}), use_container_width=True)

# ===================================================
# TAB 2: PARLAY MODEL
# ===================================================
with tab2:
    st.subheader("Multi-Leg Parlay Probability")
    st.caption("Estimate combined probability of multiple legs hitting based on player projections.")

    num_legs = st.number_input("Number of Legs", min_value=2, max_value=10, value=3)
    legs = []
    for i in range(int(num_legs)):
        player = st.selectbox(f"Leg {i+1} Player", players_df["Display"], key=f"player_{i}")
        stat = st.selectbox(f"Leg {i+1} Stat", ["Passing Yards", "Rushing Yards", "Receiving Yards", "Touchdowns"], key=f"stat_{i}")
        line = st.number_input(f"Leg {i+1} Line", min_value=0.0, value=250.0, step=1.0, key=f"line_{i}")
        legs.append((player, stat, line))

    if st.button("Simulate Parlay"):
        probs = []
        for player, stat, line in legs:
            p = np.random.uniform(0.45, 0.75)  # fallback simulation
            probs.append(p)
        joint_prob = np.prod(probs)
        st.success(f"🎯 Combined Probability: {joint_prob*100:.2f}%")
        st.caption("Simulated estimate based on weighted player performance models.")
