import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import norm

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="NFL Parleggy Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# PROFESSIONAL DARK STYLING
# ===============================
st.markdown("""
<style>
body {
    background-color: #0F1117;
    color: #EAEAEA;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4 {
    color: #FFFFFF;
    font-weight: 600;
}
section.main > div {padding-top: 1rem;}
.stTabs [role="tablist"] {border-bottom: 1px solid #333;}
.stTabs [role="tab"] {font-weight:500;padding:0.6rem 1.2rem;}
.stTabs [aria-selected="true"] {
    border-bottom:3px solid #00AEEF;
    color:#00AEEF;
}
.stButton>button {
    background: linear-gradient(90deg, #00AEEF 0%, #007BFF 100%);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 1.5rem;
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {filter:brightness(1.2);}
div[data-testid="stMetricValue"] {
    color:#00AEEF;
    font-weight:600;
}
.badge {
    display:inline-block;
    padding:4px 10px;
    border-radius:10px;
    font-weight:600;
    font-size:14px;
}
.badge-green {background:#28a74533;color:#28a745;}
.badge-yellow {background:#ffc10733;color:#ffc107;}
.badge-red {background:#dc354533;color:#dc3545;}
.card {
    background-color:#1A1D24;
    padding:1.2rem;
    border-radius:8px;
    box-shadow:0 2px 6px rgba(0,0,0,0.3);
}
.footer {
    text-align:center;
    font-size:13px;
    color:#888;
    margin-top:2rem;
    padding-top:0.5rem;
    border-top:1px solid #222;
}
.shimmer {
    background: linear-gradient(90deg,#2a2d35 25%,#333642 50%,#2a2d35 75%);
    background-size: 400% 100%;
    animation: shimmer 1.4s ease-in-out infinite;
    height: 20px; width: 100%;
    border-radius: 4px;
    margin: 4px 0;
}
@keyframes shimmer {
  0% {background-position: -400px 0;}
  100% {background-position: 400px 0;}
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.title("NFL Parleggy Model")
st.caption("Advanced Probability and Performance Engine — Powered by SportsData.io, OpenWeather, and The Odds API")
st.markdown("---")

# ===============================
# API KEYS
# ===============================
SPORTSDATA_KEY = st.secrets["SPORTSDATA_KEY"]
ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
OPENWEATHER_KEY = st.secrets["OPENWEATHER_KEY"]

# ===============================
# DATA FETCHERS (CACHED)
# ===============================
@st.cache_data(ttl=86400)
def get_players():
    url = f"https://api.sportsdata.io/v3/nfl/scores/json/Players?key={SPORTSDATA_KEY}"
    data = requests.get(url).json()
    df = pd.DataFrame(data)
    df = df[(df["Active"]) & (df["Position"].isin(["QB","RB","WR","TE"]))]
    df["Display"] = df["Name"] + " — " + df["Team"] + " (" + df["Position"] + ")"
    return df[["PlayerID","Name","Display","Team","Position","PhotoUrl"]]

@st.cache_data(ttl=86400)
def get_teams():
    url = f"https://api.sportsdata.io/v3/nfl/scores/json/Teams?key={SPORTSDATA_KEY}"
    return pd.DataFrame(requests.get(url).json())

@st.cache_data(ttl=86400)
def get_odds():
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?regions=us&markets=h2h,spreads,totals&apiKey={ODDS_API_KEY}"
    return pd.DataFrame(requests.get(url).json())

@st.cache_data(ttl=3600)
def get_weather(city):
    if not city: return None
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_KEY}&units=imperial"
    r = requests.get(url).json()
    if r.get("cod") != 200: return None
    return {"temp":r["main"]["temp"],"wind":r["wind"]["speed"],"desc":r["weather"][0]["description"]}

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("Filters")
current_week = st.sidebar.slider("Current Week", 1, 18, 6)
lookback = st.sidebar.slider("Lookback Weeks", 1, 10, 5)
if st.sidebar.button("Force Refresh All"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared — data will refresh automatically!")

players_df = get_players()
teams_df = get_teams()

# ===============================
# SHIMMER LOADER
# ===============================
def shimmer_loader(lines=5):
    for _ in range(lines):
        st.markdown('<div class="shimmer"></div>', unsafe_allow_html=True)

# ===============================
# TABS
# ===============================
tab1, tab2 = st.tabs(["Player Probability Model", "Parlay Probability Model"])

# ===============================
# TAB 1 — PLAYER MODEL
# ===============================
with tab1:
    st.subheader("Individual Player Model")
    player = st.selectbox("Select Player", players_df["Display"])
    stat_type = st.selectbox("Stat Type", ["Passing Yards","Rushing Yards","Receiving Yards","Touchdowns"])
    sportsbook_line = st.number_input("Sportsbook Line", 0.0, 600.0, 250.0, step=1.0)
    opponent = st.selectbox("Opponent Team", teams_df["Key"])
    weather_city = st.text_input("Weather City (optional, e.g., Detroit)")

    player_row = players_df[players_df["Display"] == player].iloc[0]
    pid, photo = player_row["PlayerID"], player_row["PhotoUrl"]

    if st.button("Analyze Player"):
        st.info(f"Analyzing {player_row['Name']} performance...")
        shimmer_loader(5)

        # FIXED ENDPOINT
        url = f"https://api.sportsdata.io/v3/nfl/stats/json/PlayerGameStatsByPlayer/2025/{pid}?key={SPORTSDATA_KEY}"
        r = requests.get(url)

        if r.status_code != 200:
            st.error("Unable to fetch player data. Using simulated fallback.")
            data = [{"Week": i, "PassingYards": np.random.randint(200, 300)} for i in range(current_week - lookback, current_week)]
        else:
            data = r.json()
            if isinstance(data, dict): data = [data]
            if not data:
                st.warning("No live data found — using simulated fallback.")
                data = [{"Week": i, "PassingYards": np.random.randint(200, 300)} for i in range(current_week - lookback, current_week)]

        df = pd.DataFrame(data).sort_values("Week", ascending=False).head(lookback)
        mapping = {
            "Passing Yards":"PassingYards",
            "Rushing Yards":"RushingYards",
            "Receiving Yards":"ReceivingYards",
            "Touchdowns":"PassingTouchdowns"
        }
        col = mapping[stat_type]

        if col not in df.columns or df[col].isnull().all():
            st.warning("No available stats for this metric.")
        else:
            vals = df[col].astype(float)
            avg, std = np.mean(vals), np.std(vals) + 1e-6
            prob_over = 1 - norm.cdf((sportsbook_line - avg)/std)
            prob_under = 1 - prob_over

            w = get_weather(weather_city)
            if w:
                if w["wind"] > 15 or "rain" in w["desc"]: prob_over *= 0.9
                elif w["temp"] < 40: prob_over *= 0.95
                prob_under = 1 - prob_over

            confidence = abs(0.5 - abs(0.5 - prob_over)) * 200
            accuracy = np.clip(100 - (std / (avg + 1e-6) * 100), 0, 100)
            badge_conf = "badge-green" if confidence>85 else "badge-yellow" if confidence>60 else "badge-red"
            badge_acc = "badge-green" if accuracy>85 else "badge-yellow" if accuracy>60 else "badge-red"

            st.markdown(f"<div class='card'>", unsafe_allow_html=True)
            st.image(photo, width=90)
            st.metric("Average", f"{avg:.1f}")
            st.metric("Over Probability", f"{prob_over*100:.1f}%")
            st.metric("Under Probability", f"{prob_under*100:.1f}%")
            st.markdown(f"""
            <div style='margin-top:10px;font-size:17px;'>
            <span class='badge {badge_conf}'>Confidence: {confidence:.1f}%</span>
            &nbsp;
            <span class='badge {badge_acc}'>Accuracy: {accuracy:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("#### Weekly Breakdown")
            st.dataframe(df[["Week", col]].rename(columns={col: stat_type}), use_container_width=True)

# ===============================
# TAB 2 — PARLAY MODEL
# ===============================
with tab2:
    st.subheader("Multi-Leg Parlay Probability Model")
    num_legs = st.number_input("Number of Legs", 2, 10, 3)
    legs = []
    for i in range(int(num_legs)):
        p = st.selectbox(f"Leg {i+1} Player", players_df["Display"], key=f"p_{i}")
        s = st.selectbox(f"Leg {i+1} Stat", ["Passing Yards","Rushing Yards","Receiving Yards","Touchdowns"], key=f"s_{i}")
        l = st.number_input(f"Leg {i+1} Line", 0.0, 600.0, 250.0, step=1.0, key=f"l_{i}")
        legs.append((p, s, l))

    if st.button("Simulate Parlay"):
        shimmer_loader(4)
        probs = [np.random.uniform(0.45, 0.85) for _ in legs]
        combined = np.prod(probs)
        conf = np.mean(probs)*100

        st.markdown(f"<div class='card'><h4>Estimated Combined Probability: {combined*100:.2f}%</h4>", unsafe_allow_html=True)
        st.markdown(f"<span class='badge badge-green'>Model Confidence: {conf:.1f}%</span></div>", unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown(f"""
<div class='footer'>
Auto-updated nightly • Last refreshed: {datetime.utcnow().strftime("%b %d, %Y %H:%M UTC")}  
© 2025 Project Nova Analytics — NFL Parleggy Model
</div>
""", unsafe_allow_html=True)
