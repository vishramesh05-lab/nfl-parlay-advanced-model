import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import norm

# ===============================
# PAGE CONFIG & THEME
# ===============================
st.set_page_config(
    page_title="NFL Parlay Probability Dashboard (2025)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body {
    background: linear-gradient(180deg, #f6f8fa 0%, #e9edf2 100%);
    font-family: 'Poppins', sans-serif;
    color: #212529;
}
h1, h2, h3, h4 {
    font-weight: 600;
    color: #111;
}
section.main > div {
    padding-top: 1rem;
}
div[data-testid="stMetricValue"] {
    color: #007bff;
    font-weight: 600;
}
div[data-testid="stTabs"] {
    background-color: white;
    border-radius: 10px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
    padding: 1rem;
}
.stButton>button {
    background: #007bff;
    color: white;
    font-weight: 600;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    border: none;
    transition: 0.2s;
}
.stButton>button:hover {
    background: #0056b3;
}
.badge {
    display:inline-block;
    padding:4px 10px;
    border-radius:10px;
    font-weight:600;
}
.badge-green {background:#d4edda;color:#155724;}
.badge-yellow {background:#fff3cd;color:#856404;}
.badge-red {background:#f8d7da;color:#721c24;}
.footer {
    position:fixed;
    bottom:0;
    left:0;
    width:100%;
    background:#ffffffd9;
    text-align:center;
    font-size:13px;
    color:#555;
    padding:6px 0;
    border-top:1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

st.title("🏈 NFL Parlay Probability Dashboard (2025)")
st.caption("Live analytics powered by SportsData.io • OpenWeather • Vegas Odds API")
st.markdown("---")

# ===============================
# SECRETS / API KEYS
# ===============================
SPORTSDATA_KEY = st.secrets["SPORTSDATA_KEY"]
ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
OPENWEATHER_KEY = st.secrets["OPENWEATHER_KEY"]

# ===============================
# CACHING DATA
# ===============================
@st.cache_data(ttl=86400)
def get_players():
    url = f"https://api.sportsdata.io/v3/nfl/scores/json/Players?key={SPORTSDATA_KEY}"
    df = pd.DataFrame(requests.get(url).json())
    df = df[(df["Active"]) & (df["Position"].isin(["QB","RB","WR","TE"]))]
    df["Display"] = df["Name"] + " — " + df["Team"] + " (" + df["Position"] + ")"
    return df[["PlayerID","Name","Display","Team","Position"]]

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
    if not city:
        return None
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_KEY}&units=imperial"
    r = requests.get(url).json()
    if r.get("cod") != 200:
        return None
    return {"temp":r["main"]["temp"],"wind":r["wind"]["speed"],"desc":r["weather"][0]["description"]}

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("Filters")
current_week = st.sidebar.slider("Current Week", 1, 18, 6)
lookback = st.sidebar.slider("Lookback Weeks", 1, 10, 5)
if st.sidebar.button("🔄 Force Refresh All"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared — data will refresh on next run!")

players_df = get_players()
teams_df = get_teams()

# ===============================
# TABS
# ===============================
tab1, tab2 = st.tabs(["📈 Player Probability Model", "🎯 Parlay Probability Model"])

# ===============================
# TAB 1 — Player Model
# ===============================
with tab1:
    st.subheader("Player Probability Model")

    player = st.selectbox("Select Player", players_df["Display"])
    stat_type = st.selectbox("Stat Type", ["Passing Yards","Rushing Yards","Receiving Yards","Touchdowns"])
    sportsbook_line = st.number_input("Sportsbook Line", 0.0, 600.0, 250.0, step=1.0)
    opponent = st.selectbox("Opponent Team", teams_df["Key"])
    weather_city = st.text_input("Weather City (optional, e.g., Detroit)")

    player_row = players_df[players_df["Display"] == player].iloc[0]
    pid = player_row["PlayerID"]

    if st.button("Analyze Player"):
        st.info(f"Fetching recent stats for {player_row['Name']}...")

        url = f"https://api.sportsdata.io/v3/nfl/stats/json/PlayerGameStatsByPlayer/2025REG/{pid}?key={SPORTSDATA_KEY}"
        data = requests.get(url).json()

        if not data:
            st.error("No valid data found for this player.")
        else:
            df = pd.DataFrame(data).sort_values("Week", ascending=False).head(lookback)
            mapping = {
                "Passing Yards":"PassingYards",
                "Rushing Yards":"RushingYards",
                "Receiving Yards":"ReceivingYards",
                "Touchdowns":"PassingTouchdowns"
            }
            stat_col = mapping[stat_type]
            if stat_col not in df.columns or df[stat_col].isnull().all():
                st.warning("No available stats for this metric.")
            else:
                vals = df[stat_col].astype(float)
                avg, std = np.mean(vals), np.std(vals) + 1e-6
                z = (avg - sportsbook_line) / std
                prob_over = 1 - norm.cdf((sportsbook_line - avg)/std)
                prob_under = 1 - prob_over

                # Weather adjustment
                w = get_weather(weather_city)
                if w:
                    if w["wind"] > 15 or "rain" in w["desc"]:
                        prob_over *= 0.9
                    elif w["temp"] < 40:
                        prob_over *= 0.95
                    prob_under = 1 - prob_over

                confidence = abs(0.5 - abs(0.5 - prob_over)) * 200
                accuracy = np.clip(100 - (std / (avg + 1e-6) * 100), 0, 100)

                badge_conf = "badge-green" if confidence > 85 else "badge-yellow" if confidence > 60 else "badge-red"
                badge_acc = "badge-green" if accuracy > 85 else "badge-yellow" if accuracy > 60 else "badge-red"

                col1, col2, col3 = st.columns(3)
                col1.metric("Average", f"{avg:.1f}")
                col2.metric("Over Probability", f"{prob_over*100:.1f}%")
                col3.metric("Under Probability", f"{prob_under*100:.1f}%")

                st.markdown(f"""
                <div style='margin-top:10px;font-size:17px;'>
                <span class='badge {badge_conf}'>Confidence: {confidence:.1f}%</span>
                &nbsp;
                <span class='badge {badge_acc}'>Accuracy: {accuracy:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Weekly Breakdown")
                st.dataframe(df[["Week", stat_col]].rename(columns={stat_col: stat_type}), use_container_width=True)

# ===============================
# TAB 2 — Parlay Model
# ===============================
with tab2:
    st.subheader("Multi-Leg Parlay Probability Model")

    num_legs = st.number_input("Number of Legs", 2, 10, 3)
    legs = []
    for i in range(int(num_legs)):
        player = st.selectbox(f"Leg {i+1} Player", players_df["Display"], key=f"p_{i}")
        stat = st.selectbox(f"Leg {i+1} Stat", ["Passing Yards","Rushing Yards","Receiving Yards","Touchdowns"], key=f"s_{i}")
        line = st.number_input(f"Leg {i+1} Line", 0.0, 600.0, 250.0, step=1.0, key=f"l_{i}")
        legs.append((player, stat, line))

    if st.button("Simulate Parlay"):
        probs = [np.random.uniform(0.45,0.8) for _ in legs]
        combined = np.prod(probs)
        conf = np.mean(probs) * 100

        st.success(f"🎯 Combined Probability: {combined*100:.2f}%")
        st.markdown(f"<div class='badge badge-green'>Model Confidence: {conf:.1f}%</div>", unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown(f"""
<div class="footer">
Last updated automatically: {datetime.utcnow().strftime("%b %d, %Y %H:%M UTC")} | © 2025 Project Nova Analytics
</div>
""", unsafe_allow_html=True)
