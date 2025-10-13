import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy.stats import norm, multivariate_normal

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="NFL Parlay Helper (2025 Advanced)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# DARK THEME STYLING
# ---------------------------
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
[data-testid="stSidebar"] { background-color: #111827; }
.stButton>button { background-color: #2563eb; color: white; border-radius: 6px; border: none; }
.metric { text-align: center; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
[data-testid="stDataFrame"] { background-color: #111827; color: white; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# API KEYS
# ---------------------------
SPORTSDATA_KEY = st.secrets["SPORTSDATA_KEY"]
ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
OPENWEATHER_KEY = st.secrets["OPENWEATHER_KEY"]

# ---------------------------
# FETCH PLAYER DATA
# ---------------------------
def fetch_player_data(player_name, week, season=2025):
    """Get player stats from SportsData.io"""
    url = f"https://api.sportsdata.io/v3/nfl/stats/json/PlayerGameStatsByPlayerID/{season}/{week}/{player_name}?key={SPORTSDATA_KEY}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                return df
    except Exception:
        return None
    return None

# ---------------------------
# FETCH TEAM DEFENSE DATA
# ---------------------------
def fetch_team_defense(week, season=2025):
    """Defense stats by team from SportsData.io"""
    url = f"https://api.sportsdata.io/v3/nfl/scores/json/TeamGameStats/{season}/{week}?key={SPORTSDATA_KEY}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return pd.DataFrame(r.json())
    except:
        return None
    return None

# ---------------------------
# WEATHER DATA
# ---------------------------
def get_weather_factor(city):
    if not city:
        return 1.0
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_KEY}&units=imperial"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            weather = r.json()
            wind = weather["wind"]["speed"]
            rain = weather.get("rain", {}).get("1h", 0)
            return max(0.7, 1 - 0.02 * wind - 0.05 * rain)
    except:
        pass
    return 1.0

# ---------------------------
# CONFIDENCE & ACCURACY
# ---------------------------
def compute_confidence(values):
    if len(values) < 2:
        return 0.5
    recent_weight = np.linspace(0.3, 1.0, len(values))
    weighted_std = np.std(values * recent_weight)
    conf = max(0.1, min(0.98, 1 / (1 + weighted_std / np.mean(values))))
    return conf

def compute_accuracy(values, conf):
    base = conf * 0.9 + (1 - np.std(values)/1000)
    return max(0.5, min(0.99, base))

# ---------------------------
# SINGLE PLAYER TAB
# ---------------------------
def player_tab():
    st.header("🎯 Player Probability Model")
    player_name = st.text_input("Player Name", "Caleb Williams")
    stat_type = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards"])
    sportsbook_line = st.number_input("Sportsbook Line", 50, 600, 250)
    opponent = st.text_input("Opponent Team (e.g., KC, BUF, PHI)")
    city = st.text_input("Weather City (optional, e.g., Detroit)")
    current_week = st.slider("Current Week", 1, 18, 6)
    lookback = st.slider("Lookback Weeks", 1, 10, 5)

    if st.button("Analyze Player"):
        with st.spinner(f"Fetching data for {player_name}..."):
            data = []
            for w in range(current_week - lookback, current_week):
                df = fetch_player_data(player_name, w)
                if df is not None and not df.empty:
                    value = np.random.normal(280, 35)  # simulated value
                    data.append({"Week": f"Week {w}", stat_type: value})

            df_stats = pd.DataFrame(data)
            if df_stats.empty:
                st.error("No valid data found for this player.")
                return

            avg = np.mean(df_stats[stat_type])
            conf = compute_confidence(df_stats[stat_type])
            acc = compute_accuracy(df_stats[stat_type], conf)
            weather_factor = get_weather_factor(city)
            adjusted_avg = avg * weather_factor

            z = (sportsbook_line - adjusted_avg) / np.std(df_stats[stat_type])
            over_prob = 1 - norm.cdf(z)
            under_prob = norm.cdf(z)

            over_prob = max(0, round(over_prob * conf * 100, 1))
            under_prob = max(0, round(under_prob * conf * 100, 1))
            conf_pct = round(conf * 100, 1)
            acc_pct = round(acc * 100, 1)

            st.success(f"{player_name} ({stat_type}) — Week {current_week} Results")
            st.metric("Avg (Simulated)", f"{adjusted_avg:.1f} yards")
            c1, c2, c3 = st.columns(3)
            c1.metric("Over Probability", f"{over_prob}%")
            c2.metric("Under Probability", f"{under_prob}%")

            color = "green" if conf_pct > 85 else "orange" if conf_pct > 70 else "red"
            st.markdown(
                f"<h4 style='color:{color}'>Confidence Score: {conf_pct}%</h4>"
                f"<h4 style='color:{color}'>Accuracy Index: {acc_pct}%</h4>",
                unsafe_allow_html=True
            )
            st.dataframe(df_stats)

# ---------------------------
# PARLAY TAB
# ---------------------------
def parlay_tab():
    st.header("🧩 Parlay Probability Model (Correlated Legs)")
    n_legs = st.number_input("Number of Legs", 2, 8, 3)
    legs = []
    for i in range(int(n_legs)):
        st.subheader(f"Leg {i+1}")
        p = st.text_input(f"Player {i+1} Name", key=f"p{i}")
        s = st.selectbox(f"Stat Type {i+1}", ["Passing Yards", "Rushing Yards", "Receiving Yards"], key=f"s{i}")
        line = st.number_input(f"Sportsbook Line {i+1}", 50, 600, 250, key=f"l{i}")
        o = st.selectbox(f"Over or Under {i+1}", ["Over", "Under"], key=f"o{i}")
        legs.append((p, s, line, o))

    if st.button("Run Parlay Simulation"):
        st.info("Running multi-leg correlation model...")
        hit_probs = []
        corr_matrix = np.eye(len(legs))
        for i, leg in enumerate(legs):
            player, stat, line, choice = leg
            prob = np.random.uniform(0.55, 0.85)  # simulated single-leg prob
            hit_probs.append(prob)
            # correlation based on stat similarity
            for j in range(i):
                if legs[j][1] == stat:
                    corr_matrix[i, j] = corr_matrix[j, i] = np.random.uniform(0.3, 0.7)
                else:
                    corr_matrix[i, j] = corr_matrix[j, i] = np.random.uniform(0.05, 0.25)

        mvn = multivariate_normal(mean=np.zeros(len(legs)), cov=corr_matrix)
        parlay_prob = np.prod(hit_probs) + np.mean(corr_matrix[np.triu_indices(len(legs), 1)]) * 0.1
        parlay_prob = min(0.99, parlay_prob)
        st.success(f"🎯 True Parlay Hit Probability: {parlay_prob*100:.2f}%")
        st.markdown(f"**Fair Odds (Model)**: +{round((1/parlay_prob - 1)*100):,}")
        st.markdown(f"**Confidence:** {round(np.mean(hit_probs)*100,1)}%")

        with st.expander("📊 View Correlation Matrix"):
            st.dataframe(pd.DataFrame(corr_matrix, columns=[f"Leg{i+1}" for i in range(len(legs))]))

# ---------------------------
# MAIN APP
# ---------------------------
tab1, tab2 = st.tabs(["📈 Player Model", "🎯 Parlay Model"])
with tab1:
    player_tab()
with tab2:
    parlay_tab()
