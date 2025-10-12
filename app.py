import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="NFL Parlay Helper (Dual Probabilities, 2025)", layout="wide")
st.title("🏈 NFL Parlay Helper (Dual Probabilities, 2025)")
st.caption("Estimate 1: Historical hit rate. Estimate 2: Context-adjusted (defense, weather, injuries).")
st.caption("Build: vA21")

SEASON = 2025

# ----------------------------
# Cached Data Loaders
# ----------------------------
@st.cache_data(ttl=1800)
def get_all_players():
    url = "https://api.sleeper.app/v1/players/nfl"
    r = requests.get(url, timeout=30)
    df = pd.DataFrame(r.json()).T.reset_index().rename(columns={"index": "player_id"})
    df["full_name"] = df["full_name"].fillna("")
    df["team"] = df["team"].fillna("")
    df["position"] = df["position"].fillna("")
    return df[["player_id", "full_name", "team", "position"]]

@st.cache_data(ttl=600)
def get_weekly_stats(week):
    url = f"https://api.sleeper.app/v1/stats/nfl/regular/{week}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def get_defense_rankings():
    """Fetch simple defensive ranks from a public JSON feed (example placeholder)."""
    try:
        resp = requests.get("https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams", timeout=20)
        data = resp.json()
        # fallback example ranks (stub)
        return pd.DataFrame([
            {"team": "KC", "rush_def_rank": 6, "pass_def_rank": 10},
            {"team": "DET", "rush_def_rank": 12, "pass_def_rank": 20},
            {"team": "SF", "rush_def_rank": 2, "pass_def_rank": 4},
        ])
    except Exception:
        return pd.DataFrame([
            {"team": "KC", "rush_def_rank": 6, "pass_def_rank": 10},
            {"team": "DET", "rush_def_rank": 12, "pass_def_rank": 20},
            {"team": "SF", "rush_def_rank": 2, "pass_def_rank": 4},
        ])

@st.cache_data(ttl=600)
def get_weather(team):
    """Mock weather lookup (you can plug a real API like OpenWeather later)."""
    sample = {
        "KC": {"temp": 72, "wind": 8, "precip": 0},
        "DET": {"temp": 68, "wind": 5, "precip": 0},
        "BUF": {"temp": 48, "wind": 18, "precip": 0.4},
    }
    return sample.get(team, {"temp": 70, "wind": 6, "precip": 0})

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.markdown("### Filters")
current_week = st.sidebar.slider("Current week", 1, 18, 6)
lookback = st.sidebar.slider("Lookback (weeks)", 1, 10, 5)

# ----------------------------
# Inputs
# ----------------------------
player_name = st.text_input("Player name")
stat = st.selectbox("Stat", ["Passing Yards", "Rushing Yards", "Receiving Yards", "Fantasy Points (PPR)"])
line = st.number_input("Sportsbook line", min_value=0.0, step=0.5)
opp_team = st.text_input("Opponent team (e.g., BUF)")

# ----------------------------
# Core Logic
# ----------------------------
if st.button("Analyze"):
    try:
        st.info("Fetching live data...")
        players = get_all_players()
        match = players[players["full_name"].str.contains(player_name, case=False, na=False)]

        if match.empty:
            st.warning("No matching player found.")
        else:
            pid = str(match.iloc[0]["player_id"])
            team = str(match.iloc[0]["team"])
            pos = str(match.iloc[0]["position"])
            name = str(match.iloc[0]["full_name"])
            st.success(f"✅ Found {name} ({team}, {pos})")

            # Pull weekly stats
            weekly = []
            for wk in range(max(1, current_week - lookback + 1), current_week + 1):
                s = get_weekly_stats(wk)
                val = 0
                if pid in s:
                    p = s[pid]
                    if stat == "Passing Yards":
                        val = p.get("pass_yd", 0)
                    elif stat == "Rushing Yards":
                        val = p.get("rush_yd", 0)
                    elif stat == "Receiving Yards":
                        val = p.get("rec_yd", 0)
                    elif stat == "Fantasy Points (PPR)":
                        val = p.get("pts_ppr", 0)
                weekly.append({"week": wk, stat: val})
            df = pd.DataFrame(weekly)
            st.dataframe(df)

            avg_val = df[stat].mean()
            total_games = len(df)
            base_hits = (df[stat] > line).sum()
            base_prob = (base_hits / total_games) * 100
            st.metric("Baseline Over Probability", f"{base_prob:.1f}%")

            # ----------------------------
            # Context-adjusted probability
            # ----------------------------
            st.markdown("### 🧠 Context-Adjusted Probability")

            def_adj = 1.0
            weather_adj = 1.0
            injury_adj = 1.0

            # Defensive rank adjustment
            ranks = get_defense_rankings()
            if not ranks.empty and opp_team:
                opp = ranks[ranks["team"] == opp_team.upper()]
                if not opp.empty:
                    if "Rushing" in stat:
                        rank = opp.iloc[0]["rush_def_rank"]
                    else:
                        rank = opp.iloc[0]["pass_def_rank"]
                    def_adj = max(0.8, min(1.2, 1.1 - (rank - 16) / 80))

            # Weather adjustment
            w = get_weather(opp_team.upper() if opp_team else team)
            if w["wind"] > 15:
                weather_adj *= 0.9
            if w["precip"] > 0.3:
                weather_adj *= 0.85
            if w["temp"] < 45:
                weather_adj *= 0.95

            # Injury adjustment (placeholder: assume random)
            injury_adj = 0.95 if np.random.rand() < 0.1 else 1.0

            adj_prob = base_prob * def_adj * weather_adj * injury_adj
            adj_prob = min(100, max(0, adj_prob))

            st.success(
                f"Adjusted Over Probability: **{adj_prob:.1f}%** "
                f"(Def {def_adj:.2f} × Weather {weather_adj:.2f} × Inj {injury_adj:.2f})"
            )

    except Exception as e:
        st.error(f"Error: {e}")
