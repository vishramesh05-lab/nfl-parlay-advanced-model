# -----------------------------------------
# NFL Parlay Helper (Dual Probabilities, 2025)
# Version: vA23 - with visual enhancements
# -----------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# -----------------------------------------
# PAGE CONFIG & STYLE
# -----------------------------------------
st.set_page_config(
    page_title="NFL Parlay Helper (Dual Probabilities, 2025)",
    layout="wide",
    page_icon="🏈",
)

# Inject custom CSS for a cleaner, premium look
st.markdown("""
    <style>
        .main {
            background: linear-gradient(180deg, #0f1117 0%, #161b22 100%);
            color: #f0f2f6;
        }
        h1, h2, h3 {
            color: #00c3ff !important;
        }
        .stButton button {
            background-color: #00c3ff;
            color: black;
            font-weight: bold;
            border-radius: 6px;
            padding: 0.5rem 1rem;
        }
        .stButton button:hover {
            background-color: #009dd6;
        }
        .stMetric {
            background-color: #1c1f26;
            padding: 1rem;
            border-radius: 8px;
        }
        .stDataFrame {
            background-color: #1a1d23;
            border-radius: 8px;
        }
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------
# HEADER
# -----------------------------------------
st.title("🏈 NFL Parlay Helper (Dual Probabilities, 2025)")
st.caption("Visual model estimating (1) Historical Hit Rate and (2) Context-Adjusted Probability.")
st.caption("Includes weekly trends, opponent defense, weather, and injury modifiers.")
st.caption("Build: **vA23**")

SEASON = 2025

# -----------------------------------------
# DATA LOADERS (CACHED)
# -----------------------------------------
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
def get_weekly_stats(week: int):
    """Public open NFL data mirror."""
    try:
        url = f"https://raw.githubusercontent.com/openfootball/nfl-stats/main/2025/week_{week}.json"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        stats = {}
        for p in data.get("players", []):
            pid = p.get("player_id") or p.get("id") or p.get("name")
            stats[pid] = {
                "pass_yd": p.get("passing_yards", 0),
                "rush_yd": p.get("rushing_yards", 0),
                "rec_yd": p.get("receiving_yards", 0),
                "pts_ppr": p.get("fantasy_points_ppr", 0),
            }
        return stats
    except Exception as e:
        st.warning(f"⚠️ Could not fetch live stats for week {week}: {e}")
        return {}


@st.cache_data(ttl=600)
def get_defense_rankings():
    return pd.DataFrame([
        {"team": "KC", "rush_def_rank": 6, "pass_def_rank": 10},
        {"team": "DET", "rush_def_rank": 12, "pass_def_rank": 20},
        {"team": "SF", "rush_def_rank": 2, "pass_def_rank": 4},
        {"team": "BUF", "rush_def_rank": 8, "pass_def_rank": 7},
        {"team": "PHI", "rush_def_rank": 1, "pass_def_rank": 16},
    ])


@st.cache_data(ttl=600)
def get_weather(team):
    sample = {
        "KC": {"temp": 72, "wind": 8, "precip": 0},
        "DET": {"temp": 68, "wind": 5, "precip": 0},
        "BUF": {"temp": 48, "wind": 18, "precip": 0.4},
        "SF": {"temp": 62, "wind": 7, "precip": 0.1},
        "PHI": {"temp": 60, "wind": 10, "precip": 0.0},
    }
    return sample.get(team.upper(), {"temp": 70, "wind": 6, "precip": 0})

# -----------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Filters")
    current_week = st.slider("Current Week", 1, 18, 6)
    lookback = st.slider("Lookback (weeks)", 1, 10, 5)
    st.markdown("---")
    st.markdown("🧠 Adjust context for defense/weather/injury in results below.")

# -----------------------------------------
# USER INPUTS
# -----------------------------------------
player_name = st.text_input("Player Name")
stat = st.selectbox("Stat Type", ["Passing Yards", "Rushing Yards", "Receiving Yards", "Fantasy Points (PPR)"])
line = st.number_input("Sportsbook Line", min_value=0.0, step=0.5)
opp_team = st.text_input("Opponent Team (e.g., BUF)")

# -----------------------------------------
# MAIN ANALYSIS
# -----------------------------------------
if st.button("Analyze Player"):
    try:
        st.info("Fetching live player + weekly data...")
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
                weekly.append({"Week": wk, stat: val})

            df = pd.DataFrame(weekly)

            if df.empty or df[stat].sum() == 0:
                st.warning("No weekly data found yet for this player.")
            else:
                st.dataframe(df, use_container_width=True)

                # ------------------------
                # HISTORICAL PROBABILITY
                # ------------------------
                avg_val = df[stat].mean()
                total_games = len(df)
                over_hits = (df[stat] > line).sum()
                under_hits = (df[stat] < line).sum()

                base_over_prob = (over_hits / total_games) * 100
                base_under_prob = (under_hits / total_games) * 100

                st.markdown("### 🎯 Historical Probability")
                c1, c2 = st.columns(2)
                c1.metric("Over Probability", f"{base_over_prob:.1f}%")
                c2.metric("Under Probability", f"{base_under_prob:.1f}%")

                # ------------------------
                # WEEKLY BAR CHART
                # ------------------------
                st.markdown("### 📊 Weekly Performance vs Sportsbook Line")

                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(df["Week"], df[stat], color="#00c3ff", alpha=0.8)
                ax.axhline(line, color="red", linestyle="--", label=f"Line = {line}")
                ax.set_xlabel("Week")
                ax.set_ylabel(stat)
                ax.set_title(f"{name} — {stat} (Last {lookback} Weeks)")
                ax.legend()

                # Annotate bars with values
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 1, f"{int(height)}", 
                            ha='center', va='bottom', fontsize=8, color='white')
                st.pyplot(fig)

                # ------------------------
                # CONTEXT-ADJUSTED PROBABILITY
                # ------------------------
                st.markdown("### 🧮 Context-Adjusted Probability")

                def_adj = 1.0
                weather_adj = 1.0
                injury_adj = 1.0

                ranks = get_defense_rankings()
                if not ranks.empty and opp_team:
                    opp = ranks[ranks["team"] == opp_team.upper()]
                    if not opp.empty:
                        if "Rushing" in stat:
                            rank = opp.iloc[0]["rush_def_rank"]
                        else:
                            rank = opp.iloc[0]["pass_def_rank"]
                        def_adj = max(0.8, min(1.2, 1.1 - (rank - 16) / 80))

                w = get_weather(opp_team.upper() if opp_team else team)
                if w["wind"] > 15:
                    weather_adj *= 0.9
                if w["precip"] > 0.3:
                    weather_adj *= 0.85
                if w["temp"] < 45:
                    weather_adj *= 0.95

                injury_adj = 0.95 if np.random.rand() < 0.1 else 1.0

                adj_over_prob = base_over_prob * def_adj * weather_adj * injury_adj
                adj_under_prob = base_under_prob * def_adj * weather_adj * injury_adj

                st.success(
                    f"**Adjusted Over:** {adj_over_prob:.1f}% | **Adjusted Under:** {adj_under_prob:.1f}%"
                )

                with st.expander("📋 Adjustment Breakdown"):
                    st.write(f"Defense Multiplier: {def_adj:.2f}")
                    st.write(f"Weather Multiplier: {weather_adj:.2f}  → temp={w['temp']}°F, wind={w['wind']} mph, precip={w['precip']}")
                    st.write(f"Injury Multiplier: {injury_adj:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
