# NFL Parlay Helper (Dual Probabilities, 2025)
# Streamlit app using live Sleeper API + probability model for sportsbook over/under

import streamlit as st
import pandas as pd
import numpy as np
import requests

# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="NFL Parlay Helper (Dual Probabilities, 2025)", layout="wide")
st.title("🏈 NFL Parlay Helper (Dual Probabilities, 2025)")
st.caption("Live data + probability model: calculates chance of a player hitting their sportsbook line.")
st.caption("Build: vA19")

SEASON = 2025

# ----------------------------
# Sidebar Upload (optional)
# ----------------------------
st.sidebar.markdown("### Upload 2025 player CSV (optional)")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 2025 player stats", type=["csv"])

# ----------------------------
# Helper Functions
# ----------------------------
@st.cache_data(show_spinner=True, ttl=60 * 30)
def get_all_players():
    """Fetch full player dictionary from Sleeper (includes names + IDs)."""
    url = "https://api.sleeper.app/v1/players/nfl"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data).T.reset_index().rename(columns={"index": "player_id"})
    df["full_name"] = df["full_name"].fillna("")
    df["team"] = df["team"].fillna("")
    df["position"] = df["position"].fillna("")
    return df[["player_id", "full_name", "team", "position"]]

@st.cache_data(show_spinner=True, ttl=60 * 10)
def get_weekly_stats(week: int):
    """Fetch per-week stats for all players."""
    url = f"https://api.sleeper.app/v1/stats/nfl/regular/{week}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.markdown("### Filters")
current_week = st.sidebar.slider("Current week for matchup context", 1, 18, 6)
lookback = st.sidebar.slider("Lookback (N weeks)", 1, 10, 5)

# ----------------------------
# Main Inputs
# ----------------------------
player_name = st.text_input("Player name")
stat = st.selectbox("Stat", ["Passing Yards", "Rushing Yards", "Receiving Yards", "Fantasy Points (PPR)"])
line = st.number_input("Sportsbook line", min_value=0.0, step=0.5)

# ----------------------------
# Analyze
# ----------------------------
if st.button("Analyze"):
    try:
        st.info("🔍 Fetching player info and weekly stats...")
        players_df = get_all_players()

        # Match player by name
        match = players_df[players_df["full_name"].str.contains(player_name, case=False, na=False)]
        if match.empty:
            st.warning("No matching player found. Try refining the name.")
        else:
            player_id = match.iloc[0]["player_id"]
            player_team = match.iloc[0]["team"]
            player_pos = match.iloc[0]["position"]

            st.success(f"✅ Found {match.iloc[0]['full_name']} ({player_team}, {player_pos})")

            weekly_data = []
            for week in range(max(1, current_week - lookback + 1), current_week + 1):
                stats = get_weekly_stats(week)
                if player_id in stats:
                    pdata = stats[player_id]
                    if stat == "Passing Yards":
                        val = pdata.get("pass_yd", 0)
                    elif stat == "Rushing Yards":
                        val = pdata.get("rush_yd", 0)
                    elif stat == "Receiving Yards":
                        val = pdata.get("rec_yd", 0)
                    elif stat == "Fantasy Points (PPR)":
                        val = pdata.get("pts_ppr", 0)
                    else:
                        val = 0
                    weekly_data.append({"week": week, stat: val})
                else:
                    weekly_data.append({"week": week, stat: 0})

            week_df = pd.DataFrame(weekly_data)
            if week_df.empty:
                st.warning(f"No weekly {stat} data found for {player_name} yet.")
            else:
                st.dataframe(week_df)
                avg_val = week_df[stat].mean()
                st.metric(label=f"Avg {stat} (Weeks {max(1, current_week - lookback + 1)}–{current_week})",
                          value=f"{avg_val:.2f}")

                # ----------------------------
                # Probability Calculation
                # ----------------------------
                st.markdown("### 🎯 Probability Calculator")

                over_button = st.button("📈 Over")
                under_button = st.button("📉 Under")

                if over_button or under_button:
                    total_games = len(week_df)
                    if total_games == 0:
                        st.warning("No data available for probability calculation.")
                    else:
                        if over_button:
                            hits = (week_df[stat] > line).sum()
                            prob = (hits / total_games) * 100
                            st.success(f"Probability of **Over {line} {stat.lower()}** = **{prob:.1f}%** ({hits}/{total_games} games hit)")
                        elif under_button:
                            hits = (week_df[stat] < line).sum()
                            prob = (hits / total_games) * 100
                            st.info(f"Probability of **Under {line} {stat.lower()}** = **{prob:.1f}%** ({hits}/{total_games} games hit)")

    except Exception as e:
        st.error(f"Error fetching data: {e}")
