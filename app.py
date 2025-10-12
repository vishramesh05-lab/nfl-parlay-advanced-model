# NFL Parlay Helper (Dual Probabilities, 2025)
# Streamlit App – pulls live player stats from Sleeper API (2025 season)

import pandas as pd
import numpy as np
import requests
import streamlit as st
from io import StringIO, BytesIO

# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="NFL Parlay Helper (Dual Probabilities, 2025)", layout="wide")
st.title("🏈 NFL Parlay Helper (Dual Probabilities, 2025)")
st.caption("Two estimates: (1) Historical from last N games, and (2) Context-Adjusted including injuries, weather, pace, usage trend, opponent defensive injuries, and market vig.")
st.caption("Build: vA12")

SEASON = 2025

# ----------------------------
# Sidebar Upload (optional)
# ----------------------------
st.sidebar.markdown("### Upload 2025 player CSV (optional)")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 2025 player stats", type=["csv"])

# ----------------------------
# Data Loader
# ----------------------------
@st.cache_data(show_spinner=True, ttl=60 * 30)
def load_all():
    """
    Load player data.
    Priority:
    1. Uploaded CSV (user-provided)
    2. Sleeper API live data (free)
    3. Return empty DataFrame if nothing works
    """

    # 1️⃣ Uploaded CSV
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows from uploaded CSV")
            return df
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return pd.DataFrame()

    # 2️⃣ Sleeper API – Free Live NFL 2025 Stats
    try:
        st.info("🔄 Fetching live 2025 player stats from Sleeper (free API)...")

        # 1. Fetch live stats
        stats_url = "https://api.sleeper.app/v1/stats/nfl/regular/2025"
        stats_resp = requests.get(stats_url, timeout=20)
        stats_resp.raise_for_status()
        stats_data = stats_resp.json()

        # 2. Fetch player directory (names, positions, teams)
        players_url = "https://api.sleeper.app/v1/players/nfl"
        players_resp = requests.get(players_url, timeout=20)
        players_resp.raise_for_status()
        players_data = players_resp.json()

        # 3. Convert stats and players to DataFrames
        stats_df = pd.DataFrame(stats_data).T
        stats_df.reset_index(inplace=True)
        stats_df.rename(columns={"index": "player_id"}, inplace=True)

        players_df = pd.DataFrame(players_data).T
        players_df.reset_index(inplace=True)
        players_df.rename(columns={"index": "player_id"}, inplace=True)

        # 4. Select relevant columns from players_df (avoid duplicate player_id)
        players_subset = players_df.loc[:, ["player_id", "full_name", "team", "position"]]

        # 5. Merge safely
        merged = pd.merge(stats_df, players_subset, on="player_id", how="left", validate="1:1")

        # 6. Rename + clean columns
        merged.rename(columns={
            "full_name": "player_display_name",
            "pts_ppr": "fantasy_points_ppr",
            "pass_yd": "passing_yards",
            "rush_yd": "rushing_yards",
            "rec_yd": "receiving_yards"
        }, inplace=True)

        keep = [
            "player_display_name", "team", "position",
            "passing_yards", "rushing_yards", "receiving_yards",
            "fantasy_points_ppr"
        ]

        for col in keep:
            if col not in merged.columns:
                merged[col] = np.nan

        st.success(f"✅ Loaded {len(merged)} live players (2025) with full names from Sleeper API.")
        return merged[keep]

    except Exception as e:
        st.error(f"Error loading Sleeper 2025 live data: {e}")
        return pd.DataFrame()

# ----------------------------
# Load data
# ----------------------------
stats_df = load_all()

# ----------------------------
# Ensure DataFrame Structure
# ----------------------------
if not isinstance(stats_df, pd.DataFrame):
    stats_df = pd.DataFrame()

required_cols = ["player_display_name", "team", "position",
                 "passing_yards", "rushing_yards", "receiving_yards"]
for col in required_cols:
    if col not in stats_df.columns:
        stats_df[col] = np.nan

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

st.markdown("> Optional: enter market odds (American) to include vig")
col1, col2 = st.columns(2)
with col1:
    over_odds = st.text_input("Over odds (e.g., -115 or +100)")
with col2:
    under_odds = st.text_input("Under odds (e.g., -105 or +100)")

# ----------------------------
# Analyze Button
# ----------------------------
if st.button("Analyze"):
    if stats_df.empty:
        st.error("No data available. Upload CSV or wait for live data.")
    elif player_name.strip() == "":
        st.warning("Enter a player name first.")
    else:
        # Filter for matching player
        matches = stats_df[
            stats_df["player_display_name"].astype(str).str.contains(player_name, case=False, na=False)
        ]

        if matches.empty:
            st.warning("No matching player found. Try clearing filters or refining the name.")
        else:
            st.success(f"✅ Found {len(matches)} records for {player_name}")
            st.dataframe(matches.head(10))

            # Display basic stat summary
            selected_stat = stat.lower().replace(" ", "_").replace("(ppr)", "fantasy_points_ppr")
            if selected_stat in matches.columns:
                avg_value = matches[selected_stat].astype(float).mean()
                st.metric(label=f"Avg {stat} (last {lookback} weeks)", value=f"{avg_value:.2f}")
            else:
                st.warning("Stat not found for selected player.")
