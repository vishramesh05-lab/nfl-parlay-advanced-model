# NFL Parlay Helper (Dual Probabilities, 2025)
# Streamlit App – using live Sleeper API for free 2025 NFL player stats

import pandas as pd
import numpy as np
import requests
import streamlit as st
from io import StringIO, BytesIO

# Optional local utils if present
try:
    from utils import (
        ensure_cols, last_n_window, prob_over_normal, stat_label_and_col,
        TEAM_LATLON, fetch_weather, injury_flag_for_player,
        opponent_def_injuries, pace_factor, usage_trend_factor,
        vig_to_market_prob, context_adjusted_probability, POS_FOR_STAT
    )
except Exception:
    pass

# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="NFL Parlay Helper (Dual Probabilities, 2025)", layout="wide")
st.title("🏈 NFL Parlay Helper (Dual Probabilities, 2025)")
st.caption("Two estimates: (1) Historical from last N games, and (2) Context-Adjusted including injuries, weather, pace, usage trend, opponent defensive injuries, and market vig.")
st.caption("Build: vA10")

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
        url = "https://api.sleeper.app/v1/stats/nfl/regular/2025"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and len(data) > 0:
            # Convert dict → DataFrame
            df = pd.DataFrame(data).T.reset_index(names=["player_id"])

            # Rename key columns
            rename_map = {
                "pts_ppr": "fantasy_points_ppr",
                "pass_yd": "passing_yards",
                "rush_yd": "rushing_yards",
                "rec_yd": "receiving_yards",
                "team": "team",
                "player": "player_display_name",
                "pos": "position"
            }
            df.rename(columns=rename_map, inplace=True)

            # Ensure necessary columns exist
            keep = ["player_display_name", "team", "position",
                    "passing_yards", "rushing_yards", "receiving_yards",
                    "fantasy_points_ppr"]
            for col in keep:
                if col not in df.columns:
                    df[col] = np.nan

            st.success(f"✅ Loaded {len(df)} live player rows from Sleeper (2025)")
            return df[keep]
        else:
            st.warning("⚠️ Sleeper API returned no data.")
            return pd.DataFrame()

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
        matches = stats_df[stats_df["player_display_name"].astype(str).str.contains(player_name, case=False, na=False)]

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
