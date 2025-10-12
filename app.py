# NFL Parlay Helper (Dual Probabilities, 2025)
# Streamlit app using live Sleeper API for free NFL stats

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO, BytesIO

# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="NFL Parlay Helper (Dual Probabilities, 2025)", layout="wide")
st.title("🏈 NFL Parlay Helper (Dual Probabilities, 2025)")
st.caption("Two estimates: (1) Historical from last N games, and (2) Context-Adjusted including injuries, weather, pace, usage trend, opponent defensive injuries, and market vig.")
st.caption("Build: vA17")

SEASON = 2025

# ----------------------------
# Sidebar Upload (optional)
# ----------------------------
st.sidebar.markdown("### Upload 2025 player CSV (optional)")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 2025 player stats", type=["csv"])

# ----------------------------
# Helper Function – Deduplicate Columns
# ----------------------------
def make_unique_columns(columns):
    seen = {}
    result = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
    return result

# ----------------------------
# Data Loader
# ----------------------------
@st.cache_data(show_spinner=True, ttl=60 * 30)
def load_all():
    """Load player data from uploaded CSV or Sleeper API."""
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

        stats_url = "https://api.sleeper.app/v1/stats/nfl/regular/2025"
        stats_resp = requests.get(stats_url, timeout=20)
        stats_resp.raise_for_status()
        stats_data = stats_resp.json()

        stats_df = pd.DataFrame(stats_data).T.reset_index()
        stats_df.rename(columns={"index": "player_id"}, inplace=True)
        stats_df.columns = make_unique_columns(stats_df.columns)
        stats_df = stats_df.drop_duplicates(subset=["player_id"], keep="first")

        players_url = "https://api.sleeper.app/v1/players/nfl"
        players_resp = requests.get(players_url, timeout=20)
        players_resp.raise_for_status()
        players_data = players_resp.json()

        players_df = pd.DataFrame(players_data).T.reset_index()
        players_df.rename(columns={"index": "player_id"}, inplace=True)
        players_df.columns = make_unique_columns(players_df.columns)

        players_subset = players_df.loc[:, ["player_id", "full_name", "team", "position"]]
        players_subset = players_subset.drop_duplicates(subset=["player_id"])

        merged = pd.merge(stats_df, players_subset, on="player_id", how="left")

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

        st.success(f"✅ Loaded {len(merged)} unique live players (2025) from Sleeper API.")
        return merged[keep]

    except Exception as e:
        st.error(f"Error loading Sleeper 2025 live data: {e}")
        return pd.DataFrame()

# ----------------------------
# Load Season Data
# ----------------------------
stats_df = load_all()

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
# Analyze Button (Per-Week Breakdown)
# ----------------------------
if st.button("Analyze"):
    if stats_df.empty:
        st.error("No data available. Upload CSV or wait for live data.")
    elif player_name.strip() == "":
        st.warning("Enter a player name first.")
    else:
        matches = stats_df[
            stats_df["player_display_name"].astype(str).str.contains(player_name, case=False, na=False)
        ]

        if matches.empty:
            st.warning("No matching player found. Try refining the name.")
        else:
            st.success(f"✅ Found {len(matches)} record(s) for {player_name}")

            # --- Fetch weekly stats directly from Sleeper API ---
            try:
                weekly_rows = []
                for week_num in range(1, current_week + 1):
                    url = f"https://api.sleeper.app/v1/stats/nfl/regular/{week_num}"
                    resp = requests.get(url, timeout=20)
                    resp.raise_for_status()
                    data = resp.json()

                    for pid, pdata in data.items():
                        # Extract player name and stat dynamically
                        player_id = str(pid)
                        player_stats = pdata if isinstance(pdata, dict) else {}
                        if not player_stats:
                            continue

                        # Match on name (from metadata if available)
                        name_key = player_stats.get("player", "")
                        if player_name.lower() in str(name_key).lower():
                            # Select correct stat key
                            if stat == "Passing Yards":
                                key = "pass_yd"
                            elif stat == "Rushing Yards":
                                key = "rush_yd"
                            elif stat == "Receiving Yards":
                                key = "rec_yd"
                            elif stat == "Fantasy Points (PPR)":
                                key = "pts_ppr"
                            else:
                                key = None

                            if key and key in player_stats:
                                weekly_rows.append({
                                    "week": week_num,
                                    stat: player_stats.get(key, 0)
                                })

                if weekly_rows:
                    week_df = pd.DataFrame(weekly_rows).sort_values("week")
                    st.dataframe(week_df)
                    avg_value = week_df[stat].astype(float).mean()
                    st.metric(label=f"Avg {stat} (Weeks 1–{current_week})", value=f"{avg_value:.2f}")
                else:
                    st.warning(f"No weekly {stat} data found for {player_name} yet.")

            except Exception as e:
                st.error(f"Error fetching weekly data: {e}")
