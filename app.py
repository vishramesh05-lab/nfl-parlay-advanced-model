import streamlit as st
import pandas as pd
from utilities import (
    VERSION,
    normalize_columns,
    detect_schema,
    build_week_matrix,
    compute_usage_features,
    rank_recommendations_by_pos,
    export_recommendations_csv
)

st.set_page_config(page_title="Player Parlay – Usage Picker", layout="wide")
st.title("🏈 Player Parlay – Usage-Based Picks (Single CSV)")
st.caption(VERSION)

with st.sidebar:
    st.header("Upload")
    csv = st.file_uploader("FantasyPros 2025 Offense Snap Counts CSV", type=["csv"])
    st.markdown(
        "- Works out-of-the-box with **FantasyPros_Fantasy_Football_2025_Offense_Snap_Counts.csv**\n"
        "- No API keys, no extra files"
    )
    st.divider()
    st.header("Settings")
    recs_per_pos = st.slider("Top picks per position", 3, 20, 8)
    min_weeks = st.slider("Min games played (to consider)", 1, 8, 2)
    min_snap_pct = st.slider("Min last-3-weeks snap %", 0, 100, 25)
    trend_weight = st.slider("Trend weight (WoW delta)", 0.0, 2.0, 1.0, 0.1)
    l3_weight = st.slider("Last-3-weeks weight", 0.0, 2.0, 1.0, 0.1)
    routes_weight = st.slider("Routes weight (if available)", 0.0, 2.0, 0.6, 0.1)
    targets_weight = st.slider("Targets weight (if available)", 0.0, 2.0, 0.8, 0.1)
    st.caption("Tip: If your CSV doesn't have routes/targets, the model falls back to snap% only.")

if not csv:
    st.info("Upload your **FantasyPros Offense Snap Counts** CSV to begin.")
    st.stop()

try:
    df_raw = pd.read_csv(csv)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.subheader("Preview")
st.dataframe(df_raw.head(15), use_container_width=True)

# Normalize columns and detect schema
df = normalize_columns(df_raw.copy())
schema = detect_schema(df)

with st.expander("Detected schema"):
    st.json(schema)

# Build a week x player matrix for snaps and snap%
wk_df = build_week_matrix(df, schema)

with st.expander("Weekly matrix (first 100 rows)"):
    st.dataframe(wk_df.head(100), use_container_width=True)

# Compute features: last-3-weeks avg, WoW trend, usage score
feat_df = compute_usage_features(
    wk_df,
    routes_weight=routes_weight,
    targets_weight=targets_weight,
    l3_weight=l3_weight,
    trend_weight=trend_weight
)

# Filters
mask = (feat_df["games_count"] >= min_weeks) & (feat_df["l3_snap_pct"] >= min_snap_pct)
feat_filtered = feat_df.loc[mask].copy()

st.subheader("Usage features (filtered)")
st.dataframe(
    feat_filtered[[
        "player","team","pos","week_last","games_count",
        "l3_snap_pct","wow_snap_pct","l3_routes","l3_targets","usage_score"
    ]].sort_values("usage_score", ascending=False),
    use_container_width=True
)

# Recommendations by position (Over/Under style lists)
st.subheader("🎯 Recommendations")
recs = rank_recommendations_by_pos(feat_filtered, per_pos=recs_per_pos)

cols = st.columns(2)
with cols[0]:
    st.markdown("#### 📈 Over candidates (high usage + rising trend)")
    st.dataframe(recs["over"], use_container_width=True)
with cols[1]:
    st.markdown("#### 📉 Under candidates (low usage + falling trend)")
    st.dataframe(recs["under"], use_container_width=True)

# Download
csv_bytes = export_recommendations_csv(recs)
st.download_button(
    "💾 Download picks (CSV)",
    data=csv_bytes,
    file_name="player_parlay_usage_picks.csv",
    mime="text/csv",
    use_container_width=True
)

st.caption(
    "Note: This app is usage-signal driven (snap%, routes, targets if present). "
    "It doesn't need lines/odds; use these lists to shop for player props you like."
)
