import streamlit as st
import pandas as pd
import numpy as np
from utilities import (
    TRAINABLE_TARGETS,
    parse_and_validate_schema,
    train_model,
    predict_for_upcoming,
    american_to_implied_prob,
    VERSION_STRING,
)

st.set_page_config(page_title="Team Odds Model – Moneyline / ATS / Totals", layout="wide")

# ===== Sidebar =====
st.sidebar.title("Team Odds Predictor")
st.sidebar.caption(VERSION_STRING)

st.sidebar.markdown("### 1) Upload data")
hist_file = st.sidebar.file_uploader("Historical games CSV (for training)", type=["csv"])
future_file = st.sidebar.file_uploader("Upcoming games CSV (for predictions)", type=["csv"])

st.sidebar.markdown("### 2) Choose targets")
targets = st.sidebar.multiselect(
    "Select prediction targets",
    TRAINABLE_TARGETS,
    default=TRAINABLE_TARGETS,  # ["moneyline","spread","over_under"]
)

st.sidebar.markdown("### 3) Model / CV")
model_type = st.sidebar.selectbox("Model type", ["log_reg", "random_forest", "gbm"])
cv_folds = st.sidebar.slider("Cross-validation folds", 3, 10, 5)
calibrate = st.sidebar.checkbox("Calibrate probabilities (Platt scaling)", value=True)

st.sidebar.markdown("### 4) Odds settings (optional but recommended)")
st.sidebar.caption(
    "If your upcoming CSV includes American odds columns, the app will compute implied probability and EV."
)

st.sidebar.divider()
st.sidebar.markdown("**Notes / Tips**")
st.sidebar.info(
    "- Targets expected in historical data:\n"
    "  - moneyline: `result_home_win` ∈ {0,1}\n"
    "  - spread (ATS): `result_home_cover` ∈ {0,1}\n"
    "  - over_under: `result_over` ∈ {0,1}\n\n"
    "- Odds columns in upcoming data (optional):\n"
    "  - `home_ml`, `away_ml` (American odds)\n"
    "  - `spread` (float, e.g., -3.5 means home favored), `home_spread_odds`, `away_spread_odds`\n"
    "  - `total` (game total), `over_odds`, `under_odds`"
)

st.title("🏈 Team Odds Model — Moneyline / ATS / Over–Under")
st.caption("Upload data → train → get probabilities, recommended picks, and expected value (EV).")

# ===== Body =====
if not hist_file:
    st.warning("Upload a **Historical games CSV** to begin.")
    st.stop()

try:
    df_hist_raw = pd.read_csv(hist_file)
except Exception as e:
    st.error(f"Could not read historical CSV: {e}")
    st.stop()

with st.expander("Preview: Historical data (first 20 rows)"):
    st.dataframe(df_hist_raw.head(20), use_container_width=True)

# Validate / split features & targets from historical data
parsed = parse_and_validate_schema(df_hist_raw, expected_targets=targets)
if parsed.errors:
    st.error("Schema issues found in the historical CSV:")
    for err in parsed.errors:
        st.write(f"• {err}")
    st.stop()

st.success("Historical data looks good.")

# Train one model per target
trained_models = {}
metrics_blocks = []
train_button = st.button("🚀 Train Models", type="primary", use_container_width=True)

if train_button:
    with st.spinner("Training models…"):
        for target in targets:
            model_obj = train_model(
                df_hist_raw,
                target=target,
                model_type=model_type,
                cv_folds=cv_folds,
                calibrate=calibrate,
            )
            trained_models[target] = model_obj
            metrics_blocks.append((target, model_obj["metrics"]))

    st.success("Models trained.")

    cols = st.columns(len(metrics_blocks)) if metrics_blocks else [st]
    for i, (target, m) in enumerate(metrics_blocks):
        with cols[i]:
            st.markdown(f"### Metrics — **{target}**")
            st.json(m, expanded=False)

# Predict (if both trained and upcoming present)
if not future_file:
    st.info("Upload an **Upcoming games CSV** to generate predictions and EV.")
    st.stop()

try:
    df_future_raw = pd.read_csv(future_file)
except Exception as e:
    st.error(f"Could not read upcoming CSV: {e}")
    st.stop()

with st.expander("Preview: Upcoming games data (first 20 rows)"):
    st.dataframe(df_future_raw.head(20), use_container_width=True)

if not trained_models:
    st.warning("Train your models first, then re-run predictions.")
    st.stop()

predict_button = st.button("📈 Predict Upcoming Games", type="secondary", use_container_width=True)

if predict_button:
    all_outputs = []
    with st.spinner("Scoring upcoming games…"):
        for target in targets:
            res_df = predict_for_upcoming(trained_models[target], df_future_raw.copy(), target)
            res_df.insert(0, "target", target)
            all_outputs.append(res_df)

    if all_outputs:
        results_df = pd.concat(all_outputs, ignore_index=True)

        # Order columns nicely if present
        preferred_cols = [
            "target",
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "prediction_side",
            "model_prob",
            "book_implied_prob",
            "edge",
            "ev_per_$100",
            # moneyline / spread / total odds:
            "home_ml", "away_ml",
            "spread", "home_spread_odds", "away_spread_odds",
            "total", "over_odds", "under_odds",
        ]
        exist_cols = [c for c in preferred_cols if c in results_df.columns]
        other_cols = [c for c in results_df.columns if c not in exist_cols]
        results_df = results_df[exist_cols + other_cols]

        st.markdown("## Results")
        st.dataframe(results_df, use_container_width=True)

        # Download
        csv = results_df.to_csv(index=False)
        st.download_button(
            "💾 Download Predictions CSV",
            data=csv,
            file_name="team_odds_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Quick EV summary by target
        st.markdown("### 📊 EV Summary (Top opportunities)")
        for t in targets:
            sub = results_df[results_df["target"] == t].copy()
            if "ev_per_$100" in sub.columns and not sub.empty:
                st.markdown(f"**{t}** — top edges")
                show = sub.sort_values("ev_per_$100", ascending=False).head(10)
                st.dataframe(show[[
                    c for c in ["game_date", "home_team", "away_team", "prediction_side",
                                "model_prob", "book_implied_prob", "edge", "ev_per_$100"]
                    if c in show.columns
                ]], use_container_width=True)
            else:
                st.caption(f"{t}: no odds present to compute EV.")
