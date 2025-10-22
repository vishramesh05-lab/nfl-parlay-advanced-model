import streamlit as st
import pandas as pd
import numpy as np

from utilities import (
    VERSION,
    ingest_files_to_games_df,
    build_home_game_rows,
    compute_targets_from_lines,
    TRAINABLE_TARGETS,
    train_model,
    predict_proba_for_homerow,
    american_to_implied_prob,
    expected_value_per_100,
    make_homerow_from_rolling_avgs,
)

st.set_page_config(page_title="Team Odds – ML / ATS / O/U", layout="wide")
st.title("🏈 Team Odds — Moneyline / ATS / Over–Under (JSON/CSV uploads)")
st.caption(VERSION)

with st.sidebar:
    st.header("1) Upload raw game files")
    st.caption("Upload one or more **team-game** files (JSON or CSV). Use your `6.json`, `6 (1).json`, etc.")
    files = st.file_uploader("Team game logs", type=["json", "csv"], accept_multiple_files=True)

    st.divider()
    st.header("2) Training settings")
    model_type = st.selectbox("Model type", ["log_reg", "random_forest", "gbm"])
    cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
    calibrate = st.checkbox("Calibrate probabilities (Platt)", value=True)
    last_n_games = st.slider("Rolling window for matchup simulator (games)", 2, 10, 4)

    st.divider()
    st.header("3) Optional EV inputs (for simulator)")
    home_ml = st.text_input("Home Moneyline (e.g., -135 or +155)", value="")
    away_ml = st.text_input("Away Moneyline (e.g., +115)", value="")
    spread = st.text_input("Home Spread (e.g., -3.5 means home favored)", value="")
    ou_total = st.text_input("Total (e.g., 45.5)", value="")
    home_spread_odds = st.text_input("Home spread odds (e.g., -110)", value="")
    away_spread_odds = st.text_input("Away spread odds (e.g., -110)", value="")
    over_odds = st.text_input("Over odds (e.g., -105)", value="")
    under_odds = st.text_input("Under odds (e.g., -115)", value="")

if not files:
    st.info("Upload at least one **team-game JSON/CSV** to begin.")
    st.stop()

# ---------- Ingest ----------
with st.spinner("Parsing and unifying uploads…"):
    raw_df, notes = ingest_files_to_games_df(files)

st.success(f"Ingested {len(raw_df):,} team-game rows from {len(files)} file(s).")
with st.expander("Ingest notes / preview"):
    st.write(notes)
    st.dataframe(raw_df.head(25), use_container_width=True)

# Canonical home rows (one per game from the home team's perspective)
home_rows = build_home_game_rows(raw_df)

st.subheader("Canonical home rows (one per game)")
st.dataframe(home_rows.head(25), use_container_width=True)

# Derive targets from lines (if present)
home_rows = compute_targets_from_lines(home_rows)

missing_targets = [t for t in TRAINABLE_TARGETS if t not in home_rows.columns]
if missing_targets:
    st.warning(
        "Some targets are missing. The app will auto-compute them if your data has "
        "`Score`, `OpponentScore`, `OverUnder`, and team-`PointSpread` (per-team line). "
        "Otherwise, that target will be skipped."
    )

# ---------- Train ----------
st.markdown("## Train models")
available_targets = [t for t in TRAINABLE_TARGETS if t in home_rows.columns]
if not available_targets:
    st.error("No targets available. Need at least one of: moneyline, spread, over_under.")
    st.stop()

train_btn = st.button("🚀 Train", type="primary", use_container_width=True)
trained = {}

if train_btn:
    with st.spinner("Training…"):
        for t in available_targets:
            trained[t] = train_model(
                df_home=home_rows,
                target=t,
                model_type=model_type,
                cv_folds=cv_folds,
                calibrate=calibrate,
            )
    st.success("Models trained.")
    cols = st.columns(len(trained) if trained else 1)
    for i, (t, mobj) in enumerate(trained.items()):
        with cols[i]:
            st.markdown(f"**{t}** CV metrics")
            st.json(mobj["metrics"], expanded=False)

if not trained:
    st.info("Train the models to enable the Matchup Simulator.")
    st.stop()

# ---------- Matchup simulator ----------
st.markdown("## 🧪 Matchup Simulator (no CSV needed)")
teams = sorted(set(home_rows["home_team"]).union(set(home_rows["away_team"])))
c1, c2, c3 = st.columns([1,1,1])
with c1:
    sim_home = st.selectbox("Home team", teams, index=0 if teams else None)
with c2:
    sim_away = st.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)
with c3:
    sim_games = st.slider("Use last N games per team", 2, 10, last_n_games)

run_sim = st.button("📈 Predict this matchup", use_container_width=True)

if run_sim:
    with st.spinner("Building feature row from rolling averages and scoring…"):
        # Build a synthetic home-row using rolling averages for each team
        synth_row = make_homerow_from_rolling_avgs(
            home_rows_all=home_rows,
            team_home=sim_home,
            team_away=sim_away,
            last_n=sim_games,
            spread_text=spread,
            total_text=ou_total,
            home_ml_text=home_ml,
            away_ml_text=away_ml,
            home_spread_odds_text=home_spread_odds,
            away_spread_odds_text=away_spread_odds,
            over_odds_text=over_odds,
            under_odds_text=under_odds,
        )

        st.markdown("#### Synthetic matchup row (features fed to models)")
        st.dataframe(pd.DataFrame([synth_row]).T, use_container_width=True)

        # Score each available target
        rows = []
        for t, mobj in trained.items():
            p = predict_proba_for_homerow(mobj, synth_row)
            rec, book_p, edge, ev = None, None, None, None

            if t == "moneyline":
                rec = "HOME" if p >= 0.5 else "AWAY"
                hm = synth_row.get("home_ml"); aw = synth_row.get("away_ml")
                if hm is not None and aw is not None:
                    if rec == "HOME":
                        book_p = american_to_implied_prob(hm)
                        ev = expected_value_per_100(p, hm)
                        edge = p - (book_p if book_p is not None else 0.0)
                    else:
                        book_p = american_to_implied_prob(aw)
                        ev = expected_value_per_100(1-p, aw)
                        edge = (1-p) - (book_p if book_p is not None else 0.0)

            elif t == "spread":
                rec = "HOME cover" if p >= 0.5 else "AWAY cover"
                ho = synth_row.get("home_spread_odds"); ao = synth_row.get("away_spread_odds")
                if ho is not None and ao is not None:
                    if rec.startswith("HOME"):
                        book_p = american_to_implied_prob(ho)
                        ev = expected_value_per_100(p, ho)
                        edge = p - (book_p if book_p is not None else 0.0)
                    else:
                        book_p = american_to_implied_prob(ao)
                        ev = expected_value_per_100(1-p, ao)
                        edge = (1-p) - (book_p if book_p is not None else 0.0)

            elif t == "over_under":
                rec = "OVER" if p >= 0.5 else "UNDER"
                oo = synth_row.get("over_odds"); uo = synth_row.get("under_odds")
                if oo is not None and uo is not None:
                    if rec == "OVER":
                        book_p = american_to_implied_prob(oo)
                        ev = expected_value_per_100(p, oo)
                        edge = p - (book_p if book_p is not None else 0.0)
                    else:
                        book_p = american_to_implied_prob(uo)
                        ev = expected_value_per_100(1-p, uo)
                        edge = (1-p) - (book_p if book_p is not None else 0.0)

            rows.append({
                "target": t,
                "prediction": rec,
                "model_prob": round(float(p), 4),
                "book_implied_prob": None if book_p is None else round(float(book_p), 4),
                "edge": None if edge is None else round(float(edge), 4),
                "ev_per_$100": None if ev is None else round(float(ev), 2),
            })

        out_df = pd.DataFrame(rows)
        st.markdown("### Results")
        st.dataframe(out_df, use_container_width=True)
