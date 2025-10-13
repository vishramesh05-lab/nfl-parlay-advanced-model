# -*- coding: utf-8 -*-
# NFL Parleggy AI Model – Live JSON ➜ Features ➜ Probabilities (+ Parlay)
# Author: Project Nova Analytics (Vish)
#
# Notes
# - Reads all *.json files inside ./data (repo_root/data)
# - Caches merged data for 30 minutes (auto-refresh supported)
# - Two tabs:
#     1) Player Probability Model (single leg)
#     2) Parlay Probability Model (multi-leg, independence assumption)
# - Nightly "retrain" timestamp @ 12:00 AM EST (bookkeeping only; no heavy fit here)
# - Robust stat detection so SportsDataIO-style column variations still work

import os, time, math, json
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Try to use your utils.py if present; fall back to local helpers if not.
try:
    import utils as U  # your helper functions if available
except Exception:
    U = None

# ====== PAGE CONFIG (must be FIRST Streamlit call) ======
st.set_page_config(
    page_title="NFL Parleggy AI Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====== Safe cache clear + housekeeping ======
def _clear_caches():
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass

# Store last "retrain" time in session (display only)
if "last_retrain_utc" not in st.session_state:
    st.session_state.last_retrain_utc = None

# ====== Style (dark professional) ======
st.markdown(
    """
    <style>
    /* Clean dark theme tweaks */
    body {background:#0f1117; color:#eef2ff; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto;}
    h1,h2,h3 {color:#00b4ff; font-weight:600;}
    .block-container {padding-top:1.0rem;}
    .card{background:#111b2e;border:1px solid #14244a;border-radius:10px;padding:0.9rem 1rem;margin:0.4rem 0;}
    .badge{display:inline-block;background:#1b2b4a;color:#bfe7ff;border-radius:6px;padding:4px 8px;font-size:12px;margin-right:6px}
    .badge.green{background:#0e4b2e;color:#b9f3d0}
    .badge.red{background:#3b151a;color:#ffc9c9}
    .metric{font-size:34px;font-weight:700;color:#eaffff}
    .metric-sub{font-size:12px;color:#9fb0c0}
    .footer{color:#889; font-size:12px; text-align:center; margin-top: 18px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# =============== Data loading / detection ================
# =========================================================

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

NAME_CANDIDATES = [
    "Name","Player","PlayerName","player_name","FullName","Fullname","Full Name"
]
WEEK_CANDIDATES = ["Week","week","GameWeek","game_week","GmWeek","PlayerGameWeek"]
TEAM_CANDIDATES = ["Team","TeamAbbr","TeamName","TeamCode"]
OPP_CANDIDATES  = ["Opponent","Opp","OpponentTeam","OppTeam","GlobalOpponentID","OpponentID","OpponentAbbr"]

# Market keyword map (lowercase contains)
MARKET_KEYWORDS = {
    "Passing Yards":       [["pass"], ["yd"]],
    "Rushing Yards":       [["rush"], ["yd"]],
    "Receiving Yards":     [["receiv"], ["yd"]],
    "Receptions":          [["recep","rec"], []],
    "Completions":         [["comp","compl"], []],
    "Pass Attempts":       [["pass","att"], []],
    "Carries":             [["rush","att"], []],
    "Passing TDs":         [["pass"], ["td","touch"]],
    "Rushing TDs":         [["rush"], ["td","touch"]],
    "Receiving TDs":       [["receiv"], ["td","touch"]],
}

def _list_week_files(path: str):
    try:
        return sorted(
            [f for f in os.listdir(path) if f.lower().endswith(".json")],
        )
    except Exception:
        return []

@st.cache_data(ttl=1800, show_spinner=False)  # cache 30 min
def load_all_jsons():
    """Load & merge every JSON file from ./data."""
    if U and hasattr(U, "load_all_jsons"):
        return U.load_all_jsons()  # your robust loader if present

    frames = []
    for fname in _list_week_files(DATA_PATH):
        fpath = os.path.join(DATA_PATH, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                js = json.load(f)
            # JSON may be an array or object → normalize
            if isinstance(js, list):
                df = pd.json_normalize(js)
            else:
                df = pd.json_normalize(js)
            df["__source_file"] = fname
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out

def _detect_col(df: pd.DataFrame, candidates) -> str | None:
    # exact
    for c in candidates:
        if c in df.columns: return c
    # case-insensitive
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower: return cols_lower[c.lower()]
    return None

def detect_name_column(df):
    if U and hasattr(U, "detect_name_column"):
        return U.detect_name_column(df)
    return _detect_col(df, NAME_CANDIDATES)

def detect_week_column(df):
    if U and hasattr(U, "detect_week_column"):
        return U.detect_week_column(df)
    return _detect_col(df, WEEK_CANDIDATES)

def detect_team_column(df):
    if U and hasattr(U, "detect_team_column"):
        return U.detect_team_column(df)
    return _detect_col(df, TEAM_CANDIDATES)

def detect_opp_column(df):
    if U and hasattr(U, "detect_opp_column"):
        return U.detect_opp_column(df)
    return _detect_col(df, OPP_CANDIDATES)

def _find_by_keywords(df: pd.DataFrame, require_all_groups: list[list[str]]) -> str | None:
    """
    Find a column that contains all tokens from each group in 'require_all_groups'
    Example: [["pass"],["yd"]]  → column must include 'pass' AND 'yd' (lowercased)
    """
    lcmap = {c.lower(): c for c in df.columns}
    for lc, original in lcmap.items():
        ok = True
        for group in require_all_groups:
            if not any(token in lc for token in group):
                ok = False
                break
        if ok:
            return original
    return None

def find_target_column(df: pd.DataFrame, market: str) -> str | None:
    if U and hasattr(U, "find_target_column"):
        s, col = U.find_target_column(df, market)
        return col
    # fallback: use keyword map
    req = MARKET_KEYWORDS.get(market)
    if not req:
        return None
    return _find_by_keywords(df, req)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if U and hasattr(U, "standardize_columns"):
        return U.standardize_columns(df)

    # Make sure Name & Week exist if possible
    name_col = detect_name_column(df)
    week_col = detect_week_column(df)
    if name_col and name_col != "Name":
        df = df.rename(columns={name_col: "Name"})
        name_col = "Name"
    if week_col and week_col != "Week":
        df = df.rename(columns={week_col: "Week"})
        week_col = "Week"

    # Convert week to numeric if present
    if "Week" in df.columns:
        try:
            df["Week"] = pd.to_numeric(df["Week"], errors="coerce")
        except Exception:
            pass

    # Strip/clean names
    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str).str.strip()

    return df


# =========================================================
# ================= Probability helpers ===================
# =========================================================

def normal_over_probability(samples: np.ndarray, line: float) -> float:
    """
    Approximate P(X > line) assuming normal(μ, σ^2).
    Uses error function; independent of SciPy.
    """
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return np.nan
    mu = float(np.mean(samples))
    sd = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
    if sd <= 1e-9:
        return 1.0 if mu > line else 0.0
    z = (line - mu) / (sd * math.sqrt(2.0))
    # P(X>line) = 0.5 * erfc((line-mu)/(sqrt(2)*sd))
    return 0.5 * math.erfc(z)

def confidence_score(samples: np.ndarray) -> float:
    """
    Simple confidence: up with N and down with variability.
    Returns [0..1].
    """
    samples = samples[~np.isnan(samples)]
    n = samples.size
    if n == 0:
        return 0.0
    sd = float(np.std(samples, ddof=1)) if n > 1 else 0.0
    # dampen when very noisy; cap to [0,1]
    # heuristic: base on sqrt(n) and sd scale
    base = min(1.0, math.sqrt(n) / 4.0)          # ~0.5 at n=4, ~0.71 at n=8, ->1 near n=16
    penalty = 1.0 / (1.0 + sd)                    # 0..1
    return max(0.0, min(1.0, base * penalty))


# =========================================================
# ===================== UI / Controls =====================
# =========================================================

with st.sidebar:
    st.subheader("Controls")
    if st.button("🔄 Retrain Now (All Markets)", use_container_width=True):
        # bookkeeping retrain timestamp + clear caches
        st.session_state.last_retrain_utc = datetime.now(timezone.utc)
        _clear_caches()
        st.success("Caches cleared. Data will refresh from JSON on next run.")

    # Show last retrain
    if st.session_state.last_retrain_utc:
        ts = st.session_state.last_retrain_utc.strftime("%b %d %Y %H:%M UTC")
        st.caption(f"**Last Retrain:** {ts}")
    else:
        st.caption("**Last Retrain:** never")

    st.caption("Auto: every 30 min (data cache) • Full retrain nightly @ 12:00 AM EST")

st.title("NFL Parleggy AI Model")
st.caption("Live JSON ➜ Feature Engine ➜ Probabilities • Auto-refresh every 30 min • Nightly retrain 12:00 AM EST")

# Load + prep data
@st.cache_data(ttl=1800, show_spinner=True)
def _load_data_ready():
    df = load_all_jsons()
    df = standardize_columns(df)
    return df

data = _load_data_ready()

tab1, tab2 = st.tabs(["Player Probability Model", "Parlay Probability Model"])

# ---------------------------------------------------------
# Tab 1: Player Probability
# ---------------------------------------------------------
with tab1:
    if data.empty:
        st.error("No JSON data found. Put your weekly player JSON files in the `/data` folder.")
        st.stop()

    name_col = detect_name_column(data)
    if not name_col:
        st.error("Could not detect player name column. Ensure your JSON has `Name`, `Player`, or `PlayerName`.")
        st.stop()

    # Player list
    players = sorted(pd.Series(data[name_col].astype(str).unique()).dropna().tolist())
    colA, colB, colC, colD = st.columns([1.2, 1.0, 0.9, 1.1])
    with colA:
        player = st.selectbox("Player", players, index=0, key="single_player")

    markets = list(MARKET_KEYWORDS.keys())
    with colB:
        market = st.selectbox("Pick Type / Market", markets, index=0)

    with colC:
        sportsbook_line = st.number_input("Sportsbook Line", value=50.0, step=0.5, format="%.2f")

    # lookback
    with colD:
        lookback = st.slider("Lookback (weeks)", 1, 8, value=5)

    # Opponent (optional)
    opp_text = st.text_input("Opponent (e.g., KC, BUF, PHI)", "")

    # Analyze button
    run = st.button("Analyze Player", type="primary")

    # Do work
    if run:
        target_col = find_target_column(data, market)
        if not target_col:
            st.error(
                f"Could not find a matching stat column for **{market}** in your JSON files. "
                "Please ensure a column exists with keywords like the market name "
                "(e.g., for Passing Yards something like `PassingYards` or `PassYds`)."
            )
            st.stop()

        dfp = data[data[name_col].astype(str) == str(player)].copy()
        if dfp.empty:
            st.error("No rows found for that player.")
            st.stop()

        # Optional opponent filter if usable
        opp_col = detect_opp_column(dfp)
        if opp_text and opp_col and opp_col in dfp.columns:
            dfp = dfp[dfp[opp_col].astype(str).str.contains(opp_text, case=False, na=False)]

        # Sort by week, take last N
        week_col = detect_week_column(dfp)
        if week_col and week_col in dfp.columns:
            dfp = dfp.sort_values(week_col, ascending=False)
        dfp_recent = dfp.head(lookback).copy()

        if target_col not in dfp_recent.columns:
            st.error(f"Detected column `{target_col}` not found in filtered data.")
            st.stop()

        vals = pd.to_numeric(dfp_recent[target_col], errors="coerce").to_numpy(dtype=float)
        mean_pred = float(np.nanmean(vals)) if vals.size else float("nan")
        p_over = normal_over_probability(vals, sportsbook_line)
        p_under = float(1.0 - p_over) if not np.isnan(p_over) else np.nan
        conf = confidence_score(vals)

        # Metrics strip
        st.write("")
        m1, m2, m3 = st.columns([1,1,1])
        m1.metric("Predicted Mean", f"{mean_pred:,.1f}" if not np.isnan(mean_pred) else "—")
        m2.metric("Over Probability", f"{(p_over*100):.1f}%" if not np.isnan(p_over) else "—")
        m3.metric("Under Probability", f"{(p_under*100):.1f}%" if not np.isnan(p_under) else "—")

        # Confidence banner
        level = "Low"
        color = "red"
        if conf >= 0.75:
            level, color = "High", "green"
        elif conf >= 0.45:
            level, color = "Moderate", "badge"
        st.markdown(
            f"""<div class="card"><span class="badge {color}">{level} Confidence</span>
            <span class="metric-sub">&nbsp;Model confidence: {conf*100:.1f}% • Samples: {len(vals)}</span></div>""",
            unsafe_allow_html=True,
        )

        # Recent table
        st.subheader("Recent Weeks (table)")
        show_cols = ["__source_file", target_col]
        if week_col: show_cols.append(week_col)
        if opp_col:  show_cols.append(opp_col)
        if name_col not in show_cols:
            show_cols.append(name_col)
        cols = [c for c in show_cols if c in dfp_recent.columns]
        st.dataframe(dfp_recent[cols].rename(columns={name_col:"Name"}), use_container_width=True)

        # Distribution preview
        st.subheader("Market Distribution Preview")
        plot_df = pd.DataFrame({ "Value": pd.to_numeric(dfp[target_col], errors="coerce") })
        plot_df = plot_df.dropna()
        if not plot_df.empty:
            fig = px.histogram(plot_df, x="Value", nbins=20, title=f"{player} • {market} (history)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough historical values to draw a distribution.")

# ---------------------------------------------------------
# Tab 2: Parlay Probability
# ---------------------------------------------------------
with tab2:
    if data.empty:
        st.info("No JSON data available yet.")
    else:
        name_col = detect_name_column(data)
        players = sorted(pd.Series(data[name_col].astype(str).unique()).dropna().tolist())
        markets = list(MARKET_KEYWORDS.keys())

        st.write("Build up to four legs (independence assumed).")
        legs = []
        default_lines = {
            "Passing Yards": 250.0, "Rushing Yards": 65.0, "Receiving Yards": 65.0,
            "Receptions": 5.5, "Completions": 22.5, "Pass Attempts": 33.5, "Carries": 15.5,
            "Passing TDs": 1.5, "Rushing TDs": 0.5, "Receiving TDs": 0.5
        }

        for i in range(1, 5):
            with st.expander(f"Leg {i}", expanded=(i==1)):
                col1, col2, col3, col4 = st.columns([1.2, 1.0, 1.0, 1.0])
                p = col1.selectbox("Player", players, index=min(i-1, len(players)-1), key=f"par_p{i}")
                m = col2.selectbox("Market", markets, index=0, key=f"par_m{i}")
                line_val = default_lines.get(m, 50.0)
                ln = col3.number_input("Sportsbook Line", value=float(line_val), step=0.5, key=f"par_l{i}")
                lb = col4.slider("Lookback (wks)", 1, 8, value=5, key=f"par_w{i}")

                legs.append((p,m,ln,lb))

        if st.button("Compute Parlay Probability", type="primary"):
            joint = 1.0
            details = []
            for (p, m, ln, lb) in legs:
                if not p: 
                    continue
                col = find_target_column(data, m)
                if not col:
                    details.append((p,m,float("nan")))
                    continue
                sub = data[data[name_col].astype(str) == str(p)].copy()
                week_col = detect_week_column(sub)
                if week_col and week_col in sub.columns:
                    sub = sub.sort_values(week_col, ascending=False)
                sub = sub.head(lb)
                vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
                pov = normal_over_probability(vals, ln)
                details.append((p,m,pov))
                if not np.isnan(pov):
                    joint *= pov

            if not details:
                st.error("No valid legs entered.")
            else:
                st.subheader("Leg probabilities")
                rows = []
                for p,m,pov in details:
                    rows.append({"Player":p,"Market":m,"Over Prob": (pov*100 if not np.isnan(pov) else np.nan)})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                st.subheader("Parlay (assumes independence)")
                if details and not any(np.isnan(d[2]) for d in details if d):
                    st.success(f"Joint Over Probability: **{joint*100:.2f}%**")
                else:
                    st.warning("Not all legs had valid probabilities; joint probability not computed.")


# ===== Footer =====
st.markdown(
    '<div class="footer">Auto-refresh every 30 min • Last updated '
    + datetime.now(timezone.utc).strftime("%b %d %Y %H:%M UTC")
    + " • © 2025 Project Nova Analytics</div>",
    unsafe_allow_html=True,
)
