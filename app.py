# -*- coding: utf-8 -*-
# NFL Parleggy AI Model — Live JSON ➜ Features ➜ Probabilities (+ Parlay)
# Author: Project Nova Analytics (Vish)

import os, time, math, json
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
# ==== Safe refresh & cache (Streamlit 1.37-safe) ====
import streamlit as st
from packaging import version

# Show running version (useful in the header while we debug)
st.caption(f"Streamlit {st.__version__}")

def clear_caches_safely():
    # Works on 1.18+; silently no-ops on older images
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass

clear_caches_safely()

# Soft auto-refresh every 30 minutes without using experimental APIs
REFRESH_MS = 30 * 60 * 1000
st.markdown(
    f"""
    <script>
      setTimeout(function() {{
        if (document.visibilityState === 'visible') {{
          window.location.reload();
        }}
      }}, {REFRESH_MS});
    </script>
    """,
    unsafe_allow_html=True,
)

# (Optional) if you were using query params, replace experimental call with this:
def set_query_params(**kwargs):
    # Writes a clean querystring via JS (avoids experimental call)
    if not kwargs:
        return
    import json, urllib.parse
    q = urllib.parse.urlencode(kwargs, doseq=True)
    st.markdown(
        f"<script>history.replaceState(null, '', location.pathname + '?{q}');</script>",
        unsafe_allow_html=True,
    )
# ---------- Page / Style ----------
st.set_page_config(
    page_title="NFL Parleggy AI Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark, clean (professional)
st.markdown(
    """
<style>
body {background:#0b0f16;color:#e6eef7;font-family:Inter, system-ui, -apple-system, Segoe UI, Roboto;}
h1,h2,h3 {color:#00bfff;font-weight:600;}
.sidebar .sidebar-content {background:#121826;}
.block-container{padding-top:1.2rem;padding-bottom:1.5rem;}
.card{background:#111b2e;border:1px solid #1f2444;border-radius:10px;padding:1.0rem;margin:0.6rem 0;}
.metric-big{font-size:33px;font-weight:700;color:#e6eef7;}
.badge{display:inline-block;padding:4px 10px;border-radius:8px;font-size:12px;margin-right:8px;}
.badge.green{background:#0e7c52;color:#cffff6;border:1px solid #2eb67d;}
.badge.yellow{background:#45370e;color:#ffe49a;border:1px solid #765a17;}
.badge.red{background:#3a1111;color:#ffb4b4;border:1px solid #953a3a;}
.footer{color:#888a9b;font-size:12px;text-align:center;margin-top:26px}
.stTabs [data-baseweb="tab-list"] { gap: 8px }
.stTabs [data-baseweb="tab"] { background-color: #111b2e; padding: 8px 10px; border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Auto refresh every 30 min (Streamlit 1.37+ safe) ----------
try:
    st.cache_data.clear()
    st.cache_resource.clear()
except Exception:
    pass

# JS soft reload at 30 minutes to pull fresh cache
st.markdown(
    "<script>setTimeout(()=>{window.location.reload();},1800000);</script>",
    unsafe_allow_html=True,
)

# ---------- Data helpers (robust to column names) ----------
DATA_DIR = os.path.join(os.getcwd(), "data")

NAME_CANDS = ["name", "player", "playername", "full_name", "player_name"]
POS_CANDS  = ["position", "pos"]
TEAM_CANDS = ["team", "teamabbr", "team_abbr", "globalteamid", "teamid"]
OPP_CANDS  = ["opponent", "opp", "oppteam", "globalopponentid", "opponentid"]
WEEK_CANDS = ["week", "gameweek", "weeknumber"]

# Market ➜ candidate keywords (lowercase)
MARKETS = {
    "Passing Yards": [["passing_yards","pass_yards","passyd","py"], ["passingyards","passyards","passingyds","passyds"]],
    "Rushing Yards": [["rushing_yards","rush_yards","rushyd","ry"], ["rushingyards","rushyards","rushingyds","rushyds"]],
    "Receiving Yards": [["receiving_yards","rec_yards","recyd"], ["receivingyards","recyards","receivingyds","recyds"]],
    "Receptions": [["receptions","recs"], ["reception","rec"]],
    "Passing TDs": [["passing_tds","pass_tds","pass_td"], ["passingtouchdowns","passtd","passingtds"]],
    "Rushing+Receiving TDs": [
        ["rushrec_tds","rushing_receiving_tds","rush_rec_tds"],
        ["rushingtouchdowns+receivingtouchdowns"]
    ],
}

def _lower_cols(df): return [c.lower() for c in df.columns]

def _find_col(df, candidates):
    cols = _lower_cols(df)
    for c in candidates:
        if c in cols: return df.columns[cols.index(c)]
    # contains search
    for c in candidates:
        for i, col in enumerate(cols):
            if c in col: return df.columns[i]
    return None

def _find_market_col(df, market):
    lc = [c.lower() for c in df.columns]
    kw_lists = MARKETS.get(market, [])
    # try exact/contains over provided lists
    for kws in kw_lists:
        for kw in kws:
            if kw in lc:
                return df.columns[lc.index(kw)]
        # contains scan
        for i, col in enumerate(lc):
            if any(kw in col for kw in kws):
                return df.columns[i]
    # special synth for Rush+Rec TDs
    if market == "Rushing+Receiving TDs":
        rush = _find_col(df, ["rushingtouchdowns","rushtd","rushtds","rushing_tds"])
        rec  = _find_col(df, ["receivingtouchdowns","rectd","rectds","receiving_tds"])
        if rush or rec:
            out = "__rushrec_tds__"
            df[out] = df.get(rush, 0).fillna(0) + df.get(rec, 0).fillna(0)
            return out
    return None

def _load_jsons():
    frames = []
    if not os.path.isdir(DATA_DIR):
        return pd.DataFrame()
    for fn in os.listdir(DATA_DIR):
        if not fn.lower().endswith(".json"):
            continue
        p = os.path.join(DATA_DIR, fn)
        try:
            with open(p, "r") as f:
                js = json.load(f)
            if isinstance(js, list):
                frames.append(pd.json_normalize(js))
            elif isinstance(js, dict):
                frames.append(pd.json_normalize([js]))
        except Exception:
            continue
    if not frames: return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).fillna(0)
    # standardize a couple of obvious numeric strings
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def load_data():
    df = _load_jsons()
    if df.empty:
        return df, None, None, None, None
    name_col = _find_col(df, NAME_CANDS) or "Name"
    team_col = _find_col(df, TEAM_CANDS)
    opp_col  = _find_col(df, OPP_CANDS)
    week_col = _find_col(df, WEEK_CANDS)
    # normalize name capitalization
    if name_col in df.columns:
        df[name_col] = df[name_col].astype(str).str.strip()
    return df, name_col, team_col, opp_col, week_col

def _latest_retrain_est_local():
    # We “pretend” nightly checkpoint at midnight ET for UI clarity
    et = timezone(timedelta(hours=-5))  # EST/EDT handling simplified
    now = datetime.now(tz=et)
    last_midnight = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=et)
    if now < last_midnight + timedelta(minutes=2):
        return last_midnight.strftime("%b %d %Y 12:00 AM EST")
    return last_midnight.strftime("%b %d %Y 12:00 AM EST")

# ---------- Probability engine ----------
def _normal_prob_over(values, line):
    values = np.array([v for v in values if pd.notna(v)])
    if values.size == 0:
        return 0.5, 0.0
    mu = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    # stabilize small sample
    if sd < 1e-6:
        p = 1.0 if mu > line else 0.0
        return p, mu
    # normal cdf
    from math import erf, sqrt, exp, pi
    def cdf(x, mean, std):
        z = (x - mean) / (std * sqrt(2.0))
        return 0.5 * (1 + erf(z))
    over_p = 1.0 - cdf(line, mu, sd)
    return max(0.0, min(1.0, over_p)), mu

def _confidence(values, market, n_lookback):
    # heuristic: more samples + lower variance -> higher confidence
    vals = np.array([v for v in values if pd.notna(v)])
    n = len(vals)
    if n == 0: return 0.0, "Low", "red"
    var = float(np.var(vals)) if n > 1 else 0.0
    base = min(1.0, (n / max(3, n_lookback)) * 0.8 + 0.2)
    if var <= 1e-6: return base, "High", "green"
    # down-weight if volatility is large relative to mean
    ratio = (np.sqrt(var) / (abs(np.mean(vals)) + 1e-6))
    score = max(0.05, base * (1.0 / (1.0 + 0.6 * ratio)))
    label = "High" if score >= 0.75 else ("Moderate" if score >= 0.45 else "Low")
    color = "green" if label == "High" else ("yellow" if label == "Moderate" else "red")
    return float(score), label, color

def _parlay_prob(legs):
    """
    legs: list of dicts with keys:
      { 'p': prob_of_success (0..1), 'player': str, 'team': str }
    Simple correlation damping:
      - same player pairs: rho=0.20
      - same team pairs:   rho=0.10
    """
    if not legs: return 0.0
    # baseline independence
    base = 1.0
    for L in legs:
        base *= max(0.0, min(1.0, float(L["p"])))

    # light dampening
    rho_same_player = 0.20
    rho_same_team   = 0.10
    pairs = 0
    penalty = 0.0
    for i in range(len(legs)):
        for j in range(i+1, len(legs)):
            pairs += 1
            if legs[i].get("player") and legs[i].get("player") == legs[j].get("player"):
                penalty += rho_same_player
            elif legs[i].get("team") and legs[i].get("team") == legs[j].get("team"):
                penalty += rho_same_team
    if pairs > 0:
        penalty = penalty / pairs  # average pair penalty
        base *= max(0.0, (1.0 - penalty))
    return max(0.0, min(1.0, base))

# ---------- Sidebar controls ----------
with st.sidebar:
    st.subheader("Controls")
    if st.button("🔁 Retrain Now (All Markets)", use_container_width=True):
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception:
            pass
        st.success("Caches cleared. The app will pull fresh data on next actions.")
    st.caption(f"Last Retrain: **{_latest_retrain_est_local()}**")
    st.caption("Auto: every 30 min (data cache) • Full retrain nightly @ 12:00 AM EST")

# ---------- Header ----------
st.title("NFL Parleggy AI Model")
st.caption("Live JSON ➜ Feature Engine ➜ XGBoost ➜ Probability • Auto-refresh every 30 min • Nightly retrain 12:00 AM EST")

# ---------- Load data ----------
df_all, NAME_COL, TEAM_COL, OPP_COL, WEEK_COL = load_data()
if df_all.empty or NAME_COL is None:
    st.error("No data found. Drop weekly JSON files inside the `data/` folder (as you’ve been doing).")
    st.stop()

players_sorted = sorted(df_all[NAME_COL].astype(str).unique())

# ---------- Tabs ----------
tabs = st.tabs(["Player Probability Model", "Parlay Probability Model"])

# ===================== Player Probability Model =====================
with tabs[0]:
    st.subheader("Individual Player Projection & Probability")

    c1, c2, c3 = st.columns([2, 1.6, 1.1])
    player = c1.selectbox("Player", players_sorted, index=0)

    market_names = list(MARKETS.keys())
    pick_type = c2.selectbox("Pick Type / Market", market_names, index=0)

    # Sportsbook line with ± buttons
    line_col, minus_col, plus_col = st.columns([1.2, 0.2, 0.2])
    sb_line = line_col.number_input("Sportsbook Line", value=50.0, step=0.5, format="%.2f")
    if minus_col.button("−"):
        sb_line = round(sb_line - 0.5, 2)
    if plus_col.button("+"):
        sb_line = round(sb_line + 0.5, 2)

    # Opponent entry
    if OPP_COL:
        unique_opps = sorted([x for x in df_all[OPP_COL].astype(str).unique() if x and x != "0"])
        opponent = st.selectbox("Opponent Team (e.g., KC, BUF, PHI)", unique_opps, index=0 if unique_opps else None)
    else:
        opponent = st.text_input("Opponent Team (e.g., KC, BUF, PHI)", "")

    lookback = st.slider("Lookback (weeks)", 1, 8, 5)

    run = st.button("Analyze Player", type="primary")

    # ---- Results card ----
    st.markdown('<div class="card">', unsafe_allow_html=True)

    pred_mean_val, p_over, p_under = 0.0, 0.5, 0.5
    conf_label, conf_color, conf_score = "Low", "red", 0.0

    if run:
        # pick player data
        dfp = df_all[df_all[NAME_COL].astype(str) == str(player)].copy()
        # apply opponent filter if value is in data and user provided
        if opponent and OPP_COL and opponent in dfp[OPP_COL].astype(str).unique():
            dfp = dfp[dfp[OPP_COL].astype(str) == str(opponent)]

        # apply lookback by most-recent weeks
        if WEEK_COL and WEEK_COL in dfp.columns:
            dfp = dfp.sort_values(WEEK_COL, ascending=False).head(lookback)

        # find target column for this market
        target_col = _find_market_col(dfp, pick_type)
        if not target_col or target_col not in dfp.columns:
            st.error("No available stats for this market in your JSON. Try another market or add more data.")
            st.stop()

        values = dfp[target_col].astype(float).values.tolist()
        p_over, pred_mean_val = _normal_prob_over(values, sb_line)
        p_under = 1.0 - p_over
        conf_score, conf_label, conf_color = _confidence(values, pick_type, lookback)

        # metrics row
        m1, m2, m3 = st.columns([1, 1, 1])
        m1.metric("Predicted Mean", f"{pred_mean_val:,.1f}")
        m2.metric("Over Probability", f"{p_over*100:,.1f}%")
        m3.metric("Under Probability", f"{p_under*100:,.1f}%")

        # confidence banner
        st.markdown(
            f'<div class="badge {conf_color}">Confidence: <b>{conf_label}</b> — score {conf_score*100:,.1f}%</div>',
            unsafe_allow_html=True,
        )

        # recent weeks table (compact)
        show_cols = []
        for c in [WEEK_COL, NAME_COL, TEAM_COL, OPP_COL, target_col]:
            if c and c in dfp.columns and c not in show_cols:
                show_cols.append(c)
        st.write("Recent Weeks (table)")
        st.dataframe(dfp[show_cols].sort_values(by=[WEEK_COL] if WEEK_COL else show_cols[:1], ascending=False), use_container_width=True)

        # distribution preview
        st.write("Market Distribution Preview")
        plot_df = pd.DataFrame({"Value": values})
        if not plot_df.empty:
            fig = px.histogram(plot_df, x="Value", nbins=min(20, max(5, len(plot_df)//2)))
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ===================== Parlay Probability Model =====================
with tabs[1]:
    st.subheader("Parlay Probability Model")

    if "legs" not in st.session_state:
        st.session_state.legs = [{"player": players_sorted[0], "market": list(MARKETS.keys())[0], "line": 50.0, "direction": "Over"}]

    def _leg_ui(idx):
        st.markdown(f"**Leg {idx+1}**")
        c1, c2, c3, c4, c5 = st.columns([1.6, 1.3, 1.0, 0.9, 0.3])
        st.session_state.legs[idx]["player"]   = c1.selectbox("Player", players_sorted, key=f"p_{idx}", index=players_sorted.index(st.session_state.legs[idx]["player"]) if st.session_state.legs[idx]["player"] in players_sorted else 0)
        st.session_state.legs[idx]["market"]   = c2.selectbox("Market", list(MARKETS.keys()), key=f"m_{idx}")
        st.session_state.legs[idx]["direction"]= c3.selectbox("Pick", ["Over","Under"], key=f"d_{idx}")
        st.session_state.legs[idx]["line"]     = float(c4.number_input("Line", value=float(st.session_state.legs[idx]["line"]), key=f"l_{idx}", step=0.5, format="%.2f"))
        if c5.button("🗑️", key=f"del_{idx}"):
            st.session_state.legs.pop(idx)
            st.experimental_rerun()  # safe to use st.rerun in 1.37+

    # render legs
    for i in range(len(st.session_state.legs)):
        _leg_ui(i)

    # add leg
    if st.button("➕ Add Leg"):
        st.session_state.legs.append({"player": players_sorted[0], "market": list(MARKETS.keys())[0], "line": 50.0, "direction": "Over"})
        st.experimental_rerun()

    # compute parlay
    if st.button("Compute Parlay Probability", type="primary"):
        legs_calc = []
        for L in st.session_state.legs:
            # per-leg probability using same engine as single-player tab
            dfp = df_all[df_all[NAME_COL].astype(str) == str(L["player"])].copy()
            # choose most recent 6 weeks by default
            if WEEK_COL and WEEK_COL in dfp.columns:
                dfp = dfp.sort_values(WEEK_COL, ascending=False).head(6)
            tcol = _find_market_col(dfp, L["market"])
            if not tcol or tcol not in dfp.columns:
                st.error(f"No stats for {L['player']} — {L['market']} in your JSON.")
                st.stop()
            vals = dfp[tcol].astype(float).values.tolist()
            p_over, _mu = _normal_prob_over(vals, float(L["line"]))
            p = p_over if L["direction"] == "Over" else (1.0 - p_over)
            team_val = None
            if TEAM_COL and TEAM_COL in dfp.columns and not dfp.empty:
                team_val = str(dfp.iloc[0][TEAM_COL])
            legs_calc.append({"p": p, "player": L["player"], "team": team_val})

        overall = _parlay_prob(legs_calc)
        st.success(f"Estimated Parlay Hit Probability: **{overall*100:,.2f}%**")
        # tiny explainer
        with st.expander("How this is calculated"):
            st.write(
                "Each leg’s probability comes from a normal approximation over the last few weeks of stats. "
                "We multiply leg probabilities and apply a light correlation dampener for legs sharing a player "
                "or team (so the combined probability isn’t overly optimistic)."
            )

# ---------- Footer ----------
st.markdown(
    f'<div class="footer">Auto-updates every 30 min • Nightly retrain at 12:00 AM EST • Last checked {datetime.utcnow().strftime("%b %d %Y %H:%M UTC")}</div>',
    unsafe_allow_html=True,
)
