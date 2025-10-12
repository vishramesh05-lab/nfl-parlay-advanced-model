
import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    ensure_cols, last_n_window, prob_over_normal, stat_label_and_col,
    TEAM_LATLON, fetch_weather, injury_flag_for_player, opponent_def_injuries,
    pace_factor, usage_trend_factor, vig_to_market_prob, context_adjusted_probability,
    POS_FOR_STAT
)

st.set_page_config(page_title="NFL Parlay Helper (Dual Probabilities, 2025)", layout="wide")
st.title("🏈 NFL Parlay Helper (Dual Probabilities, 2025)")
st.caption("Two estimates: (1) Historical from last N games, and (2) Context-Adjusted including injuries, weather, pace, usage trend, opponent defensive injuries, and market vig. For informational/educational use only.")

SEASON = 2025

@st.cache_data(show_spinner=True, ttl=60*60)
def load_all(season: int):
    # Try nflreadpy first; if unavailable/old, fall back to nfl_data_py
    stats = inj = depth = sched = None

    try:
        import nflreadpy as nfl
        try:
            stats = nfl.load_player_stats(seasons=[season], summary_level="week")
        except Exception:
            stats = None
        try:
            inj = nfl.load_injuries(seasons=[season])
        except Exception:
            inj = None
        try:
            depth = nfl.load_depth_charts(seasons=[season])
        except Exception:
            depth = None
        try:
            sched = nfl.load_schedules(seasons=[season])
        except Exception:
            sched = None
    except Exception:
        pass

    # Fallback to nfl_data_py if needed
    if stats is None:
        try:
            import nfl_data_py as nd
            stats = nd.import_weekly_data([season])
            inj = inj if inj is not None else nd.import_injuries([season])
            depth = depth if depth is not None else nd.import_depth_charts([season])
            sched = sched if sched is not None else nd.import_schedules([season])
        except Exception:
            stats = None

    # Final safety: return empty DataFrames if any are still None
    import pandas as pd
    if stats is None:
        stats = pd.DataFrame()
    if inj is None:
        inj = pd.DataFrame()
    if depth is None:
        depth = pd.DataFrame()
    if sched is None:
        sched = pd.DataFrame()

    # Normalize columns & return
    stats = ensure_cols(stats)
    return stats, inj, depth, sched
with st.spinner("Loading nflverse data..."):
    stats_df, inj_df, depth_df, sched_df = load_all(SEASON)

if stats_df is None or stats_df.empty:
    st.error("Could not load stats at the moment.")
    st.stop()

weeks = sorted(stats_df["week"].dropna().unique().tolist())
wmin, wmax = int(min(weeks)), int(max(weeks))

st.sidebar.header("Filters")
week_for_matchup = st.sidebar.slider("Current week for matchup context", min_value=wmin, max_value=wmax, value=wmax, step=1)
n_weeks = st.sidebar.slider("Lookback (N weeks)", min_value=3, max_value=8, value=4, step=1)
team_opt = ["(All)"] + sorted([t for t in stats_df["team"].dropna().unique() if isinstance(t,str)])
pos_opt = ["(All)"] + sorted(stats_df["position_group"].dropna().unique().tolist())
team_sel = st.sidebar.selectbox("Team (optional)", options=team_opt)
pos_sel = st.sidebar.selectbox("Position group (optional)", options=pos_opt)

search = st.text_input("Player name")
col_stat, col_line = st.columns([2,1])
with col_stat:
    stat_name = st.selectbox("Stat", options=list(stat_label_and_col.__globals__["STAT_MAP"].keys()), index=1)
with col_line:
    line_value = st.number_input("Sportsbook line", min_value=0.0, value=60.5, step=0.5)

with st.expander("Optional: enter market odds to incorporate vig (American odds)"):
    c1, c2 = st.columns(2)
    with c1:
        over_odds = st.text_input("Over odds (e.g., -115 or +100)", value="")
    with c2:
        under_odds = st.text_input("Under odds (e.g., -105 or +100)", value="")

go = st.button("Analyze")
st.divider()

# Filter candidates list
filtered = stats_df.copy()
if team_sel != "(All)":
    filtered = filtered[filtered["team"]==team_sel]
if pos_sel != "(All)":
    filtered = filtered[filtered["position_group"]==pos_sel]

def find_best_candidate(df, name):
    if not name:
        return None
    cands = df[df["player_display_name"].str.contains(name, case=False, na=False)]
    if cands.empty:
        return None
    # choose the most recent occurrence
    return cands.sort_values("week").tail(1).iloc[0]

if not go:
    st.info("Enter a player name, stat, line, and click Analyze.")
else:
    cand = find_best_candidate(filtered, search)
    if cand is None:
        st.warning("No matching player found. Try clearing team/position filters or refine the name.")
    else:
        player = cand["player_display_name"]; team = cand["team"]
        stat_label, stat_col = stat_label_and_col(stat_name)
        if stat_col not in stats_df.columns:
            st.error("Selected stat not available in dataset.")
            st.stop()

        st.subheader(f"{player} ({team}, {cand.get('position','')}) — {stat_label} vs Line {line_value}")
        hist = last_n_window(stats_df, player, n_weeks, week_max=week_for_matchup)
        show_cols = ["season","week","team","opponent_team",stat_col]
        show_cols = [c for c in show_cols if c in hist.columns]
        if hist.empty:
            st.warning("Not enough recent games for historical estimate.")
        else:
            # --- Section 1: Historical probability ---
            p_hist, params = prob_over_normal(hist[stat_col].astype(float), float(line_value))
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Last-N Avg", f"{hist[stat_col].mean():.1f}")
            with c2: st.metric("Std Dev (min-reg.)", f"{params['sigma']:.1f}")
            with c3: st.metric("Prob OVER (Historical)", f"{p_hist*100:.1f} %")

            st.markdown("**Recent games (last N)**")
            st.dataframe(hist[show_cols], use_container_width=True, height=220)

            # --- Gather context for Section 2 ---
            # Opponent for selected week
            opp_team = None
            if sched_df is not None and not sched_df.empty:
                s = sched_df[(sched_df["season"]==SEASON) & (sched_df["week"]==week_for_matchup) &
                             ((sched_df["home_team"]==team) | (sched_df["away_team"]==team))]
                if not s.empty:
                    r = s.iloc[0]
                    opp_team = r["away_team"] if r["home_team"]==team else r["home_team"]
                    game_date = str(r.get("game_date",""))
                else:
                    game_date = ""
            else:
                game_date = ""

            # Weather (stadium location → open-meteo)
            weather = {}
            if team in TEAM_LATLON:
                lat, lon = TEAM_LATLON[team]
                # assume 4:25pm local if exact kickoff unknown
                iso_time = (game_date + "T16:00") if game_date else pd.Timestamp.now().strftime("%Y-%m-%dT16:00")
                weather = fetch_weather(lat, lon, iso_time)

            # Player injury
            player_inj = injury_flag_for_player(inj_df, player)

            # Opponent defensive injuries
            def_inj = opponent_def_injuries(inj_df, opp_team) if opp_team else {}

            # Pace proxy
            pace_mult = pace_factor(stats_df, team, opp_team) if opp_team else 1.0

            # Usage trend (last3 vs prior3)
            usage_mult = usage_trend_factor(hist, stat_col)

            # Market odds → implied prob
            def parse_odds(s):
                s = s.strip()
                if not s: return None
                try:
                    return float(s)
                except: return None
            market_prob = None
            o_odds = parse_odds(over_odds); u_odds = parse_odds(under_odds)
            if o_odds is not None and u_odds is not None:
                market_prob = vig_to_market_prob(o_odds, u_odds)

            # --- Section 2: Context-Adjusted probability ---
            p_final, details = context_adjusted_probability(
                hist_samples=hist[stat_col].astype(float),
                line_value=float(line_value),
                stat_label=stat_label,
                player_injury_flag=player_inj,
                weather_dict=weather,
                opp_def_inj=def_inj,
                pace_mult=pace_mult,
                usage_mult=usage_mult,
                market_prob_over=market_prob
            )

            d1, d2, d3 = st.columns(3)
            with d1: st.metric("Prob OVER (Adjusted)", f"{details['p_final']*100:.1f} %")
            with d2: st.metric("Adj. Mean (μ')", f"{details['mu_adjusted']:.1f}")
            with d3: st.metric("Market Over (de‑vig)", f"{(details['p_market_over']*100 if not np.isnan(details['p_market_over']) else 0):.1f} %")

            with st.expander("Explain adjustments"):
                st.markdown(f"""
**Player injury:** `{player_inj}` → multiplier **{details['m_injury']:.2f}**  
**Weather:** wind/precip/temp → **{details['m_weather']:.2f}**  
**Opponent defensive injuries:** {def_inj if def_inj else "{}"} → **{details['m_opp_def']:.2f}**  
**Pace proxy (both teams):** **{details['m_pace']:.2f}**  
**Usage trend (last3 vs prior):** **{details['m_usage']:.2f}**  
**Historical μ, σ:** {details['mu_hist']:.1f}, {details['sigma_hist']:.1f}  
**Adjusted μ':** {details['mu_adjusted']:.1f}  
**Model Adj Prob:** {details['p_model_adj']*100:.1f}%  
**Market (de‑vig) Over prob:** {("" if np.isnan(details['p_market_over']) else f"{details['p_market_over']*100:.1f}%")}  
**Final (blend 65% model / 35% market):** {details['p_final']*100:.1f}%
""")
            st.caption("Heuristics only. Not betting advice. Weather via Open‑Meteo, injuries/schedules via nflverse at runtime.")
