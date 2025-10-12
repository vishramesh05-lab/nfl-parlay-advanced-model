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
st.caption("Two estimates: (1) Historical from last N games, and (2) Context-Adjusted including injuries, weather, pace, usage trend, opponent defensive injuries, and market vig.")
st.caption("Build: vA8")

SEASON = 2025

@st.cache_data(show_spinner=True, ttl=60*60)
def load_all(season: int):
    import pandas as pd
    import requests
    import streamlit as st

    try:
        # Pull the ESPN player stats endpoint (reliable for live + completed games)
        url = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2025/types/2/statistics/players?limit=5000"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        # Gather all player stat URLs
        player_urls = [p["$ref"] for p in data.get("items", [])]
        players = []

        for ref in player_urls[:250]:  # Limit to first 250 for speed
            r = requests.get(ref)
            if r.status_code != 200:
                continue
            info = r.json()
            athlete = info.get("athlete", {})
            stats = info.get("stats", [])

            player_name = athlete.get("displayName")
            team = athlete.get("team", {}).get("abbreviation")
            position = athlete.get("position", {}).get("abbreviation")

            # Map key stats
            row = {
                "player_display_name": player_name,
                "team": team,
                "position": position,
                "rushing_yards": None,
                "receiving_yards": None,
                "passing_yards": None,
            }

            for s in stats:
                name = s.get("name", "").lower()
                val = s.get("value", 0)
                if "rush" in name and "yard" in name:
                    row["rushing_yards"] = val
                elif "rec" in name and "yard" in name:
                    row["receiving_yards"] = val
                elif "pass" in name and "yard" in name:
                    row["passing_yards"] = val

            players.append(row)

        stats = pd.DataFrame(players)

        if stats.empty:
            raise ValueError("No players returned from ESPN player stats API.")

        stats["season"] = season
        stats["week"] = 1
        stats["position_group"] = stats["position"]

        st.success(f"✅ Loaded {len(stats)} live ESPN player records for 2025 season.")

    except Exception as e:
        st.error(f"Error loading ESPN 2025 player stats: {e}")
        stats = pd.DataFrame(columns=[
            "player_display_name", "team", "position",
            "rushing_yards", "receiving_yards", "passing_yards",
            "week", "season", "position_group"
        ])

    inj = pd.DataFrame()
    depth = pd.DataFrame()
    sched = pd.DataFrame()
    return stats, inj, depth, sched
with st.spinner("Loading nflverse data..."):
    stats_df, inj_df, depth_df, sched_df = load_all(SEASON)

# ----------------- NORMALIZE CORE COLUMNS (no .empty calls) -----------------
if not isinstance(stats_df, pd.DataFrame):
    stats_df = pd.DataFrame()
if not isinstance(sched_df, pd.DataFrame):
    sched_df = pd.DataFrame()

stats_df.columns = [str(c) for c in stats_df.columns]
sched_df.columns = [str(c) for c in sched_df.columns]

# Week column
wk_col = None
for c in stats_df.columns:
    if str(c).lower() == "week":
        wk_col = c; break
if wk_col and wk_col != "week":
    stats_df = stats_df.rename(columns={wk_col: "week"})
if "week" not in stats_df.columns:
    if all(col in sched_df.columns for col in ["game_id","week"]) and "game_id" in stats_df.columns:
        try:
            stats_df = stats_df.merge(sched_df[["game_id","week"]], on="game_id", how="left")
        except Exception:
            pass
if "week" not in stats_df.columns:
    stats_df["week"] = np.nan  # last resort so app won’t crash

# Team/opponent
if "team" not in stats_df.columns:
    if "recent_team" in stats_df.columns:
        stats_df = stats_df.rename(columns={"recent_team":"team"})
    elif "team_abbr" in stats_df.columns:
        stats_df = stats_df.rename(columns={"team_abbr":"team"})
    else:
        stats_df["team"] = "UNK"
if "opponent_team" not in stats_df.columns:
    if "opponent" in stats_df.columns:
        stats_df["opponent_team"] = stats_df["opponent"]
    else:
        stats_df["opponent_team"] = np.nan

# Weeks list (safe)
try:
    weeks = pd.to_numeric(stats_df["week"], errors="coerce").dropna().astype(int).unique().tolist()
    weeks = sorted(list(set(weeks)))
except Exception:
    weeks = []
if not weeks:
    weeks = list(range(1,19))  # fallback so UI renders
wmin, wmax = int(min(weeks)), int(max(weeks))
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")
week_for_matchup = st.sidebar.slider("Current week for matchup context", min_value=wmin, max_value=wmax, value=wmax, step=1)
n_weeks = st.sidebar.slider("Lookback (N weeks)", min_value=3, max_value=8, value=4, step=1)
team_opt = ["(All)"] + sorted([t for t in stats_df["team"].dropna().unique() if isinstance(t,str)])
pos_opt  = ["(All)"] + sorted(stats_df["position_group"].dropna().unique().tolist())
team_sel = st.sidebar.selectbox("Team (optional)", options=team_opt)
pos_sel  = st.sidebar.selectbox("Position group (optional)", options=pos_opt)

search = st.text_input("Player name")
col_stat, col_line = st.columns([2,1])
with col_stat:
    stat_name = st.selectbox("Stat", options=list(stat_label_and_col.__globals__["STAT_MAP"].keys()), index=1)
with col_line:
    line_value = st.number_input("Sportsbook line", min_value=0.0, value=60.5, step=0.5)

with st.expander("Optional: enter market odds (American) to include vig"):
    c1, c2 = st.columns(2)
    over_odds = c1.text_input("Over odds (e.g., -115 or +100)", value="")
    under_odds = c2.text_input("Under odds (e.g., -105 or +100)", value="")

go = st.button("Analyze")
st.divider()

# Filters
filtered = stats_df.copy()
if team_sel != "(All)":
    filtered = filtered[filtered["team"]==team_sel]
if pos_sel != "(All)":
    filtered = filtered[filtered["position_group"]==pos_sel]

def find_best_candidate(df, name):
    if not name: return None
    cands = df[df["player_display_name"].str.contains(name, case=False, na=False)]
    if cands.empty: return None
    return cands.sort_values("week").tail(1).iloc[0]

if not go:
    st.info("Enter a player, stat, line → Analyze.")
else:
    cand = find_best_candidate(filtered, search)
    if cand is None:
        st.warning("No matching player found. Try clearing filters or refine the name.")
    else:
        player = cand["player_display_name"]; team = cand["team"]
        stat_label, stat_col = stat_label_and_col(stat_name)
        if stat_col not in stats_df.columns:
            st.error("Selected stat not available in dataset."); st.stop()

        st.subheader(f"{player} ({team}, {cand.get('position','')}) — {stat_label} vs Line {line_value}")
        hist = last_n_window(stats_df, player, n_weeks, week_max=week_for_matchup)
        show_cols = [c for c in ["season","week","team","opponent_team",stat_col] if c in hist.columns]
        if hist.empty:
            st.warning("Not enough recent games for historical estimate.")
        else:
            # Section 1: Historical
            p_hist, params = prob_over_normal(hist[stat_col].astype(float), float(line_value))
            c1, c2, c3 = st.columns(3)
            c1.metric("Last-N Avg", f"{hist[stat_col].mean():.1f}")
            c2.metric("Std Dev (min-reg.)", f"{params['sigma']:.1f}")
            c3.metric("Prob OVER (Historical)", f"{p_hist*100:.1f} %")
            st.markdown("**Recent games (last N)**")
            st.dataframe(hist[show_cols], use_container_width=True, height=220)

            # Context inputs
            # Opponent for selected week (use schedules if present)
            opp_team = None; game_date = ""
            if not sched_df.empty and "season" in sched_df.columns and "week" in sched_df.columns:
                s = sched_df[(sched_df.get("season", SEASON)==SEASON) &
                             (sched_df["week"]==week_for_matchup) &
                             ((sched_df.get("home_team")==team) | (sched_df.get("away_team")==team))]
                if not s.empty:
                    r = s.iloc[0]
                    opp_team = r["away_team"] if r["home_team"]==team else r["home_team"]
                    game_date = str(r.get("game_date",""))

            # Weather via stadium coords
            weather = {}
            if team in TEAM_LATLON:
                lat, lon = TEAM_LATLON[team]
                iso_time = (game_date + "T16:00") if game_date else pd.Timestamp.now().strftime("%Y-%m-%dT16:00")
                weather = fetch_weather(lat, lon, iso_time)

            player_inj = injury_flag_for_player(inj_df, player)
            def_inj = opponent_def_injuries(inj_df, opp_team) if opp_team else {}
            pace_mult = pace_factor(stats_df, team, opp_team) if opp_team else 1.0
            usage_mult = usage_trend_factor(hist, stat_col)

            def parse_odds(s):
                s = s.strip()
                if not s: return None
                try: return float(s)
                except: return None
            market_prob = None
            o_odds = parse_odds(over_odds); u_odds = parse_odds(under_odds)
            if o_odds is not None and u_odds is not None:
                market_prob = vig_to_market_prob(o_odds, u_odds)

            # Section 2: Context-Adjusted
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
            d1.metric("Prob OVER (Adjusted)", f"{details['p_final']*100:.1f} %")
            d2.metric("Adj. Mean (μ')", f"{details['mu_adjusted']:.1f}")
            d3.metric("Market Over (de-vig)", f"{(details['p_market_over']*100 if not np.isnan(details['p_market_over']) else 0):.1f} %")

            with st.expander("Explain adjustments"):
                st.markdown(f"""
**Player injury:** `{player_inj}` → **{details['m_injury']:.2f}**  
**Weather:** → **{details['m_weather']:.2f}**  
**Opponent defensive injuries:** {def_inj if def_inj else "{}"} → **{details['m_opp_def']:.2f}**  
**Pace proxy:** **{details['m_pace']:.2f}**  
**Usage trend:** **{details['m_usage']:.2f}**  
**μ, σ:** {details['mu_hist']:.1f}, {details['sigma_hist']:.1f} → μ' **{details['mu_adjusted']:.1f}**  
**Model Adj Prob:** {details['p_model_adj']*100:.1f}%  
**Market (de-vig) Over:** {("" if np.isnan(details['p_market_over']) else f"{details['p_market_over']*100:.1f}%")}  
**Final (blend 65% model / 35% market):** {details['p_final']*100:.1f}%
""")
            st.caption("Heuristics only. Not betting advice. Weather via Open-Meteo; injuries/schedules via nflverse at runtime.")
