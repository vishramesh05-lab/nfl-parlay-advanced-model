
import pandas as pd
import numpy as np
import math, datetime, time, json, requests
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from scipy.stats import norm

# --- Basic stat mappings ---
STAT_MAP = {
    "Passing Yards": "passing_yards",
    "Rushing Yards": "rushing_yards",
    "Receiving Yards": "receiving_yards",
    "Receptions": "receptions",
    "Passing TDs": "passing_tds",
    "Rushing TDs": "rushing_tds",
    "Receiving TDs": "receiving_tds"
}
POS_FOR_STAT = {
    "Passing Yards":"QB","Passing TDs":"QB",
    "Rushing Yards":"RB","Rushing TDs":"RB",
    "Receiving Yards":"WR","Receiving TDs":"WR","Receptions":"WR"
}

NUMERIC_COLS = [
    "passing_yards","passing_tds","interceptions",
    "rushing_yards","rushing_tds",
    "receptions","receiving_yards","receiving_tds",
    "targets","air_yards","fantasy_points_ppr"
]

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = 0.0
    if "recent_team" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"recent_team":"team"})
    if "opponent_team" not in df.columns and "opponent" in df.columns:
        df["opponent_team"] = df["opponent"]
    if "player_display_name" not in df.columns:
        if "player_name" in df.columns:
            df["player_display_name"] = df["player_name"]
        else:
            df["player_display_name"] = df.get("player", "Unknown")
    if "position" not in df.columns:
        df["position"] = df.get("pos","UNK")
    if "position_group" not in df.columns:
        df["position_group"] = df["position"].astype(str).str.replace(r"\d+","", regex=True)
    if "team" not in df.columns:
        df["team"] = df.get("team_abbr","UNK")
    return df

def last_n_window(df: pd.DataFrame, player_name: str, n: int, week_max: Optional[int]=None) -> pd.DataFrame:
    sub = df[df["player_display_name"]==player_name].copy()
    if week_max is not None and "week" in sub.columns:
        sub = sub[sub["week"] <= week_max]
    sub = sub.sort_values("week").tail(n)
    return sub

def prob_over_normal(samples: pd.Series, line: float, min_std: float=5.0, shrink: float=0.2) -> Tuple[float, Dict[str,float]]:
    """Return probability over line under a regularized normal; also return mu/sigma used."""
    if len(samples) == 0:
        return float("nan"), {"mu": float("nan"), "sigma": float("nan")}
    mu = float((1.0 - shrink) * samples.mean() + shrink * samples.median())
    sigma = float(max(samples.std(ddof=1) if samples.size>1 else min_std, min_std))
    p_over = float(1.0 - norm.cdf(line, loc=mu, scale=sigma))
    return p_over, {"mu":mu, "sigma":sigma}

# --- Team → approximate stadium coordinates (lat, lon). (Basic list; sufficient for weather fetch.) ---
TEAM_LATLON = {
    "ARI": (33.5275, -112.2625), "ATL": (33.7555, -84.4010), "BAL": (39.2779, -76.6227),
    "BUF": (42.7738, -78.7868), "CAR": (35.2251, -80.8526), "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3390, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3240, -81.6370),
    "KC":  (39.0489, -94.4840), "LV":  (36.0908, -115.183), "LAC": (34.0139, -118.285),
    "LAR": (34.0139, -118.285), "MIA": (25.9570, -80.2389), "MIN": (44.9733, -93.2573),
    "NE":  (42.0909, -71.2643), "NO":  (29.9509, -90.0810), "NYG": (40.8128, -74.0742),
    "NYJ": (40.8128, -74.0742), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316), "SF":  (37.4030, -121.9700), "TB":  (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9078, -76.8645)
}

def fetch_weather(lat: float, lon: float, iso_datetime: str) -> Dict[str, Any]:
    """Get hourly weather forecast for game time using open-meteo. iso_datetime: 'YYYY-MM-DDTHH:00' local stadium time.
       Returns dict with wind_speed, precipitation, temperature."""
    try:
        # Open-Meteo hourly API; use UTC time for simplicity and get nearest hour
        # We ask for wind_speed_10m, precipitation, temperature_2m
        base = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "wind_speed_10m,precipitation,temperature_2m",
            "forecast_days": 1, "timezone": "auto"
        }
        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        # get closest hour
        times = data.get("hourly",{}).get("time",[])
        winds = data.get("hourly",{}).get("wind_speed_10m",[])
        precs = data.get("hourly",{}).get("precipitation",[])
        temps = data.get("hourly",{}).get("temperature_2m",[])
        if not times:
            return {}
        # pick mid-afternoon local by default if iso_datetime unspecified
        target_idx = min(range(len(times)), key=lambda i: abs(pd.Timestamp(times[i]) - pd.Timestamp(iso_datetime)))
        return {
            "wind_speed_10m": winds[target_idx] if target_idx < len(winds) else None,
            "precipitation": precs[target_idx] if target_idx < len(precs) else None,
            "temperature_2m": temps[target_idx] if target_idx < len(temps) else None,
            "timestamp": times[target_idx]
        }
    except Exception:
        return {}

def injury_flag_for_player(inj_df: pd.DataFrame, player_name: str) -> str:
    if inj_df is None or inj_df.empty:
        return "unknown"
    sub = inj_df[inj_df["player_display_name"].str.contains(player_name, case=False, na=False)]
    if sub.empty:
        return "unknown"
    # take the most recent status
    sub = sub.sort_values("report_date" if "report_date" in sub.columns else "week").tail(1)
    s = str(sub.iloc[0].get("status", "unknown")).lower()
    if "out" in s or "ir" in s or "inactive" in s or "doubt" in s:
        return "out/doubt"
    if "question" in s or "lp" in s:
        return "questionable"
    if "probable" in s or "fp" in s or "active" in s or "full" in s:
        return "probable"
    return "unknown"

def opponent_def_injuries(inj_df: pd.DataFrame, opp_team: str) -> Dict[str,int]:
    """Count # of defensive starters with negative status (OUT/Doubt/IR) for opp team by broad position groups."""
    if inj_df is None or inj_df.empty or opp_team is None:
        return {}
    df = inj_df.copy()
    # filter team
    if "team" in df.columns:
        df = df[df["team"]==opp_team]
    # positions that affect stats broadly
    def_neg = df[df["status"].astype(str).str.contains("Out|IR|Doubt", case=False, na=False)]
    # map positions to buckets
    buckets = {"CB":0,"S":0,"LB":0,"DL":0,"EDGE":0}
    for _, r in def_neg.iterrows():
        pos = str(r.get("position","")).upper()
        for b in list(buckets.keys()):
            if b in pos:
                buckets[b]+=1
    return {k:int(v) for k,v in buckets.items()}

def pace_factor(stats_df: pd.DataFrame, team: str, opp: str) -> float:
    """Very rough pace proxy: relative #player-weeks per game for team vs league avg.
       Returns multiplier ~1.0 = neutral, >1 faster, <1 slower."""
    if stats_df is None or stats_df.empty:
        return 1.0
    # count appearances per team-week
    tmp = stats_df.groupby(["team","week"], as_index=False).size().rename(columns={"size":"appearances"})
    team_avg = tmp[tmp["team"]==team]["appearances"].mean()
    opp_avg = tmp[tmp["team"]==opp]["appearances"].mean()
    league_avg = tmp["appearances"].mean()
    if pd.isna(team_avg) or pd.isna(opp_avg) or pd.isna(league_avg) or league_avg==0:
        return 1.0
    # blend team + opp pace vs league
    rel = ((team_avg + opp_avg)/2.0) / league_avg
    return float(np.clip(rel, 0.9, 1.1))  # clamp to avoid extremes

def usage_trend_factor(hist: pd.DataFrame, stat_col: str) -> float:
    """If last 3 games average > prior 3 by a margin, boost. Otherwise neutral.
       Returns multiplier ~1.0 neutral, up to 1.1 boost or 0.95 dampen."""
    if hist is None or hist.empty or "week" not in hist.columns:
        return 1.0
    h = hist.sort_values("week")
    if len(h) < 4:
        return 1.0
    last3 = h.tail(3)[stat_col].mean()
    prev3 = h.tail(6).head(3)[stat_col].mean() if len(h)>=6 else h.head(max(1, len(h)-3))[stat_col].mean()
    if prev3==0 or pd.isna(prev3) or pd.isna(last3):
        return 1.0
    delta = (last3 - prev3)/max(1.0, prev3)
    if delta > 0.25:
        return 1.1
    if delta < -0.25:
        return 0.95
    return 1.0

def weather_factor(weather: Dict[str,Any], stat_label: str) -> float:
    """Heuristics by stat & weather. Wind/precip hurt passing the most; mild boost to rushing in bad weather."""
    if not weather:
        return 1.0
    wind = weather.get("wind_speed_10m")
    precip = weather.get("precipitation")
    temp = weather.get("temperature_2m")
    m = 1.0
    if wind is not None:
        if "Passing" in stat_label or "Receptions" in stat_label or "Receiving" in stat_label:
            if wind >= 20: m *= 0.92
            elif wind >= 12: m *= 0.96
    if precip is not None and precip >= 1.0:
        if "Passing" in stat_label or "Receiving" in stat_label or "Receptions" in stat_label:
            m *= 0.95
        if "Rushing" in stat_label:
            m *= 1.04
    # extreme cold slight penalty to passing
    if temp is not None and temp <= 25 and ("Passing" in stat_label):
        m *= 0.97
    return float(m)

def injury_factor_player(flag: str) -> float:
    if flag == "out/doubt":
        return 0.6
    if flag == "questionable":
        return 0.9
    return 1.0

def opponent_defense_factor(def_inj: Dict[str,int], stat_label: str) -> float:
    """If opponent missing multiple DBs → passing/receiving boost; missing LBs/DL → rushing boost."""
    if not def_inj:
        return 1.0
    cb = def_inj.get("CB",0); s = def_inj.get("S",0); lb = def_inj.get("LB",0); dl = def_inj.get("DL",0); ed = def_inj.get("EDGE",0)
    m = 1.0
    if "Passing" in stat_label or "Receiving" in stat_label or "Receptions" in stat_label:
        if cb + s >= 2: m *= 1.06
    if "Rushing" in stat_label:
        if (dl + lb + ed) >= 2: m *= 1.05
    return float(m)

def vig_to_market_prob(odds_over: Optional[float], odds_under: Optional[float]) -> Optional[float]:
    """Convert American odds for Over/Under to a market-implied Over probability (vigged), then de-vig with proportional method.
       Return None if inputs missing."""
    if odds_over is None or odds_under is None:
        return None
    def to_prob(american):
        if american is None: return None
        a = float(american)
        if a > 0: return 100.0/(a+100.0)
        else: return (-a)/( -a + 100.0 )
    p_o = to_prob(odds_over)
    p_u = to_prob(odds_under)
    if p_o is None or p_u is None: return None
    # de-vig
    s = p_o + p_u
    if s <= 0: return None
    p_over_novig = p_o / s
    return float(p_over_novig)

def blend_with_market(p_hist: float, p_market: Optional[float], weight: float=0.35) -> float:
    if p_market is None or np.isnan(p_hist):
        return p_hist
    return float((1.0-weight)*p_hist + weight*p_market)

def context_adjusted_probability(
    hist_samples: pd.Series,
    line_value: float,
    stat_label: str,
    player_injury_flag: str,
    weather_dict: Dict[str,Any],
    opp_def_inj: Dict[str,int],
    pace_mult: float,
    usage_mult: float,
    market_prob_over: Optional[float]
) -> Tuple[float, Dict[str, float]]:
    """Compute adjusted probability: scale mu by multipliers, then compute prob; blend toward market probability."""
    p_hist, params = prob_over_normal(hist_samples, line_value)
    mu = params["mu"]; sigma = params["sigma"]
    # Multiplicative adjustments on mean
    m_injury = injury_factor_player(player_injury_flag)
    m_weather = weather_factor(weather_dict, stat_label)
    m_opp = opponent_defense_factor(opp_def_inj, stat_label)
    m_pace = float(pace_mult)
    m_usage = float(usage_mult)

    mu_adj = mu * m_injury * m_weather * m_opp * m_pace * m_usage
    p_adj_model = float(1.0 - norm.cdf(line_value, loc=mu_adj, scale=sigma))
    # blend with market
    p_final = blend_with_market(p_adj_model, market_prob_over, weight=0.35)

    details = {
        "mu_hist": mu, "sigma_hist": sigma,
        "m_injury": m_injury, "m_weather": m_weather, "m_opp_def": m_opp,
        "m_pace": m_pace, "m_usage": m_usage,
        "mu_adjusted": mu_adj,
        "p_hist": p_hist, "p_model_adj": p_adj_model, "p_market_over": (market_prob_over if market_prob_over is not None else float("nan")),
        "p_final": p_final
    }
    return p_final, details

def stat_label_and_col(stat_name: str):
    return stat_name, STAT_MAP.get(stat_name, None)
