# -*- coding: utf-8 -*-
"""
Utility helpers for NFL Parleggy Model:
- JSON loading/merging
- Column detection & standardization
- Market series mapping
- Simple AI projection + probability math
"""

import os, json, math, time
from datetime import datetime, timezone
from glob import glob
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# ---- Canonical Markets exposed in UI ----
MARKET_OPTIONS = [
    "Passing Yards",
    "Rushing Yards",
    "Receiving Yards",
    "Receptions",
    "Passing TDs",
    "Rushing TDs",
    "Receiving TDs",
    "Rushing+Receiving TDs",
]

# ---- file bookkeeping for tiny UI badges ----
_LAST_CHECK = os.path.join("data", "_last_refresh_utc.txt")
_MAE_FILE   = os.path.join("data", "_model_mae.csv")

def touch_last_checked():
    try:
        with open(_LAST_CHECK, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass

def last_checked() -> str:
    if os.path.exists(_LAST_CHECK):
        try:
            with open(_LAST_CHECK, "r") as f:
                return f.read().strip()
        except Exception:
            return "unknown"
    return "never"

def load_mae_history() -> pd.DataFrame:
    if os.path.exists(_MAE_FILE):
        try:
            df = pd.read_csv(_MAE_FILE)
            df["ts"] = pd.to_datetime(df["ts"])
            return df.tail(30)
        except Exception:
            return pd.DataFrame(columns=["ts", "mae"])
    return pd.DataFrame(columns=["ts", "mae"])

def log_mae(value: float):
    try:
        df = load_mae_history()
        df = pd.concat([df, pd.DataFrame({"ts":[datetime.now(timezone.utc)], "mae":[float(value)]})], ignore_index=True)
        df.to_csv(_MAE_FILE, index=False)
    except Exception:
        pass

# ---- JSON I/O ----
def load_all_jsons(path: str) -> pd.DataFrame:
    frames = []
    for fp in glob(os.path.join(path, "*.json")):
        try:
            with open(fp, "r") as f:
                js = json.load(f)
            frames.append(pd.json_normalize(js))
        except Exception:
            # tolerate a single bad file
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).fillna(0)
    return out

# ---- column detection helpers (very forgiving) ----
def _first_match(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand in df.columns:
            return cand
        # case-insensitive
        if cand.lower() in cols:
            return df.columns[cols.index(cand.lower())]
    # substring fallback
    for c in df.columns:
        lc = c.lower()
        if any(key in lc for key in candidates):
            return c
    return None

def detect_name_column(df: pd.DataFrame) -> str:
    c = _first_match(df, ["Name","Player","PlayerName","FullName"])
    return c or df.columns[0]

def detect_week_column(df: pd.DataFrame) -> str:
    c = _first_match(df, ["Week","GameWeek","weeknumber","Game.Week"])
    return c or df.columns[0]

def detect_opponent_column(df: pd.DataFrame) -> str:
    c = _first_match(df, ["Opponent","OpponentTeam","OppTeam","Opp","vs","OppAbbr"])
    return c or df.columns[0]

def detect_team_column(df: pd.DataFrame) -> Optional[str]:
    return _first_match(df, ["Team","TeamAbbr","TeamCode","TeamID","TeamId"])

# ---- standardize names we’ll look for later ----
_CANON_MAP = {
    # yards
    "Passing Yards": ["PassingYards","PassYds","PassYards","pass_yd","passyards","pass_yards"],
    "Rushing Yards": ["RushingYards","RushYds","RushYards","rush_yd","rushingyards","rush_yards"],
    "Receiving Yards": ["ReceivingYards","RecYds","RecYards","recv_yd","receivingyards","rec_yards"],
    # receptions
    "Receptions": ["Receptions","Rec","Catches","receptions","recs","targets_caught"],
    # TDs
    "Passing TDs": ["PassingTouchdowns","PassTD","PassTDs","pass_tds","pass_td"],
    "Rushing TDs": ["RushingTouchdowns","RushTD","RushTDs","rush_tds","rush_td"],
    "Receiving TDs": ["ReceivingTouchdowns","RecTD","RecTDs","recv_tds","rec_td"],
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Light sanitation: strip spaces, coerce numerics where sensible."""
    if df.empty:
        return df
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    # try numeric coercion for any stat-like column
    for c in out.columns:
        if any(k.lower() in str(c).lower() for k in ["yard","yd","td","rec","cmp","att","mean","avg","proj","score"]):
            out[c] = pd.to_numeric(out[c], errors="ignore")
    return out

# ---- market accessors ----
def get_market_series(df: pd.DataFrame, market: str) -> Optional[pd.Series]:
    """Return a numeric Series for the chosen market, or None if not found."""
    if df is None or df.empty:
        return None

    if market == "Rushing+Receiving TDs":
        r = _series_for(df, "Rushing TDs")
        e = _series_for(df, "Receiving TDs")
        if r is None and e is None:
            return None
        r = r if r is not None else pd.Series([0]*len(df))
        e = e if e is not None else pd.Series([0]*len(df))
        return (pd.to_numeric(r, errors="coerce").fillna(0) + pd.to_numeric(e, errors="coerce").fillna(0))

    return _series_for(df, market)

def _series_for(df: pd.DataFrame, market: str) -> Optional[pd.Series]:
    # exact column?
    if market in df.columns:
        return pd.to_numeric(df[market], errors="coerce")

    # try canonical list
    if market in _CANON_MAP:
        for alt in _CANON_MAP[market]:
            c = _first_match(df, [alt])
            if c:
                return pd.to_numeric(df[c], errors="coerce")
    # last-ditch substring search
    key = market.split()[0].lower()  # 'passing', 'rushing', 'receiving', etc.
    for c in df.columns:
        if key in c.lower() and any(k in c.lower() for k in ["yd","yard","td","rec"]):
            return pd.to_numeric(df[c], errors="coerce")
    return None

def columns_for_table(df: pd.DataFrame, market: str, name_col: str, opp_col: str, week_col: str) -> List[str]:
    cols = [week_col, name_col]
    m = get_market_series(df, market)
    if m is not None:
        # find the actual source column name we used
        src = m.name if isinstance(m, pd.Series) and m.name else None
        if src and src in df.columns and src not in cols:
            cols.append(src)
    if opp_col in df.columns:
        cols.append(opp_col)
    # add a couple of useful columns if present
    for extra in ["ScoreID","Score","PlayerGameID","FanDuelSalary","DraftKingsSalary","FantasyDataSalary","GlobalOpponentID","OpponentID"]:
        if extra in df.columns and extra not in cols:
            cols.append(extra)
    # dedupe while keeping order
    seen = set(); out = []
    for c in cols:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

# ---- slicing & features ----
def slice_player(data: pd.DataFrame, player: str, opponent: str, lookback_weeks: int, week_col: str, name_col: str) -> pd.DataFrame:
    df = data.copy()
    df = df[df[name_col].astype(str).str.lower() == str(player).lower()]
    if opponent:
        opp_col = detect_opponent_column(data)
        if opp_col in df.columns:
            df = df[df[opp_col].astype(str).str.contains(opponent, case=False, na=False)]
    # recent N weeks
    if week_col in df.columns:
        try:
            df = df.sort_values(week_col, ascending=False)
        except Exception:
            df = df.sort_values(by=df.index, ascending=False)
    return df.head(max(lookback_weeks, 1)).reset_index(drop=True)

# ---- AI-ish projection (robust, fast) ----
def predict_market(pdf: pd.DataFrame, target: pd.Series, market: str) -> float:
    """
    If ≥ 6 samples, use weighted moving average (heavier on last 3).
    Else simply mean of available.
    (Keeping it fast & stable; your repo already has xgboost in requirements
     so you can extend this to a full model later.)
    """
    y = pd.to_numeric(target, errors="coerce").dropna()
    if y.empty:
        return 0.0
    n = len(y)
    if n >= 6:
        w = np.linspace(1.0, 2.0, num=n)  # up-weight recent games
        return float(np.average(y.values[::-1], weights=w))  # recent first
    return float(y.mean())

# ---- probability math (Normal approx; no SciPy) ----
def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-6:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

def prob_over_under(target: pd.Series, mu: float, line: float, market: str) -> Tuple[float, float, float]:
    """Return (P_over, P_under, sigma)."""
    y = pd.to_numeric(target, errors="coerce").dropna()
    if y.empty:
        return 0.5, 0.5, 1.0
    # robust std: if too small, inflate slightly to avoid degenerate probs
    sigma = float(np.nanstd(y.values, ddof=1))
    if not np.isfinite(sigma) or sigma < 1.0:
        sigma = max(1.0, abs(mu) * 0.20)
    p_over = 1.0 - _normal_cdf(line, mu, sigma)
    p_over = float(np.clip(p_over, 0.0, 1.0))
    return p_over, 1.0 - p_over, sigma

# ---- confidence score ----
def confidence_score(series: pd.Series, sigma: float, lookback: int) -> Tuple[float, List[str]]:
    y = pd.to_numeric(series, errors="coerce").dropna()
    n = len(y)
    if n == 0:
        return 0.0, ["no samples"]
    base = min(1.0, n / 8.0)  # saturate by 8 samples
    dispersion = 1.0 / (1.0 + (sigma / (abs(np.mean(y)) + 1e-6)))
    recent = min(1.0, lookback / 8.0)
    score = 0.5 * base + 0.3 * dispersion + 0.2 * recent
    return float(score), []
