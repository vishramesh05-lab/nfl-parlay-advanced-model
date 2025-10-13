# -*- coding: utf-8 -*-
# Utilities for NFL Parleggy AI Model
import os, json, glob, math
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from dateutil import tz
from scipy.stats import norm, poisson

# ---------- Paths ----------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data")
RETRAIN_STAMP = "/tmp/parleggy_last_retrain.txt"

# ---------- Retrain stamp ----------
def touch_retrain_stamp() -> None:
    try:
        with open(RETRAIN_STAMP, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass

def get_last_retrain_time() -> str | None:
    try:
        if not os.path.exists(RETRAIN_STAMP):
            return None
        ts = open(RETRAIN_STAMP).read().strip()
        dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        return dt.strftime("%b %d %Y %H:%M UTC")
    except Exception:
        return None

def _today_12am_est_utc() -> datetime:
    # 12:00 AM EST today in UTC (EST = UTC-5; we ignore DST here for simplicity)
    now_utc = datetime.now(timezone.utc)
    today = now_utc.date()
    # 05:00 UTC corresponds roughly to 00:00 EST (standard time)
    return datetime(today.year, today.month, today.day, 5, 0, 0, tzinfo=timezone.utc)

def maybe_mark_nightly_retrain():
    try:
        target = _today_12am_est_utc()
        last = None
        if os.path.exists(RETRAIN_STAMP):
            last = datetime.fromisoformat(open(RETRAIN_STAMP).read().strip().replace("Z","+00:00"))
        if (not last) or (last < target):
            touch_retrain_stamp()
    except Exception:
        pass

# ---------- JSON Ingestion ----------
def _read_json_any(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _to_df_from_obj(obj) -> pd.DataFrame:
    # Accept list-of-dicts or dict-with-list inside, or flat dict
    if isinstance(obj, list):
        return pd.json_normalize(obj)
    if isinstance(obj, dict):
        # If any list value, take the first list-like
        for k, v in obj.items():
            if isinstance(v, list):
                return pd.json_normalize(v)
        return pd.json_normalize(obj)
    return pd.DataFrame()

def load_all_jsons() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_PATH, "*.json")))
    frames = []
    for fp in files:
        try:
            js = _read_json_any(fp)
            df = _to_df_from_obj(js)
            if not df.empty:
                df["_source_file"] = os.path.basename(fp)
                frames.append(df)
        except Exception:
            # skip bad files but keep going
            continue
    if frames:
        out = pd.concat(frames, ignore_index=True).fillna(0)
        # unify column names to a canonical set (lower for detection)
        out.columns = [str(c) for c in out.columns]
        return out
    return pd.DataFrame()

# ---------- Column detection ----------
def _detect_one(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = [c for c in df.columns]
    # exact first
    for c in candidates:
        if c in cols: return c
    # lower match
    lc = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lc: return lc[c.lower()]
    # contains match
    for c in cols:
        for k in candidates:
            if k.lower() in c.lower():
                return c
    return None

def detect_name_column(df) -> str | None:
    # common player name fields
    return _detect_one(df, ["Name", "Player", "FullName", "PlayerName"])

def detect_pos_column(df) -> str | None:
    return _detect_one(df, ["Position", "Pos", "PlayerPosition"])

def detect_team_column(df) -> str | None:
    return _detect_one(df, ["Team", "TeamAbbr", "GlobalTeamID", "HomeTeam", "TeamID"])

def detect_opp_column(df) -> str | None:
    return _detect_one(df, ["Opponent", "OpponentAbbr", "GlobalOpponentID", "AwayTeam", "OpponentID"])

def detect_week_column(df) -> str | None:
    return _detect_one(df, ["Week", "WeekNumber", "GameWeek"])

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure types and helpful derived fields
    out = df.copy()
    wk = detect_week_column(out)
    if wk:
        out[wk] = pd.to_numeric(out[wk], errors="coerce").fillna(0).astype(int)
    # unify touchdowns combos for convenience (if present)
    rush_td = find_stat_column(out, "Rushing TDs")
    rec_td  = find_stat_column(out, "Receiving TDs")
    if rush_td or rec_td:
        out["_RR_TDs"] = 0.0
        if rush_td: out["_RR_TDs"] += pd.to_numeric(out[rush_td], errors="coerce").fillna(0)
        if rec_td:  out["_RR_TDs"] += pd.to_numeric(out[rec_td], errors="coerce").fillna(0)
    return out

# ---------- Market ➜ column mapping ----------
# We search columns by flexible keywords to succeed across JSON variations.
def find_stat_column(df: pd.DataFrame, market: str) -> str | None:
    cols = list(df.columns)

    # helper: search by multiple keyword alternatives (all must appear in the column)
    def by_keywords(options: list[list[str]]):
        # options is list of AND-keyword lists to try
        low = [c.lower() for c in cols]
        for and_keys in options:
            for i, lc in enumerate(low):
                if all(k.lower() in lc for k in and_keys):
                    return cols[i]
        return None

    if market == "Passing Yards":
        return by_keywords([["passing", "yard"], ["pass", "yard"], ["py"]]) or \
               _detect_one(df, ["PassingYards", "PassYards"])

    if market == "Rushing Yards":
        return by_keywords([["rushing", "yard"], ["rush", "yard"], ["ry"]]) or \
               _detect_one(df, ["RushingYards"])

    if market == "Receiving Yards":
        return by_keywords([["receiving", "yard"], ["recv", "yard"], ["rec", "yard"]]) or \
               _detect_one(df, ["ReceivingYards"])

    if market == "Receptions":
        return by_keywords([["receptions"], ["reception"], ["rec"]]) or \
               _detect_one(df, ["Receptions"])

    if market == "Passing TDs":
        return by_keywords([["passing", "td"], ["pass", "td"]]) or _detect_one(df, ["PassingTouchdowns"])

    if market == "Rushing TDs":
        return by_keywords([["rushing", "td"], ["rush", "td"]]) or _detect_one(df, ["RushingTouchdowns"])

    if market == "Receiving TDs":
        return by_keywords([["receiving", "td"], ["recv", "td"], ["rec", "td"]]) or _detect_one(df, ["ReceivingTouchdowns"])

    if market == "Rushing+Receiving TDs":
        # synthetic handled later using _RR_TDs
        if "_RR_TDs" in df.columns:
            return "_RR_TDs"
        # fallback find individually
        r = find_stat_column(df, "Rushing TDs")
        rc = find_stat_column(df, "Receiving TDs")
        if r or rc:
            return "_RR_TDs"  # will be created by standardize_columns
        return None

    return None

# ---------- Series extraction & probability ----------
def get_player_market_series(df: pd.DataFrame, player_name: str, market: str,
                             opponent: str|None, lookback_weeks: int):
    """Return (series, meta) for given player/market filtered to last N weeks."""
    name_col = detect_name_column(df)
    if not name_col:
        return pd.Series(dtype=float), {"recent_df": pd.DataFrame()}

    # Filter player rows
    d = df[df[name_col].astype(str).str.lower() == str(player_name).lower()].copy()
    # Opp filter if we have an opp column + user provided one
    opp_col = detect_opp_column(df)
    if opponent and opp_col:
        d = d[d[opp_col].astype(str).str.upper() == str(opponent).upper()]

    # sort by inferred "Week" or row order
    wk = detect_week_column(df)
    if wk and wk in d.columns:
        d = d.sort_values(wk, ascending=False)
    else:
        d = d.reset_index(drop=True)

    # detect stat column
    col = find_stat_column(df, market)
    if (not col) or (col not in d.columns):
        # give empty series
        return pd.Series(dtype=float), {"recent_df": pd.DataFrame()}

    # sanitize numeric
    s = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    # take last N entries
    s_lb = s.head(int(max(1, lookback_weeks)))

    # Build recent table with some common columns
    show_cols = [c for c in [detect_week_column(df), name_col, detect_opp_column(df),
                             "PassingYards", "RushingYards", "ReceivingYards",
                             "PassingTouchdowns", "RushingTouchdowns", "ReceivingTouchdowns"] if c in d.columns]
    recent = d.loc[s_lb.index, show_cols].copy() if show_cols else pd.DataFrame()

    return s_lb.reset_index(drop=True), {
        "recent_df": recent
    }

def probability_from_history(series: pd.Series, market: str, line: float) -> dict:
    """Return dict with p_over, p_under, pred_mean, histogram df (or None)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    out = {"p_over": 0.5, "p_under": 0.5, "pred_mean": 0.0, "histogram": None}
    if s.empty:
        return out

    mean = float(np.mean(s))
    out["pred_mean"] = mean

    # TD markets are count-like -> Poisson is a decent approximation
    if "TDs" in market:
        lam = max(1e-6, float(np.mean(s)))
        # P(X >= k) with k as ceil(line + 1e-9)
        k = int(math.floor(line + 1e-9) + 1)
        p_ge = 1.0 - poisson.cdf(k-1, lam)
        out["p_over"] = float(p_ge)
        out["p_under"] = float(1.0 - p_ge)
    else:
        sd = float(np.std(s, ddof=1)) if len(s) > 1 else max(1.0, abs(mean)*0.25)
        # Normal approx
        z = (line - mean) / (sd + 1e-9)
        p_under = float(norm.cdf(z))
        p_over  = float(1.0 - p_under)
        out["p_over"] = p_over
        out["p_under"] = p_under

    # histogram for preview
    try:
        out["histogram"] = pd.DataFrame({"value": s})
    except Exception:
        out["histogram"] = None

    return out

def confidence_from_series(series: pd.Series, market: str) -> tuple[float, str]:
    """Return (confidence in 0..1, label). Uses sample size and dispersion."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0, "Low Confidence"
    n = len(s)
    if "TDs" in market:
        # dispersion measured by sqrt(lambda)
        lam = np.mean(s)
        disp = math.sqrt(lam + 1e-9)
    else:
        m = np.mean(s)
        sd = np.std(s) if n > 1 else max(1.0, abs(m)*0.25)
        disp = sd / (abs(m) + 1e-9)
    # combine
    conf = min(1.0, 0.25 + 0.1*n + 0.4*(1.0/(1.0+disp)))
    label = "High Confidence" if conf >= 0.75 else ("Moderate Confidence" if conf >= 0.5 else "Low Confidence")
    return float(conf), label

# ---------- Parlay correlation ----------
def joint_probability_with_correlation(probs: list[float], labels: list[tuple[str,str,str]]) -> float:
    """
    Conservative adjustment:
    - Multiply raw probabilities
    - For any pair in same team or same player, apply a 0.95 penalty (light positive correlation)
    - Cap result to [1e-6, 1-1e-6]
    """
    base = 1.0
    for p in probs:
        base *= max(1e-6, min(1-1e-6, p))

    # light correlation penalties
    penalty = 1.0
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            p1, opp1, m1 = labels[i]
            p2, opp2, m2 = labels[j]
            if (p1 == p2) or (opp1 == opp2) or (m1 == m2):
                penalty *= 0.95  # 5% haircut

    out = max(1e-6, min(1-1e-6, base * penalty))
    return float(out)
