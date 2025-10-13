import os, json, math
import numpy as np
import pandas as pd

# --- Market keyword mapping ---
MARKET_KEYWORDS = {
    "Passing Yards": [["pass"], ["yd"]],
    "Rushing Yards": [["rush"], ["yd"]],
    "Receiving Yards": [["receiv"], ["yd"]],
    "Receptions": [["rec"], []],
    "Completions": [["comp"], []],
    "Pass Attempts": [["att","pass"], []],
    "Carries": [["rush","att"], []],
    "Passing TDs": [["pass"], ["td","touch"]],
    "Rushing TDs": [["rush"], ["td","touch"]],
    "Receiving TDs": [["receiv"], ["td","touch"]],
}

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

def _list_week_files(path: str):
    return sorted(f for f in os.listdir(path) if f.lower().endswith(".json"))

def load_all_jsons() -> pd.DataFrame:
    frames = []
    for fname in _list_week_files(DATA_PATH):
        fpath = os.path.join(DATA_PATH, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                js = json.load(f)
        except Exception:
            continue
        if isinstance(js, list):
            df = pd.json_normalize(js)
        elif isinstance(js, dict):
            for key in ("Players","players","data","Data","Items"):
                if key in js and isinstance(js[key], list):
                    df = pd.json_normalize(js[key]); break
            else:
                df = pd.json_normalize(js)
        else:
            continue
        df["__source_file"] = fname
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

# --- Column detection ---
NAME_CANDIDATES = ["Name","Player","PlayerName","FullName"]
WEEK_CANDIDATES = ["Week","GameWeek"]
TEAM_CANDIDATES = ["Team","TeamAbbr","TeamName"]
OPP_CANDIDATES  = ["Opponent","Opp","OpponentTeam","GlobalOpponentID"]

def _detect_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low: return low[c.lower()]
    return None

def detect_name_column(df): return _detect_col(df, NAME_CANDIDATES)
def detect_week_column(df): return _detect_col(df, WEEK_CANDIDATES)
def detect_team_column(df): return _detect_col(df, TEAM_CANDIDATES)
def detect_opp_column(df):  return _detect_col(df, OPP_CANDIDATES)

def standardize_columns(df):
    n, w = detect_name_column(df), detect_week_column(df)
    if n and n != "Name": df = df.rename(columns={n:"Name"})
    if w and w != "Week": df = df.rename(columns={w:"Week"})
    if "Week" in df.columns:
        df["Week"] = pd.to_numeric(df["Week"], errors="coerce")
    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str).str.strip()
    return df

def find_target_column(df, market):
    req = MARKET_KEYWORDS.get(market)
    if not req: return None
    lcmap = {c.lower(): c for c in df.columns}
    for lc, orig in lcmap.items():
        if all(any(k in lc for k in group) for group in req):
            return orig
    return None

# --- Probability / confidence helpers ---
def normal_over_probability(samples, line):
    x = np.array(samples, dtype=float)
    x = x[~np.isnan(x)]
    if not len(x): return float("nan")
    mu, sd = np.mean(x), np.std(x, ddof=1) if len(x)>1 else 0.0
    if sd < 1e-9: return 1.0 if mu > line else 0.0
    z = (line - mu)/(sd*math.sqrt(2))
    return 0.5 * math.erfc(z)

def confidence_score(samples):
    x = np.array(samples, dtype=float)
    x = x[~np.isnan(x)]
    if not len(x): return 0.0
    n, sd = len(x), np.std(x, ddof=1) if len(x)>1 else 0.0
    base = min(1.0, math.sqrt(n)/4.0)
    penalty = 1.0/(1.0+sd)
    return max(0.0, min(1.0, base*penalty))
