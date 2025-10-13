# utils.py
import os, json, math, time
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

# ---------- small persistence for "last checked" ----------
_LAST_CHECKED_FILE = "/tmp/parleggy_last_checked.txt"

def last_checked() -> str:
    try:
        with open(_LAST_CHECKED_FILE, "r") as f:
            return f.read().strip()
    except Exception:
        return "never"

def touch_last_checked() -> None:
    try:
        with open(_LAST_CHECKED_FILE, "w") as f:
            f.write(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    except Exception:
        pass

def load_mae_history() -> pd.DataFrame:
    # placeholder chart source (kept minimal so app always renders)
    return pd.DataFrame(columns=["ts", "mae"])


# ---------- JSON loading & normalization ----------
def _json_to_df(obj) -> pd.DataFrame:
    """
    Robustly normalize SportsDataIO-ish JSON payloads.
    Accepts list[dict] or dict (with nested 'Data' or similar).
    """
    if isinstance(obj, list):
        return pd.json_normalize(obj)
    if isinstance(obj, dict):
        # try common top-level keys first
        for k in ("data", "Data", "players", "Players", "teams", "Teams", "games", "Games"):
            if k in obj and isinstance(obj[k], list):
                return pd.json_normalize(obj[k])
        # fallback: normalize the dict itself
        return pd.json_normalize(obj)
    return pd.DataFrame()

def load_all_jsons(path: str) -> pd.DataFrame:
    frames = []
    if not os.path.isdir(path):
        return pd.DataFrame()
    for fn in os.listdir(path):
        if not fn.lower().endswith(".json"):
            continue
        fp = os.path.join(path, fn)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            df = _json_to_df(obj)
            if not df.empty:
                df["__source_file"] = fn
                frames.append(df)
        except Exception:
            # ignore malformed files but keep going
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out

# ---------- column detection & standardization ----------
NAME_CANDS = ["Name", "Player", "PlayerName", "FullName", "DisplayName", "search_full_name"]
FN_CANDS   = ["FirstName", "first_name"]
LN_CANDS   = ["LastName", "last_name"]
POS_CANDS  = ["Position", "Pos"]
TEAM_CANDS = ["Team", "TeamAbbr", "TeamName", "TeamShort", "GlobalTeamID", "TeamID"]
OPP_CANDS  = ["Opponent", "Opp", "OpponentTeam", "OpponentAbbr", "OpponentName", "GlobalOpponentID", "OpponentID"]
WEEK_CANDS = ["Week", "GameWeek", "week"]

# market → list of candidate stat columns (we try in order and also case-insensitive)
MARKET_COLS: Dict[str, List[str]] = {
    "Passing Yards": [
        "PassingYards", "PassYards", "Passing_Yards", "PYds", "PassYds", "Yards", "YdsPass", "PlayerPassingYards"
    ],
    "Rushing Yards": [
        "RushingYards", "RushYards", "Rushing_Yards", "RYds", "PlayerRushingYards"
    ],
    "Receiving Yards": [
        "ReceivingYards", "RecYards", "Receiving_Yards", "PlayerReceivingYards"
    ],
    "Receptions": [
        "Receptions", "Rec", "Catches"
    ],
    "Completions": [
        "PassingCompletions", "Completions", "Cmp"
    ],
    "Attempts": [
        "PassingAttempts", "PassAttempts", "Attempts", "Att"
    ],
    "Passing TDs": [
        "PassingTouchdowns", "PassingTDs", "PassTD", "PassTouchdowns"
    ],
    "Rushing TDs": [
        "RushingTouchdowns", "RushingTDs", "RushTD"
    ],
    "Receiving TDs": [
        "ReceivingTouchdowns", "ReceivingTDs", "RecTD"
    ],
    "Rushing+Receiving YDs": [
        # handled as sum if both exist; else fall back to whichever we find
        "Rushing+ReceivingYards"  # synthetic marker
    ],
    "Rushing+Receiving TDs": [
        "Rushing+ReceivingTouchdowns"  # synthetic marker
    ],
}

TEAM_ABBR = {
    # simple 3-letter normalization (expand if your JSON uses full names)
    "ARI":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR","CHI":"CHI","CIN":"CIN","CLE":"CLE","DAL":"DAL",
    "DEN":"DEN","DET":"DET","GB":"GB","HOU":"HOU","IND":"IND","JAX":"JAX","KC":"KC","LV":"LV","LAC":"LAC","LAR":"LAR",
    "MIA":"MIA","MIN":"MIN","NE":"NE","NO":"NO","NYG":"NYG","NYJ":"NYJ","PHI":"PHI","PIT":"PIT","SEA":"SEA","SF":"SF",
    "TB":"TB","TEN":"TEN","WAS":"WAS","WSH":"WAS","COMMANDERS":"WAS"
}
def _coerce_opp(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip().upper()
    s = s.replace("WASHINGTON", "WAS").replace("WFT","WAS")
    s = s.replace("JACKSONVILLE","JAX").replace("LA CHARGERS","LAC").replace("CHARGERS","LAC")
    s = s.replace("LA RAMS","LAR").replace("RAMS","LAR")
    s = s.replace("NEW ENGLAND","NE").replace("NEW ORLEANS","NO")
    # if 3-letter already:
    if len(s) == 3 and s in TEAM_ABBR:
        return s
    # pick first 3 letters if matches known
    if s in TEAM_ABBR:
        return TEAM_ABBR[s]
    if len(s) >= 3:
        return s[:3]
    return s

def _first_present(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    # exact
    for c in cands:
        if c in df.columns: return c
    # case-insensitive
    lc = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in lc: return lc[c.lower()]
    return None

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()
    out = df.copy()

    # Build Player column
    name_col = _first_present(out, NAME_CANDS)
    if not name_col:
        fn, ln = _first_present(out, FN_CANDS), _first_present(out, LN_CANDS)
        if fn and ln:
            out["Player"] = (out[fn].astype(str).str.strip() + " " + out[ln].astype(str).str.strip()).str.strip()
        else:
            # try a few other joins
            if fn and not ln:
                out["Player"] = out[fn].astype(str).str.strip()
            elif ln and not fn:
                out["Player"] = out[ln].astype(str).str.strip()
            else:
                # give up; create synthetic
                out["Player"] = out.index.astype(str)
    else:
        out["Player"] = out[name_col].astype(str).str.strip()

    # Basic fields
    pos_col  = _first_present(out, POS_CANDS)
    team_col = _first_present(out, TEAM_CANDS)
    opp_col  = _first_present(out, OPP_CANDS)
    week_col = _first_present(out, WEEK_CANDS)

    if pos_col and pos_col != "Position":
        out.rename(columns={pos_col: "Position"}, inplace=True)
    else:
        out["Position"] = out.get(pos_col, "UNK")

    if team_col and team_col != "Team":
        out.rename(columns={team_col: "Team"}, inplace=True)
    else:
        out["Team"] = out.get(team_col, "")

    if opp_col and opp_col != "Opponent":
        out.rename(columns={opp_col: "Opponent"}, inplace=True)
    else:
        out["Opponent"] = out.get(opp_col, "")

    if week_col and week_col != "Week":
        out.rename(columns={week_col: "Week"}, inplace=True)
    else:
        out["Week"] = out.get(week_col, 0)

    # Clean types
    with np.errstate(all="ignore"):
        out["Week"] = pd.to_numeric(out["Week"], errors="coerce").fillna(0).astype(int)
    out["Opponent"] = out["Opponent"].apply(_coerce_opp)
    if "Team" in out:
        out["Team"] = out["Team"].astype(str).str.upper().map(lambda s: TEAM_ABBR.get(s, s[:3] if len(s)>=3 else s))

    # Ensure all candidate numeric stat columns are numeric
    used_cols = set()
    for cands in MARKET_COLS.values():
        for c in cands:
            if c in out.columns:
                used_cols.add(c)
    # add likely stat names so casting helps
    likely = [
        "PassingYards","PassYards","RushingYards","ReceivingYards","Receptions","PassingCompletions",
        "PassingAttempts","PassingTouchdowns","RushingTouchdowns","ReceivingTouchdowns",
        "Targets","Yards","Yds","Cmp","Att","Rec"
    ]
    for col in set(list(used_cols) + likely):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


# ---------- market resolution ----------
def _find_first(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns: return n
    lc = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lc: return lc[n.lower()]
    return None

def detect_stat_columns(df: pd.DataFrame, market: str) -> List[str]:
    """Return one or two column names for this market. If two, caller will sum them."""
    if market == "Rushing+Receiving YDs":
        r = _find_first(df, MARKET_COLS["Rushing Yards"])
        y = _find_first(df, MARKET_COLS["Receiving Yards"])
        return [c for c in [r, y] if c]
    if market == "Rushing+Receiving TDs":
        r = _find_first(df, MARKET_COLS["Rushing TDs"])
        y = _find_first(df, MARKET_COLS["Receiving TDs"])
        return [c for c in [r, y] if c]
    cols = MARKET_COLS.get(market, [])
    f = _find_first(df, cols)
    return [f] if f else []

# ---------- distributions (no SciPy) ----------
def norm_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    # stable pmf using log
    return math.exp(k * math.log(max(lam, 1e-12)) - lam - math.lgamma(k + 1))

def poisson_cdf(k: int, lam: float) -> float:
    # sum_{i=0..k} pmf(i; lam)
    if k < 0:
        return 0.0
    s = 0.0
    for i in range(0, k + 1):
        s += poisson_pmf(i, lam)
        if s > 0.999999999:
            return 1.0
    return min(max(s, 0.0), 1.0)

def poisson_sf(x: float, lam: float) -> float:
    """
    Survival function P[X > x] for a Poisson with rate lam.
    If sportsbook line is .5 (e.g., 0.5 TDs), we want P[X >= 1] = 1 - P[X <= 0].
    For general line, use k = floor(x) and compute P[X >= k+1] = 1 - CDF(k).
    """
    k = math.floor(x)
    # we want strictly greater than the line
    return 1.0 - poisson_cdf(k, lam)


# ---------- probability engine ----------
def compute_player_prob(
    df: pd.DataFrame,
    player: str,
    market: str,
    line: float,
    lookback_weeks: int,
    opponent: Optional[str] = None
) -> Tuple[float, float, float, float, int, pd.DataFrame, str]:
    """
    Returns:
        mean, p_over, p_under, confidence(0..1), samples, recent_df, status_msg
    """
    status = ""
    d = df[df["Player"] == player].copy()
    if opponent:
        opp = str(opponent).strip().upper()
        d = d[(d["Opponent"] == opp) | (d["Opponent"].str.contains(opp, na=False))]

    if d.empty:
        return 0.0, 0.0, 0.0, 0.0, 0, pd.DataFrame(), "⚠️ Player not found in local data."

    # get last N weeks
    max_w = int(d["Week"].max()) if "Week" in d else None
    if max_w:
        lo = max_w - lookback_weeks + 1
        d = d[d["Week"].between(lo, max_w, inclusive="both")]

    cols = detect_stat_columns(df, market)
    if not cols:
        return 0.0, 0.0, 0.0, 0.0, 0, d, f"⚠️ No columns detected for market: {market}"

    if len(cols) == 1:
        series = pd.to_numeric(d[cols[0]], errors="coerce")
    else:
        series = sum(pd.to_numeric(d[c], errors="coerce").fillna(0) for c in cols)

    series = series.dropna()
    samples = int(series.shape[0])

    if samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, d, "⚠️ No available stats for this metric."

    mu = float(series.mean())
    sd = float(series.std(ddof=1)) if samples > 1 else max(1.0, abs(mu) * 0.15)

    # Distribution logic
    if "TD" in market or market.endswith("TDs"):
        # small counts -> Poisson
        lam = max(mu, 0.0001)
        p_over = poisson_sf(line, lam)
    else:
        # yards, attempts, receptions -> Normal approx
        # Prob over line, i.e., P[X > line] = 1 - CDF(line)
        p_over = 1.0 - norm_cdf(line, mu, max(sd, 1e-6))

    p_over = float(min(max(p_over, 0.0), 1.0))
    p_under = float(1.0 - p_over)

    # Confidence: blend of samples & dispersion
    #  - start from sample factor
    samp_factor = min(1.0, samples / 8.0)  # 8+ weeks gets full credit
    #  - lower sd relative to mean gives more confidence; clamp
    cv = (sd / (abs(mu) + 1e-6)) if abs(mu) > 1e-6 else 1.0
    disp = 1.0 / (1.0 + cv)  # in (0,1]
    confidence = float(0.65 * samp_factor + 0.35 * disp)

    recent = d.copy()
    recent["__target_value"] = series.values

    return mu, p_over, p_under, confidence, samples, recent, status


# ---------- parlay combiner ----------
def combine_parlay_ps(ps: List[float], legs_info: List[Tuple[str,str]]) -> float:
    """
    Multiply leg probabilities with a soft correlation dampener:
      - if two legs share the same player: apply 0.90 per duplicate after the first
      - if two legs share the same team or opponent: apply 0.97 per pair
      This is a lightweight guard; you can upgrade later.
    """
    if not ps:
        return 0.0
    p = 1.0
    # base product
    for x in ps:
        p *= max(0.0, min(1.0, x))

    # correlation dampener
    n = len(ps)
    players = [li[0] for li in legs_info]
    teams   = [li[1] for li in legs_info]

    # same-player duplicates
    seen = {}
    for pl in players:
        seen[pl] = seen.get(pl, 0) + 1
    for cnt in seen.values():
        if cnt > 1:
            p *= (0.90 ** (cnt - 1))

    # same-team/opponent (very soft)
    team_seen = {}
    for tm in teams:
        if not tm: continue
        team_seen[tm] = team_seen.get(tm, 0) + 1
    for cnt in team_seen.values():
        if cnt > 1:
            p *= (0.97 ** (cnt - 1))

    return float(max(0.0, min(1.0, p)))
