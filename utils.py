# -*- coding: utf-8 -*-
"""
utils.py — Data pipeline + local AI for NFL Parleggy
- Robust JSON ingest/merge from /data
- Market-aware feature/target selection
- XGBoost training per market with residual sigma for probability
- Nightly retrain & training log
- Player-level prediction and parlay combiner with correlation penalty
"""

import os, json, time, pickle, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from scipy.stats import norm

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- PATHS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

LAST_RETRAIN_FILE = os.path.join(DATA_DIR, "last_retrain.txt")
TRAIN_LOG = os.path.join(DATA_DIR, "training_log.csv")

# Markets we support
MARKETS = [
    "Passing Yards",
    "Rushing Yards",
    "Receiving Yards",
    "Rushing+Receiving TDs",
    "Passing TDs",
]

# -------------------- JSON LOADING --------------------
def load_merged_json() -> pd.DataFrame:
    """Read every .json in /data, flatten robustly, concatenate into one DataFrame."""
    frames = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith(".json"):
            continue
        fp = os.path.join(DATA_DIR, fn)
        try:
            with open(fp, "r") as f:
                js = json.load(f)
            if isinstance(js, list):
                df = pd.json_normalize(js)
            elif isinstance(js, dict):
                df = pd.DataFrame([js])
            else:
                continue
            df["source_file"] = fn
            frames.append(df)
        except Exception:
            # skip unreadable file but do not crash the app
            continue
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True).fillna(0)
    # Uniformize column names (lower for detection; keep original too)
    df_all.columns = [c for c in df_all.columns]
    return df_all

# -------------------- COLUMN DETECTION --------------------
NAME_CANDIDATES = ["Name", "player", "PlayerName", "Player", "full_name", "player_name", "display_name"]
POS_CANDIDATES  = ["position", "Position", "pos"]
WEEK_CANDIDATES = ["week", "Week", "game_week", "nfl_week"]
TEAM_CANDIDATES = ["team", "Team", "player_team", "team_abbr"]
OPP_CANDIDATES  = ["opponent", "Opponent", "opp", "opp_abbr", "opp_team"]

def _detect_col(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
        # try case-insensitive
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None

def detect_name_column(df): return _detect_col(df, NAME_CANDIDATES)
def detect_position_column(df): return _detect_col(df, POS_CANDIDATES)
def detect_week_column(df): return _detect_col(df, WEEK_CANDIDATES)
def detect_team_column(df): return _detect_col(df, TEAM_CANDIDATES)
def detect_opp_column(df): return _detect_col(df, OPP_CANDIDATES)

# -------------------- MARKET TARGET DETECTION --------------------
def find_target_column(df: pd.DataFrame, market: str) -> tuple[pd.Series | None, str | None]:
    """Heuristically find or build the target series for a given market."""
    cols = {c.lower(): c for c in df.columns}
    lc = list(cols.keys())

    def find_by_keywords(kw_list: list[str]) -> str | None:
        for c in lc:
            if all(kw in c for kw in kw_list):
                return cols[c]
        return None

    if market == "Passing Yards":
        c = find_by_keywords(["pass", "yd"])
        return (df[c] if c in df.columns else None, c)
    if market == "Rushing Yards":
        c = find_by_keywords(["rush", "yd"])
        return (df[c] if c in df.columns else None, c)
    if market == "Receiving Yards":
        # handle "rec_yd", "receiving_yards" etc.
        c = find_by_keywords(["rec", "yd"]) or find_by_keywords(["receiv", "yd"])
        return (df[c] if c in df.columns else None, c)
    if market == "Passing TDs":
        c = find_by_keywords(["pass", "td"])
        return (df[c] if c in df.columns else None, c)
    if market == "Rushing+Receiving TDs":
        cr = find_by_keywords(["rush", "td"])
        cre = find_by_keywords(["rec", "td"]) or find_by_keywords(["receiv", "td"])
        if cr or cre:
            val = (df[cr] if cr in df.columns else 0) + (df[cre] if cre in df.columns else 0)
            return (val, f"{cr or '0'}+{cre or '0'}")
        # fallback: generic "td"
        cg = find_by_keywords(["td"])
        return (df[cg] if cg in df.columns else None, cg)
    return None, None

# -------------------- FEATURE PREP --------------------
ID_COLS_BLOCK = set(NAME_CANDIDATES + POS_CANDIDATES + WEEK_CANDIDATES + TEAM_CANDIDATES + OPP_CANDIDATES + ["source_file"])

def build_training_matrix(df: pd.DataFrame, market: str):
    """Return X, y, feature_cols, target_name for model training."""
    y, target_name = find_target_column(df, market)
    if y is None:
        return None, None, [], None

    # numeric features excluding direct target + identifier columns
    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target_name]
    feature_cols = [c for c in numeric_cols if c not in ID_COLS_BLOCK]
    X = df[feature_cols].fillna(0)
    y = pd.to_numeric(y, errors="coerce").fillna(0)
    if len(X) < 20:
        return None, None, [], target_name
    return X, y, feature_cols, target_name

def build_player_feature_vector(df_all, player_name, market, name_col, week_col, lookback_weeks=4):
    """Average the player's last N rows to create a prediction row (features only)."""
    df_p = df_all[df_all[name_col].astype(str).str.lower() == str(player_name).lower()].copy()
    if df_p.empty:
        return None, 0
    if week_col and week_col in df_p.columns:
        df_p = df_p.sort_values(by=week_col)
        df_p = df_p.tail(lookback_weeks)
    # use same feature set as training
    temp_X, _, feature_cols, _ = build_training_matrix(df_all, market)
    if not feature_cols:
        return None, 0
    vec = df_p[feature_cols].mean(numeric_only=True)
    # fill missing features with global mean
    global_means = df_all[feature_cols].mean(numeric_only=True)
    vec = global_means.combine_first(vec).fillna(global_means)
    return vec.to_frame().T, len(df_p)

# -------------------- MODEL I/O --------------------
def model_path(market: str) -> str:
    key = market.replace("+", "plus").replace(" ", "_").lower()
    return os.path.join(MODEL_DIR, f"model_{key}.pkl")

def save_model(market, model, sigma, mae, target_name, feature_cols):
    obj = dict(
        market=market, model=model, sigma=float(sigma), mae=float(mae),
        target_name=target_name, feature_cols=feature_cols, trained_at=datetime.utcnow().isoformat()
    )
    with open(model_path(market), "wb") as f:
        pickle.dump(obj, f)

def load_model(market):
    p = model_path(market)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

# -------------------- TRAIN / RETRAIN --------------------
def train_market(df_all: pd.DataFrame, market: str) -> tuple[bool, float]:
    """Train one market model. Returns (ok, mae)."""
    X, y, feature_cols, target_name = build_training_matrix(df_all, market)
    if X is None or y is None or not len(feature_cols):
        return False, np.inf

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(
        n_estimators=350, learning_rate=0.06, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=2
    )
    model.fit(X_tr, y_tr)
    pred_te = model.predict(X_te)
    mae = float(mean_absolute_error(y_te, pred_te))

    # residual sigma from validation
    resid = y_te - pred_te
    sigma = float(np.std(resid)) if np.std(resid) > 1e-6 else float(np.std(y_te))

    save_model(market, model, sigma, mae, target_name, feature_cols)
    log_training(market, mae, len(X))
    return True, mae

def retrain_all_models() -> str:
    df_all = load_merged_json()
    if df_all.empty:
        return "No data found; retrain skipped."
    msgs = []
    for m in MARKETS:
        ok, mae = train_market(df_all, m)
        msgs.append(f"{m}: {'OK' if ok else 'SKIP'} (MAE {mae:.2f})")
    with open(LAST_RETRAIN_FILE, "w") as f:
        f.write(datetime.utcnow().isoformat())
    return " | ".join(msgs)

def maybe_retrain_all():
    """Auto retrain nightly at 12 AM EST (04:00 UTC)."""
    now = datetime.utcnow()
    last = None
    if os.path.exists(LAST_RETRAIN_FILE):
        with open(LAST_RETRAIN_FILE, "r") as f:
            try:
                last = datetime.fromisoformat(f.read().strip())
            except Exception:
                last = None
    due_nightly = (now.hour == 4)  # 04:00 UTC ~ 12:00 AM EST (no DST handling here)
    due_interval = (last is None) or ((now - last) > timedelta(hours=24))
    if due_nightly or due_interval:
        retrain_all_models()

def get_last_retrain_time():
    if os.path.exists(LAST_RETRAIN_FILE):
        with open(LAST_RETRAIN_FILE, "r") as f:
            try:
                t = datetime.fromisoformat(f.read().strip())
                return t.strftime("%b %d %Y %H:%M UTC")
            except Exception:
                return "Unknown"
    return "Never"

def log_training(market, mae, nrows):
    row = pd.DataFrame([{"timestamp": datetime.utcnow().isoformat(), "market": market, "mae": mae, "records": nrows}])
    if os.path.exists(TRAIN_LOG):
        try:
            old = pd.read_csv(TRAIN_LOG)
            pd.concat([old, row], ignore_index=True).to_csv(TRAIN_LOG, index=False)
        except Exception:
            row.to_csv(TRAIN_LOG, index=False)
    else:
        row.to_csv(TRAIN_LOG, index=False)

def get_training_log():
    if os.path.exists(TRAIN_LOG):
        try:
            return pd.read_csv(TRAIN_LOG)
        except Exception:
            return pd.DataFrame(columns=["timestamp", "market", "mae", "records"])
    return pd.DataFrame(columns=["timestamp", "market", "mae", "records"])

# -------------------- PREDICTION --------------------
def predict_player_market(
    df_all: pd.DataFrame,
    player_name: str,
    market: str,
    line: float,
    opponent: str,
    name_col: str,
    week_col: str | None,
    team_col: str | None,
    opp_col: str | None,
    lookback_weeks: int = 4
) -> dict:
    """Return dict with pred_mean, p_over, p_under, confidence, mae, samples, dist_samples (df)."""
    # Ensure model exists
    pack = load_model(market)
    if pack is None:
        train_market(df_all, market)
        pack = load_model(market)
        if pack is None:
            # cannot proceed
            return dict(pred_mean=0.0, p_over=0.5, p_under=0.5, confidence=0.0, mae=np.inf, samples=0,
                        dist_samples=pd.DataFrame({"value": []}))

    model: XGBRegressor = pack["model"]
    sigma: float = float(pack["sigma"]) if pack.get("sigma", 0.0) else 20.0
    mae: float = float(pack.get("mae", 15.0))
    feature_cols = pack["feature_cols"]

    # Build feature vector from last N games
    x_vec, samples = build_player_feature_vector(df_all, player_name, market, name_col, week_col, lookback_weeks)
    if x_vec is None or x_vec.empty:
        # fallback to global means
        x_vec = df_all[feature_cols].mean(numeric_only=True).to_frame().T

    # Opponent adjustment (if we have opponent column)
    if opponent and opp_col and opp_col in df_all.columns and pack.get("target_name"):
        y_series, _ = find_target_column(df_all, market)
        if y_series is not None:
            # compute historical average allowed vs that opponent
            mask_opp = df_all[opp_col].astype(str).str.upper() == opponent.upper()
            if mask_opp.any():
                opp_avg = pd.to_numeric(y_series[mask_opp], errors="coerce").fillna(0).mean()
                global_avg = pd.to_numeric(y_series, errors="coerce").fillna(0).mean()
                # scale predicted mean by relative difficulty
                adj = 0.15 * (opp_avg - global_avg)  # modest adjustment
                # store for later use
                x_vec = x_vec.copy()
                # we do not mutate features; we'll add to prediction directly

    # Prediction
    pred_mean = float(model.predict(x_vec[feature_cols])[0])

    # Use residual sigma to compute parametric probability
    # Optionally include opponent difficulty shift if computed
    try:
        if opponent and opp_col and opp_col in df_all.columns:
            y_series, _ = find_target_column(df_all, market)
            if y_series is not None:
                mask_opp = df_all[opp_col].astype(str).str.upper() == opponent.upper()
                if mask_opp.any():
                    opp_avg = pd.to_numeric(y_series[mask_opp], errors="coerce").fillna(0).mean()
                    global_avg = pd.to_numeric(y_series, errors="coerce").fillna(0).mean()
                    pred_mean += 0.15 * (opp_avg - global_avg)
    except Exception:
        pass

    # Probability using Normal CDF
    z = (line - pred_mean) / max(sigma, 1e-6)
    p_over = float(1.0 - norm.cdf(z))
    p_under = 1.0 - p_over

    # Confidence 0–100: tighter sigma & larger sample => higher
    # scale by inverse of sigma and by #samples
    conf = float(np.clip(70.0 * (1.0 / (1.0 + sigma/25.0)) + 30.0 * (min(samples, 6) / 6.0), 0, 100))

    # Distribution samples for quick histogram
    sims = np.random.normal(loc=pred_mean, scale=max(sigma, 1e-6), size=4000)
    dist_df = pd.DataFrame({"value": sims})

    return dict(
        pred_mean=pred_mean, p_over=p_over, p_under=p_under,
        confidence=conf, mae=mae, samples=samples,
        dist_samples=dist_df
    )

def confidence_color_label(conf: float) -> tuple[str, str]:
    if conf >= 80:  return "#21ba45", "High Confidence"
    if conf >= 60:  return "#fbbd08", "Moderate Confidence"
    return "#db2828", "Low Confidence"

# -------------------- HISTORY TABLE --------------------
def get_player_history(df_all, player_name, name_col, week_col, lookback_weeks=4):
    dfp = df_all[df_all[name_col].astype(str).str.lower() == str(player_name).lower()].copy()
    if dfp.empty:
        return pd.DataFrame()
    if week_col and week_col in dfp.columns:
        dfp = dfp.sort_values(by=week_col).tail(lookback_weeks)
    # select a compact set of useful numeric columns
    num_cols = dfp.select_dtypes(include=["number"]).columns.tolist()
    # keep top 6 numeric columns by variance for a compact table
    if len(num_cols) > 8:
        vari = dfp[num_cols].var().sort_values(ascending=False)
        num_cols = vari.head(6).index.tolist()
    view_cols = []
    if week_col and week_col in dfp.columns:
        view_cols.append(week_col)
    view_cols.append(name_col)
    view_cols += num_cols
    view_cols = [c for c in view_cols if c in dfp.columns]
    return dfp[view_cols].reset_index(drop=True)

# -------------------- PARLAY COMBINER --------------------
def combine_parlay_probabilities(leg_results: list[dict]) -> tuple[float, float]:
    """
    Multiply leg probabilities with a soft correlation penalty:
    - same player: -10% adjustment
    - same team/opponent across legs: -5% adjustment
    - same game (team vs same opp): -3% adjustment
    Returns (final_p, total_penalty_applied)
    """
    if not leg_results:
        return 0.0, 0.0
    base = 1.0
    for r in leg_results:
        base *= float(r["p_over"])  # assume over legs; if under legs added later, extend schema

    # Soft penalty (since we don't have structured schedule matrix here)
    penalty = 0.0
    # heuristic: reduce 3–10% depending on repeated entities detected (caller may pass metadata)
    # We can't infer teams reliably from results; caller can extend. For now, fixed 4% per duplicate pair up to 12%.
    if len(leg_results) >= 2:
        penalty = min(0.12, 0.04 * (len(leg_results) - 1))
    final_p = max(0.0, base * (1.0 - penalty))
    return final_p, penalty
