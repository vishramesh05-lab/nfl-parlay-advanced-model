import os
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# =============================
# PATHS & CONFIG
# =============================
DATA_PATH = os.path.join(os.getcwd(), "data")
MODEL_PATH = os.path.join(os.getcwd(), "models")
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# =============================
# BASIC HELPERS
# =============================

def list_jsons():
    """List all JSON data files in /data directory."""
    return [f for f in os.listdir(DATA_PATH) if f.endswith(".json")]

def load_json(fname):
    """Load individual JSON file."""
    with open(os.path.join(DATA_PATH, fname)) as f:
        return json.load(f)

def merge_jsons():
    """Merge all JSON files into one DataFrame."""
    dfs = []
    for f in list_jsons():
        try:
            js = load_json(f)
            df = pd.json_normalize(js)
            df["source_file"] = f
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True).fillna(0)
    return pd.DataFrame()

# =============================
# COLUMN DETECTION HELPERS
# =============================

def find_column(df, *keywords):
    """Find a column that contains all given keyword fragments."""
    for c in df.columns:
        name = c.lower()
        if all(k.lower() in name for k in keywords):
            return c
    return None

def find_stat_column(df, market):
    """Find the correct stat column(s) dynamically for any market."""
    market = market.lower()
    
    # Passing Yards / TDs / INTs
    if "passing yards" in market:
        return find_column(df, "pass", "yd")
    if "passing tds" in market or "passing touchdowns" in market:
        return find_column(df, "pass", "td")
    if "interceptions" in market:
        return find_column(df, "int")
    
    # Rushing
    if "rushing yards" in market:
        return find_column(df, "rush", "yd") or find_column(df, "rushing", "yards")
    if "rushing tds" in market:
        return find_column(df, "rush", "td")
    
    # Receiving
    if "receiving yards" in market:
        return find_column(df, "rec", "yd") or find_column(df, "receiv", "yards")
    if "receptions" in market:
        return find_column(df, "rec")
    if "receiving tds" in market:
        return find_column(df, "rec", "td") or find_column(df, "receiv", "td")
    
    # Combo stats
    if "rushing+receiving tds" in market or "combo tds" in market:
        rush_col = find_column(df, "rush", "td")
        rec_col = find_column(df, "rec", "td") or find_column(df, "receiv", "td")
        if rush_col or rec_col:
            return (rush_col, rec_col)
    
    # Fantasy / Target
    if "fantasy points" in market:
        return find_column(df, "fantasy")
    if "targets" in market:
        return find_column(df, "target")
    
    return None

def extract_market_series(df, market):
    """Extract numeric data for the selected market from df."""
    col = find_stat_column(df, market)
    if isinstance(col, tuple):  # combo stat
        rush, rec = col
        rush_vals = df[rush].fillna(0) if rush in df.columns else 0
        rec_vals = df[rec].fillna(0) if rec in df.columns else 0
        return (rush_vals + rec_vals).astype(float)
    elif col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    else:
        return pd.Series(np.zeros(len(df)))

# =============================
# AI MODELING FUNCTIONS
# =============================

def train_ai_model(df, target_series, features=None):
    """Train XGBoost model on given features and target."""
    if features is None:
        features = ["FanDuelSalary", "FantasyDataSalary", "DraftKingsSalary"]
        features = [f for f in features if f in df.columns]

    if not features or target_series is None or len(target_series) == 0:
        return None, 0

    try:
        X = df[features].fillna(0)
        y = target_series.fillna(0)
        if y.nunique() <= 1:
            return None, 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.9, colsample_bytree=0.8)
        model.fit(X_train, y_train)
        mae = np.mean(np.abs(model.predict(X_test) - y_test))
        return model, mae
    except Exception as e:
        print(f"Model training failed: {e}")
        return None, 0

def predict_probability(model, sportsbook_line, samples=500):
    """Monte Carlo estimate of Over/Under probability."""
    if model is None:
        return 0.5, 0.5, 0.0
    preds = model.predict(model.get_booster().dtrain.get_label()) if hasattr(model, "get_booster") else []
    mean_pred = np.mean(preds) if len(preds) else 0
    # Simulate with variance
    dist = np.random.normal(mean_pred, np.std(preds) if len(preds) > 1 else 1, samples)
    over_prob = np.mean(dist > sportsbook_line)
    under_prob = 1 - over_prob
    confidence = abs(over_prob - 0.5) * 200
    return round(over_prob*100,1), round(under_prob*100,1), round(confidence,1)

# =============================
# CACHING / REFRESH HELPERS
# =============================

def save_json_cache(df):
    path = os.path.join(DATA_PATH, "merged_cache.json")
    df.to_json(path, orient="records")

def load_cached_df():
    path = os.path.join(DATA_PATH, "merged_cache.json")
    if os.path.exists(path):
        return pd.read_json(path)
    return merge_jsons()

def should_refresh(last_refresh_time, interval_minutes=30):
    """Check if 30 minutes have passed since last refresh."""
    return (datetime.utcnow() - last_refresh_time).total_seconds() > (interval_minutes * 60)
