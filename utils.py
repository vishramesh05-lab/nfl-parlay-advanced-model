"""
utils.py  –  Parleggy v70-AI utilities
"""
import os, json, pickle, time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor

# ---------------------- CONFIG ----------------------
DATA_PATH = os.path.join(os.getcwd(), "data")
MODEL_FILE = os.path.join(DATA_PATH, "ai_model.pkl")

SPORTSDATA_KEY = os.getenv("SPORTSDATA_KEY", "")
ODDS_KEY = os.getenv("ODDS_API_KEY", "")
WEATHER_KEY = os.getenv("OPENWEATHER_KEY", "")

# ---------------------- DATA HELPERS ----------------------
def list_jsons():
    return [f for f in os.listdir(DATA_PATH) if f.endswith(".json")]

def load_json(fname):
    with open(os.path.join(DATA_PATH, fname)) as f:
        return json.load(f)

def merge_jsons():
    dfs=[]
    for f in list_jsons():
        try:
            js=load_json(f)
            df=pd.json_normalize(js)
            df["source_file"]=f
            dfs.append(df)
        except Exception:
            pass
    if dfs:
     if dfs:
    try:
        df_all = pd.concat(dfs, ignore_index=True).fillna(0)
    except Exception as e:
        st.error(f"⚠️ Error combining JSON files: {e}")
        df_all = pd.DataFrame()
    return df_all
else:
    st.warning("⚠️ No valid JSON files found in /data — skipping merge.")
    return pd.DataFrame()
    return pd.DataFrame()

# ---------------------- MODEL TRAIN ----------------------
def train_ai(df):
    """Train weighted XGBoost model using 60/25/15 weighting"""
    if df.empty:
        return None
    # select simple features
    cols=[c for c in df.columns if any(k in c.lower() for k in
           ["pass","rush","recv","score","touch","yd","yard"])]
    X=df[cols].select_dtypes(include=[np.number]).values
    y=(df.get("Score") if "Score" in df.columns else
       df[cols].sum(axis=1)*0.001)  # fallback pseudo target
    model=XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X,y)
    pickle.dump(model,open(MODEL_FILE,"wb"))
    return model

# ---------------------- WEIGHTED PREDICTION ----------------------
def weighted_probability(base_prob, player_perf, defense_adj, vegas_adj):
    """
    Combine probabilities using 60/25/15 weighting
    """
    w_player, w_def, w_vegas = 0.60, 0.25, 0.15
    final = (player_perf*w_player +
             defense_adj*w_def +
             vegas_adj*w_vegas)
    # clip
    return float(np.clip(final,0,1))

# ---------------------- SCHEDULING ----------------------
def should_retrain():
    """Return True if it’s time to do a full nightly retrain"""
    now=datetime.utcnow()-timedelta(hours=4)  # convert to EST
    return now.hour==0 and now.minute<5

def should_mini_update():
    """Return True every 30 minutes"""
    return int(time.time())%1800<10

# ---------------------- FETCH HELPERS ----------------------
def fetch_json(url):
    try:
        r=requests.get(url,timeout=10)
        if r.status_code==200:
            return r.json()
    except Exception:
        pass
    return None
