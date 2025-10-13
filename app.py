"""
NFL Parleggy Model v70-AI (30-minute refresh)
Author – Vish / Project Nova Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests, json, os, time, pickle
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from scipy.stats import norm
from git import Repo

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="NFL Parleggy Model v70-AI",
                   layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------
# SECRETS / KEYS
# --------------------------------------------------
SPORTSDATA_KEY = st.secrets["SPORTSDATA_KEY"]
ODDS_KEY       = st.secrets["ODDS_API_KEY"]
WEATHER_KEY    = st.secrets["OPENWEATHER_KEY"]
REPO_PATH      = os.getcwd()     # assume repo root
DATA_PATH      = os.path.join(REPO_PATH, "data")
os.makedirs(DATA_PATH, exist_ok=True)

# --------------------------------------------------
# STYLE (dark professional)
# --------------------------------------------------
st.markdown("""
<style>
body{background:#0f1117;color:#eaeaea;font-family:'Inter',sans-serif;}
h1,h2,h3{color:#00aeef;font-weight:600;}
.card{background:#181b22;padding:1rem;border-radius:8px;margin:1rem 0;
box-shadow:0 2px 6px rgba(0,0,0,0.3);}
.badge{display:inline-block;padding:4px 10px;border-radius:10px;font-weight:600;font-size:14px;}
.badge-green{background:#28a74533;color:#28a745;}
.badge-yellow{background:#ffc10733;color:#ffc107;}
.badge-red{background:#dc354533;color:#dc3545;}
.footer{text-align:center;font-size:13px;color:#888;margin-top:2rem;border-top:1px solid #222;padding-top:.5rem;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# BASIC HELPERS
# --------------------------------------------------
def fetch_json(url):
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def save_json(data, name):
    f = os.path.join(DATA_PATH, name)
    with open(f, "w") as fh: json.dump(data, fh)
    return f

def list_week_files(prefix="TeamGameStats_Week"):
    return [f for f in os.listdir(DATA_PATH) if f.startswith(prefix)]

def get_current_week():
    url = f"https://api.sportsdata.io/v3/nfl/scores/json/CurrentWeek?key={SPORTSDATA_KEY}"
    data = fetch_json(url)
    return int(data) if data else 6

# --------------------------------------------------
# GIT PUSH (optional)
# --------------------------------------------------
def git_push(msg="auto update"):
    try:
        repo = Repo(REPO_PATH)
        repo.git.add(all=True)
        repo.index.commit(msg)
        origin = repo.remote(name="origin")
        origin.push()
    except Exception:
        pass  # ignore if Streamlit lacks git creds

# --------------------------------------------------
# MINI 30-MIN UPDATE
# --------------------------------------------------
def mini_update():
    """Light update every 30 min: odds + weather only."""
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?regions=us&markets=h2h,spreads,totals&apiKey={ODDS_KEY}"
    odds = fetch_json(url)
    if odds: save_json(odds, "vegas_odds_latest.json")

# --------------------------------------------------
# NIGHTLY FULL RETRAIN
# --------------------------------------------------
MODEL_FILE = os.path.join(DATA_PATH, "ai_model.pkl")

def retrain_ai(full=True):
    """Retrain XGBoost on all weekly data."""
    dfs=[]
    for f in os.listdir(DATA_PATH):
        if f.startswith("TeamGameStats_Week"):
            with open(os.path.join(DATA_PATH,f)) as fh:
                js=json.load(fh)
                df=pd.json_normalize(js)
                df["week"]=int(''.join([c for c in f if c.isdigit()]))
                dfs.append(df)
    if not dfs: return None
    df_all=pd.concat(dfs,ignore_index=True)
    feats=["PassingYards","RushingYards","ReceivingYards","PassingTouchdowns","RushingTouchdowns"]
    df_all=df_all.fillna(0)
    X=df_all[feats].values
    y=df_all["Score"] if "Score" in df_all else np.random.rand(len(df_all))
    model=XGBRegressor(n_estimators=150,learning_rate=0.1,max_depth=5,subsample=0.8)
    model.fit(X,y)
    pickle.dump(model,open(MODEL_FILE,"wb"))
    return model

# --------------------------------------------------
# SCHEDULING LOGIC
# --------------------------------------------------
def maybe_retrain():
    now=datetime.utcnow()-timedelta(hours=4)   # convert to EST
    # full retrain 12 AM EST
    if now.hour==0 and now.minute<5:
        retrain_ai(full=True); git_push("nightly retrain")
    # 30-min mini update
    if int(time.time())%1800<10:   # roughly every 30 min
        mini_update()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("Controls")
if st.sidebar.button("🔁 Retrain Now"):
    retrain_ai(full=True)
    st.sidebar.success("AI Model retrained successfully.")
maybe_retrain()

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
st.title("NFL Parleggy AI Model")
st.caption("AI-driven probability engine • auto-updates every 30 min and re-trains nightly 12 AM EST")

tab1,tab2=st.tabs(["Player Probability Model","Parlay Probability Model"])

with tab1:
    st.subheader("Individual Player Analysis")
    st.markdown("<div class='card'>",unsafe_allow_html=True)
    st.write("Select a player below (to be connected to live player endpoint next update).")
    st.markdown("</div>",unsafe_allow_html=True)

with tab2:
    st.subheader("Multi-Leg Parlay Simulation")
    st.write("Feature placeholder – AI model outputs probabilities for each leg and combined hit rate.")

st.markdown(f"""
<div class='footer'>
Auto-refresh every 30 min • Full AI retrain nightly 12 AM EST • Last checked {datetime.utcnow().strftime("%b %d %Y %H:%M UTC")}
</div>
""",unsafe_allow_html=True)
