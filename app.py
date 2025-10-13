# ======================================================
# NFL Parlay Helper v64 — Smart Player Search + Confidence Model
# Author: Vishvin Ramesh
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="NFL Parlay Helper (v64)", page_icon="🏈", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>🏈 NFL Parlay Helper — 2025 Smart Dual Model</h1>",
    unsafe_allow_html=True,
)
st.caption("Live Player + Defense + Injury adjusted probabilities — SportsData.io + OpenWeather")
st.caption("Build v64 • by Vishvin Ramesh")

# ------------------------------------------------------
# API KEYS
# ------------------------------------------------------
SPORTSDATA_KEY = st.secrets.get("SPORTSDATA_KEY")
WEATHER_KEY = st.secrets.get("OPENWEATHER_KEY")

if not SPORTSDATA_KEY:
    st.error("⚠️ Add SPORTSDATA_KEY in Streamlit → Secrets.")
    st.stop()

# ------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------
st.sidebar.header("⚙️ Filters")
week = st.sidebar.slider("Current Week", 1, 18, 6)
lookback = st.sidebar.slider("Lookback Weeks", 1, 10, 5)
debug = st.sidebar.checkbox("Enable Debug Mode")

# ------------------------------------------------------
# FETCH HELPERS
# ------------------------------------------------------
def safe_fetch_json(url):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

@st.cache_data(ttl=900)
def fetch_players():
    """Get all current-week players"""
    url = f"https://api.sportsdata.io/v3/nfl/stats/json/Players?key={SPORTSDATA_KEY}"
    data = safe_fetch_json(url)
    if not data: return []
    df = pd.DataFrame(data)
    df["full_name"] = (df["FirstName"].fillna("")+" "+df["LastName"].fillna("")).str.strip()
    df = df[df["Active"]==True]
    return sorted(df["full_name"].unique().tolist())

@st.cache_data(ttl=900)
def fetch_week_stats(week_num):
    """Fetch weekly player stats"""
    url = f"https://api.sportsdata.io/v3/nfl/stats/json/PlayerGameStatsByWeek/2025REG/{week_num}?key={SPORTSDATA_KEY}"
    data = safe_fetch_json(url)
    if not data: raise Exception(f"No data for week {week_num}")
    df = pd.DataFrame(data)
    for c in ["FirstName","LastName","Name","Team","PassingYards","RushingYards",
              "ReceivingYards","PassingTouchdowns","RushingTouchdowns","ReceivingTouchdowns"]:
        if c not in df.columns: df[c]=np.nan
    df["full_name"]=(df["FirstName"].fillna("")+" "+df["LastName"].fillna("")).str.strip().str.lower()
    df.loc[df["full_name"].eq(""),"full_name"]=df["Name"].astype(str).str.lower()
    return df

def fetch_defense_stats(team):
    url=f"https://api.sportsdata.io/v3/nfl/scores/json/TeamSeasonStats/2025REG?key={SPORTSDATA_KEY}"
    data=safe_fetch_json(url)
    if not data: return None
    df=pd.DataFrame(data)
    df=df[df["Team"].eq(team.upper())]
    if df.empty: return None
    return {
        "PassYdsAllowed":df.iloc[0].get("OpponentPassingYards",0),
        "RushYdsAllowed":df.iloc[0].get("OpponentRushingYards",0),
        "PointsAllowed":df.iloc[0].get("PointsAllowed",0)
    }

def fetch_injuries(team):
    url=f"https://api.sportsdata.io/v3/nfl/scores/json/Injuries/2025REG?key={SPORTSDATA_KEY}"
    data=safe_fetch_json(url)
    if not data: return 0
    df=pd.DataFrame(data)
    return len(df[df["Team"].eq(team.upper())])

def fetch_weather(city):
    if not city or not WEATHER_KEY: return None
    url=f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=imperial"
    js=safe_fetch_json(url)
    if js: return js["main"]["temp"]
    return None

# ------------------------------------------------------
# SMART SEARCH
# ------------------------------------------------------
players = fetch_players()
player_name = st.selectbox("Search Player", options=players if players else ["Loading players…"])
stat_type = st.selectbox("Stat Type", ["Passing Yards","Rushing Yards","Receiving Yards",
                                       "Passing Touchdowns","Rushing + Receiving Touchdowns"])
sportsbook_line = st.number_input("Sportsbook Line", value=250.0, step=1.0)
opponent_team = st.text_input("Opponent Team (e.g., KC, BUF, PHI)", "")
weather_city = st.text_input("Weather City (optional)", "")

st.divider()

# ------------------------------------------------------
# MAIN ANALYSIS
# ------------------------------------------------------
if st.button("📊 Analyze Player"):
    if player_name == "Loading players…":
        st.error("Players not loaded yet.")
        st.stop()

    pname = player_name.lower().strip()
    st.info(f"Fetching last {lookback} weeks for {player_name}…")

    all_data=[]
    for wk in range(max(1, week - lookback + 1), week + 1):
        try:
            df=fetch_week_stats(wk)
            df["Week"]=wk
            all_data.append(df)
        except Exception as e:
            st.warning(f"Skipping Week {wk}: {e}")

    if not all_data:
        st.error("No data found.")
        st.stop()

    df=pd.concat(all_data,ignore_index=True)
    df=df[df["full_name"].str.contains(pname,na=False)]
    if df.empty:
        st.error(f"No player data for {player_name}.")
        st.stop()

    # Map stat column
    if stat_type=="Passing Yards": col="PassingYards"
    elif stat_type=="Rushing Yards": col="RushingYards"
    elif stat_type=="Receiving Yards": col="ReceivingYards"
    elif stat_type=="Passing Touchdowns": col="PassingTouchdowns"
    else:
        df["RushRecvTDs"]=df["RushingTouchdowns"].fillna(0)+df["ReceivingTouchdowns"].fillna(0)
        col="RushRecvTDs"

    vals=df[col].fillna(0).astype(float)
    mean=vals.mean()
    variance=np.var(vals)
    sample=len(vals)
    prob_over=np.mean(vals>sportsbook_line)*100
    prob_under=100-prob_over

    # Defense adjust
    defense=fetch_defense_stats(opponent_team)
    def_adj=0
    if defense:
        if "Yards" in stat_type:
            if "Pass" in stat_type: def_adj=(defense["PassYdsAllowed"]/250-1)*10
            elif "Rush" in stat_type: def_adj=(defense["RushYdsAllowed"]/120-1)*10
        prob_over+=def_adj

    # Injuries adjust
    injuries=fetch_injuries(opponent_team)
    if injuries>3: prob_over+=5
    elif injuries==0: prob_over-=2

    # Weather adjust
    temp=fetch_weather(weather_city)
    if temp:
        if temp<40: prob_over-=5
        elif temp>85: prob_over-=3

    prob_over=float(np.clip(prob_over,0,100))
    prob_under=100-prob_over

    # Confidence Score (0–100)
    conf=100-((variance/(mean+1))*8)
    conf*=(min(sample,10)/10)
    conf=float(np.clip(conf,0,100))

    # Per-Week Table
    df_week=df[["Week","Team","Opponent","PassingYards","RushingYards",
                "ReceivingYards","PassingTouchdowns","RushingTouchdowns",
                "ReceivingTouchdowns"]].fillna(0)
    df_week["Rushing+Receiving TDs"]=df_week["RushingTouchdowns"]+df_week["ReceivingTouchdowns"]

    # Summary Table
    summary=pd.DataFrame({
        "Metric":["Average Value","Probability Over","Probability Under",
                  "Defense Adj (%)","Injuries","Weather (°F)","Confidence Score"],
        "Value":[f"{mean:.1f}",f"{prob_over:.1f} %",f"{prob_under:.1f} %",
                 f"{def_adj:+.1f}" if defense else "N/A",injuries,
                 f"{temp:.1f}" if temp else "N/A",f"{conf:.1f} %"]
    })

    st.success(f"✅ {player_name} ({stat_type}) — Adjusted Results & Confidence Model")
    st.dataframe(summary,use_container_width=True)
    st.markdown("#### 📅 Per-Week Breakdown")
    st.dataframe(df_week.sort_values("Week"),use_container_width=True)

    if debug:
        st.subheader("Raw API Data")
        st.dataframe(df.head(25))
