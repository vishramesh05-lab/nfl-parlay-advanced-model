# ==========================================================
# NFL Parlay Helper v68 — Dark Analyst Edition
# Author: Vish (Project Nova Energy)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import beta

# -------------------- PAGE CONFIG --------------------------
st.set_page_config(page_title="NFL Parlay Helper (v68)", page_icon="🏈", layout="wide")

# -------------------- DARK THEME STYLES --------------------
st.markdown("""
    <style>
        body, [data-testid="stAppViewContainer"] {
            background-color: #111111;
            color: #EEEEEE;
        }
        [data-testid="stMetricValue"] {font-size:28px; font-weight:600; color:#0ff;}
        [data-testid="stMetricLabel"] {color:#AAA;}
        h1,h2,h3,h4 {color:#0ff;}
        .rec-bar {padding:15px;border-radius:8px;text-align:center;
                  font-weight:600;font-size:20px;margin-top:10px;}
        .rec-green {background:#006600;color:white;}
        .rec-red {background:#8B0000;color:white;}
        .rec-yellow {background:#DAA520;color:white;}
        table, th, td {background-color:#222 !important;color:#eee !important;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🏈 NFL Parlay Helper — 2025 v68</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#888;'>Defense, Efficiency, Environment, Weather & Vegas Adjusted</p>", unsafe_allow_html=True)

# -------------------- API KEYS ------------------------------
SPORTSDATA_KEY = st.secrets.get("SPORTSDATA_KEY")
ODDS_KEY = st.secrets.get("ODDS_API_KEY")
WEATHER_KEY = st.secrets.get("OPENWEATHER_API_KEY")

# -------------------- HELPERS -------------------------------
def safe_json(url, headers=None):
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

# Auto-detect current NFL week
current_week = datetime.now().isocalendar().week - 35
st.sidebar.header("⚙️ Filters")
week = st.sidebar.slider("Current Week", 1, 18, current_week)
lookback = st.sidebar.slider("Lookback (weeks)", 1, 8, 5)
player_name = st.text_input("Player Name", "Patrick Mahomes")
stat_type = st.selectbox("Stat Type", ["Passing Yards","Rushing Yards","Receiving Yards","Passing Touchdowns"])
sportsbook_line = st.number_input("Sportsbook Line", value=250.0, step=1.0)
opponent_team = st.text_input("Opponent (e.g. KC, BUF, PHI)","")
city = st.text_input("Weather City (optional)","")

# -------------------- API FETCH FUNCTIONS -------------------
@st.cache_data(ttl=900)
def fetch_sportsdata_player_stats(week):
    url=f"https://api.sportsdata.io/v3/nfl/stats/json/PlayerGameStatsByWeek/2025REG/{week}"
    headers={"Ocp-Apim-Subscription-Key":SPORTSDATA_KEY}
    return safe_json(url, headers)

@st.cache_data(ttl=900)
def fetch_sportsdata_team_stats(week):
    url=f"https://api.sportsdata.io/v3/nfl/stats/json/TeamGameStats/2025REG/{week}"
    headers={"Ocp-Apim-Subscription-Key":SPORTSDATA_KEY}
    return safe_json(url, headers)

@st.cache_data(ttl=900)
def fetch_vegas_odds():
    url=f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey={ODDS_KEY}"
    return safe_json(url)

@st.cache_data(ttl=900)
def fetch_weather(city):
    if not city or not WEATHER_KEY: return None
    url=f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=imperial"
    return safe_json(url)

@st.cache_data(ttl=900)
def fetch_sleeper_efficiency(week):
    url=f"https://api.sleeper.app/v1/stats/nfl/regular/2025/{week}"
    return safe_json(url)

# -------------------- ANALYSIS ------------------------------
if st.button("📊 Analyze Player"):
    # collect data across lookback
    frames=[]
    for w in range(max(1,week-lookback+1), week+1):
        data = fetch_sportsdata_player_stats(w)
        if not data: continue
        df = pd.DataFrame(data)
        df["Week"]=w
        frames.append(df)
    if not frames:
        st.error("No data found.")
        st.stop()

    df = pd.concat(frames,ignore_index=True)
    df["full_name"]=(df["FirstName"].fillna("")+" "+df["LastName"].fillna("")).str.strip().str.lower()
    df=df[df["full_name"].str.contains(player_name.lower(),na=False)]

    if df.empty:
        st.error("Player not found in dataset.")
        st.stop()

    # choose stat column
    col_map={"Passing Yards":"PassingYards","Rushing Yards":"RushingYards",
             "Receiving Yards":"ReceivingYards","Passing Touchdowns":"PassingTouchdowns"}
    stat_col=col_map[stat_type]
    vals=df[stat_col].fillna(0).astype(float)

    # base probability
    mean=vals.mean(); sample=len(vals)
    over_hits=(vals>sportsbook_line).sum()
    a,b=over_hits+1,(sample-over_hits)+1
    base_prob_over=beta.mean(a,b)*100; base_prob_under=100-base_prob_over

    # ---------------- Efficiency (Sleeper) -------------------
    eff_json=fetch_sleeper_efficiency(week)
    PEI=1.0
    if eff_json:
        try:
            df_eff=pd.DataFrame(eff_json).T.reset_index()
            df_eff["player"]=df_eff["index"].str.lower()
            if player_name.lower() in df_eff["player"].values:
                row=df_eff[df_eff["player"]==player_name.lower()].iloc[0]
                rush_yds=row.get("rush_yd",0); rec_yds=row.get("rec_yd",0)
                touches=row.get("rush_att",0)+row.get("rec_tgt",0)
                catch_rate=(row.get("rec",0)/max(row.get("rec_tgt",1),1))*100
                ypt=(rush_yds+rec_yds)/max(touches,1)
                league_ypt=np.nanmean((df_eff["rush_yd"].fillna(0)+df_eff["rec_yd"].fillna(0))/
                                      (df_eff["rush_att"].fillna(0)+df_eff["rec_tgt"].fillna(1)))
                PEI=(ypt/league_ypt)*(catch_rate/100)
        except Exception:
            PEI=1.0

    # ---------------- Vegas Environment ----------------------
    EF=100
    vegas=fetch_vegas_odds()
    if vegas:
        totals=[b for g in vegas for bk in g.get("bookmakers",[]) for b in bk.get("markets",[]) if b["key"]=="totals"]
        if totals:
            league_avg=np.mean([x["outcomes"][0]["point"] for x in totals if "outcomes" in x])
            for g in vegas:
                if opponent_team in g.get("home_team","")+g.get("away_team",""):
                    try:
                        ou=[b for bk in g["bookmakers"] for b in bk["markets"] if b["key"]=="totals"][0]
                        val=ou["outcomes"][0]["point"]
                        EF=(val/league_avg)*100
                    except Exception: pass

    # ---------------- Weather Adjustments --------------------
    wx=fetch_weather(city); temp=None; wind=None; humidity=None
    if wx:
        temp=wx["main"]["temp"]; wind=wx["wind"]["speed"]; humidity=wx["main"]["humidity"]
        if temp<40: base_prob_over-=4
        if wind>15: base_prob_over-=6
        if humidity<30: base_prob_over+=2

    # ---------------- Confidence + Accuracy ------------------
    variance=np.var(vals)
    conf=97-(np.sqrt(variance)/(mean+1))*35-(5*(10-min(sample,10)))
    conf=float(np.clip(conf,80,99))
    # Accuracy Index (AIx)
    a=min(sample/6,1)
    b=1-(variance/(mean+1))
    c=1-(abs(EF-100)/200)
    aix=(a*0.4+b*0.35+c*0.25)*100
    aix=float(np.clip(aix,60,99))

    # color codes
    if conf>=90: conf_color="🟢"
    elif conf>=75: conf_color="🟡"
    else: conf_color="🔴"
    if aix>=90: aix_color="🟢"
    elif aix>=75: aix_color="🟡"
    else: aix_color="🔴"

    # final adjustment with PEI and EF
    adj_prob_over=np.clip(base_prob_over*PEI*(EF/100),0,100)
    adj_prob_under=100-adj_prob_over
    rec="✅ Lean Over" if adj_prob_over>55 and conf>85 else "❌ Lean Under" if adj_prob_under>55 and conf>85 else "⚠️ Uncertain"
    rec_class="rec-green" if "Over" in rec else "rec-red" if "Under" in rec else "rec-yellow"

    # -------------------- DISPLAY ----------------------------
    st.markdown(f"<div class='rec-bar {rec_class}'>{rec}</div>", unsafe_allow_html=True)
    col1,col2,col3,col4=st.columns(4)
    col1.metric("Prob (Over)", f"{adj_prob_over:.1f}%")
    col2.metric("Prob (Under)", f"{adj_prob_under:.1f}%")
    col3.metric("Confidence", f"{conf_color} {conf:.1f}%")
    col4.metric("Accuracy Index (AIx)", f"{aix_color} {aix:.1f}%")

    st.markdown("---")
    st.subheader("📅 Weekly Breakdown")
    show=df[["Week","Team","Opponent","Position","PassingYards","RushingYards",
             "ReceivingYards","PassingTouchdowns","RushingTouchdowns","ReceivingTouchdowns"]].fillna(0)
    st.dataframe(show.sort_values("Week"),use_container_width=True)

    # Weather summary
    if wx:
        st.info(f"🌡️ Temp: {temp}°F | 💨 Wind: {wind} mph | 💧Humidity: {humidity}% | Env Factor: {EF:.1f}")
