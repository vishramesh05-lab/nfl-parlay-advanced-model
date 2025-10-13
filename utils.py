import os, json, time, random, pandas as pd
from datetime import datetime, timedelta

# Ensure data folder exists regardless of Streamlit environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_PATH, exist_ok=True)

MODEL_FILE = os.path.join(DATA_PATH, "ai_model.pkl")
LAST_RETRAIN_FILE = os.path.join(DATA_PATH, "last_retrain.txt")

# -----------------------------
# BASIC HELPERS
# -----------------------------
def list_jsons():
    return [f for f in os.listdir(DATA_PATH) if f.endswith(".json")]

def load_json(fname):
    with open(os.path.join(DATA_PATH, fname)) as f:
        return json.load(f)

def merge_jsons():
    dfs = []
    for f in list_jsons():
        try:
            df = pd.json_normalize(load_json(f))
            df["source_file"] = f
            dfs.append(df)
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True).fillna(0) if dfs else pd.DataFrame()

# -----------------------------
# RETRAIN LOGIC
# -----------------------------
def retrain_ai():
    """Simulated AI retraining placeholder."""
    time.sleep(2)
    with open(LAST_RETRAIN_FILE, "w") as f:
        f.write(datetime.utcnow().isoformat())
    return True

def get_last_retrain_time():
    if os.path.exists(LAST_RETRAIN_FILE):
        with open(LAST_RETRAIN_FILE) as f:
            return datetime.fromisoformat(f.read().strip()).strftime("%b %d %Y %H:%M UTC")
    return "Never"

def maybe_retrain():
    """Retrains every 30 minutes or nightly at 12 AM EST (4 UTC)."""
    now = datetime.utcnow()
    if not os.path.exists(LAST_RETRAIN_FILE):
        retrain_ai(); return True
    with open(LAST_RETRAIN_FILE) as f:
        last = datetime.fromisoformat(f.read().strip())
    if (now - last) > timedelta(minutes=30) or now.hour == 4:
        retrain_ai(); return True
    return False

# -----------------------------
# PLAYER & PROBABILITY FUNCTIONS
# -----------------------------
def get_player_dropdown():
    return [
        "Caleb Williams — CHI (QB)",
        "Bijan Robinson — ATL (RB)",
        "Justin Jefferson — MIN (WR)",
        "Patrick Mahomes — KC (QB)",
        "Lamar Jackson — BAL (QB)"
    ]

def fetch_player_data(player, stat_type):
    try:
        df = merge_jsons()
        if df.empty: return None
        df = df[df.apply(lambda r: player.split("—")[0].strip().lower() in str(r).lower(), axis=1)]
        return df
    except Exception:
        return None

def simulate_fallback(line):
    avg = line + random.uniform(-20, 20)
    diff = avg - line
    over = 50 + diff * 1.2
    under = 100 - over
    return avg, max(0, min(100, over)), max(0, min(100, under))

def calculate_probabilities(df, line):
    vals = df.select_dtypes(include=["number"]).mean().mean()
    diff = vals - line
    over = 50 + diff * 1.1
    under = 100 - over
    return vals, max(0, min(100, over)), max(0, min(100, under))

def calculate_parlay_probability(probs):
    combined = 1.0
    for p in probs: combined *= p
    return combined
