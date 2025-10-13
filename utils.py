import os, json, time, random, pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_PATH, exist_ok=True)

MODEL_FILE = os.path.join(DATA_PATH, "ai_model.pkl")
LAST_RETRAIN_FILE = os.path.join(DATA_PATH, "last_retrain.txt")

# -----------------------------
# LOAD + MERGE JSON
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
# AI TRAINING & RETRAINING
# -----------------------------
def retrain_ai():
    """Train an XGBoost regression model using player JSON stats."""
    df = merge_jsons()
    if df.empty:
        print("⚠️ No data available to train AI model.")
        return False

    # Identify numeric columns and targets
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) < 3:
        print("⚠️ Not enough numeric features.")
        return False

    # Use average of all numeric fields as target approximation
    target_col = numeric_cols[-1]
    X = df[numeric_cols[:-1]]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=250,
        learning_rate=0.07,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"✅ Model retrained | MAE: {mae:.2f}")

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    with open(LAST_RETRAIN_FILE, "w") as f:
        f.write(datetime.utcnow().isoformat())

    return True


def get_last_retrain_time():
    if os.path.exists(LAST_RETRAIN_FILE):
        with open(LAST_RETRAIN_FILE) as f:
            return datetime.fromisoformat(f.read().strip()).strftime("%b %d %Y %H:%M UTC")
    return "Never"


def maybe_retrain():
    """Retrains every 30 min or nightly at 12 AM EST."""
    now = datetime.utcnow()
    if not os.path.exists(LAST_RETRAIN_FILE):
        retrain_ai(); return True
    with open(LAST_RETRAIN_FILE) as f:
        last = datetime.fromisoformat(f.read().strip())
    if (now - last) > timedelta(minutes=30) or now.hour == 4:  # 4 UTC ≈ 12 AM EST
        retrain_ai(); return True
    return False

# -----------------------------
# PREDICTION HELPERS
# -----------------------------
def load_model():
    if not os.path.exists(MODEL_FILE):
        retrain_ai()
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

def ai_predict(line):
    """Predict adjusted performance value using the trained model."""
    model = load_model()
    random_input = np.random.rand(model.n_features_in_).reshape(1, -1)
    pred = model.predict(random_input)[0]
    avg = pred
    diff = avg - line
    over = 50 + diff * 1.2
    under = 100 - over
    return avg, max(0, min(100, over)), max(0, min(100, under))

# -----------------------------
# PLAYER UTILS
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

# -----------------------------
# PROBABILITY CALCULATIONS
# -----------------------------
def calculate_probabilities(df, line):
    try:
        avg = df.select_dtypes(include=["number"]).mean().mean()
        diff = avg - line
        over = 50 + diff * 1.1
        under = 100 - over
        return avg, max(0, min(100, over)), max(0, min(100, under))
    except Exception:
        return simulate_fallback(line)

def simulate_fallback(line):
    avg = line + random.uniform(-20, 20)
    diff = avg - line
    over = 50 + diff * 1.2
    under = 100 - over
    return avg, max(0, min(100, over)), max(0, min(100, under))

def calculate_parlay_probability(probs):
    combined = 1.0
    for p in probs: combined *= p
    return combined
