import os
import json
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from datetime import datetime, timedelta

# ==============================
# CONFIG / KEYS
# ==============================
SPORTSDATA_KEY = os.getenv("SPORTSDATA_KEY", "")
ODDS_KEY = os.getenv("ODDS_API_KEY", "")
WEATHER_KEY = os.getenv("OPENWEATHER_KEY", "")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
MODEL_FILE = os.path.join(DATA_PATH, "trained_model.pkl")

os.makedirs(DATA_PATH, exist_ok=True)


# ==============================
# DATA HELPERS
# ==============================

def list_jsons():
    """List all JSON files in the data folder."""
    return [f for f in os.listdir(DATA_PATH) if f.endswith(".json")]


def load_json(fname):
    """Safely load a single JSON file."""
    try:
        with open(os.path.join(DATA_PATH, fname)) as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading {fname}: {e}")
        return None


def merge_jsons():
    """Merge all JSON files into a single DataFrame."""
    dfs = []
    for f in list_jsons():
        try:
            js = load_json(f)
            if js:
                df = pd.json_normalize(js)
                df["source_file"] = f
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error processing {f}: {e}")
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True).fillna(0)
        return df_all
    return pd.DataFrame()


# ==============================
# MODEL TRAIN / AI FUNCTIONS
# ==============================

def train_ai(df):
    """Train a weighted XGBoost model on provided DataFrame."""
    if df.empty:
        print("⚠️ No data available for training.")
        return None

    try:
        # Simplified example: predict total offensive yards
        features = [c for c in df.columns if df[c].dtype in [np.float64, np.int64]]
        X = df[features].fillna(0)
        y = np.random.choice([0, 1], size=len(X))  # placeholder target

        model = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.12,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X, y)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        print("✅ Model trained and saved successfully.")
        return model
    except Exception as e:
        print(f"⚠️ Training error: {e}")
        return None


def retrain_ai():
    """Force manual retrain from Streamlit sidebar button."""
    df = merge_jsons()
    if df.empty:
        print("⚠️ No JSON data found for retraining.")
        return
    train_ai(df)


def maybe_retrain():
    """
    Automatically retrain the model every 30 minutes or nightly at 12 AM EST.
    Returns True if a retrain occurred.
    """
    try:
        now = datetime.utcnow()
        retrain_flag = False
        marker_file = os.path.join(DATA_PATH, "last_retrain.txt")

        # Check last retrain time
        if os.path.exists(marker_file):
            with open(marker_file, "r") as f:
                last_time = datetime.fromisoformat(f.read().strip())
        else:
            last_time = datetime.utcnow() - timedelta(hours=1)

        time_diff = (now - last_time).total_seconds() / 60

        # Nightly retrain or every 30 minutes
        if time_diff > 30 or now.hour == 4:  # 12 AM EST = 4 AM UTC
            df = merge_jsons()
            if not df.empty:
                train_ai(df)
                with open(marker_file, "w") as f:
                    f.write(now.isoformat())
                retrain_flag = True
                print("♻️ Auto-retrain completed.")
        return retrain_flag
    except Exception as e:
        print(f"⚠️ Retrain error: {e}")
        return False
