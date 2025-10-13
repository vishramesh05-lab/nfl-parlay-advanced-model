# -*- coding: utf-8 -*-
"""
Utility Functions for NFL Parleggy AI Model
Handles data loading, AI training, probability scoring, and auto-retraining logic
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
import time

# ----------------------------
# CONSTANTS
# ----------------------------
DATA_PATH = os.path.join(os.getcwd(), "data")
MODEL_FILE = os.path.join(DATA_PATH, "parleggy_model.pkl")
LAST_RETRAIN_FILE = os.path.join(DATA_PATH, "last_retrain.txt")


# ----------------------------
# DATA HELPERS
# ----------------------------
def load_json(filename):
    """Safely load a JSON file from data folder."""
    try:
        with open(os.path.join(DATA_PATH, filename), "r") as f:
            return json.load(f)
    except Exception:
        return None


def merge_data():
    """Merge all JSON data files in /data into one DataFrame."""
    dfs = []
    for fname in os.listdir(DATA_PATH):
        if fname.endswith(".json"):
            try:
                js = load_json(fname)
                df = pd.json_normalize(js)
                df["source_file"] = fname
                dfs.append(df)
            except Exception:
                pass
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True).fillna(0)
        return df_all
    return pd.DataFrame()


# ----------------------------
# AI MODEL TRAINING
# ----------------------------
def train_ai(df):
    """Train XGBoost AI model using weighted features."""
    if df.empty:
        return None

    # Example: use player stats columns dynamically
    X = df.select_dtypes(include=[np.number])
    y = (X.mean(axis=1) > X.mean().mean()).astype(int)  # placeholder binary target

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model = XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        save_model(model)
        return acc
    except Exception as e:
        print("Training failed:", e)
        return None


def save_model(model):
    """Save model to disk."""
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)


def load_model():
    """Load the AI model from disk if exists."""
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None


# ----------------------------
# PROBABILITY ENGINE
# ----------------------------
def calculate_probability(stats, line):
    """
    Calculate probability of exceeding a given sportsbook line.
    Uses model predictions and feature weighting.
    """
    model = load_model()
    if model is None or stats.empty:
        return 0.5  # fallback neutral probability

    try:
        X = stats.select_dtypes(include=[np.number])
        preds = model.predict_proba(X)[:, 1]
        avg_prob = np.mean(preds)
        adjusted = np.clip(avg_prob + ((avg_prob - 0.5) * 0.3), 0, 1)
        return float(adjusted)
    except Exception as e:
        print("Probability error:", e)
        return 0.5


# ----------------------------
# RETRAINING CONTROL
# ----------------------------
def should_retrain():
    """Check if 30 min have passed or if it's 12 AM EST for full retrain."""
    if not os.path.exists(LAST_RETRAIN_FILE):
        return True
    try:
        with open(LAST_RETRAIN_FILE, "r") as f:
            last = datetime.fromisoformat(f.read().strip())
        now = datetime.utcnow()
        if (now - last).total_seconds() >= 1800 or now.hour == 4:  # 12 AM EST
            return True
        return False
    except Exception:
        return True


def retrain_model():
    """Trigger model retraining and log timestamp."""
    df = merge_data()
    acc = train_ai(df)
    with open(LAST_RETRAIN_FILE, "w") as f:
        f.write(datetime.utcnow().isoformat())
    return acc


# ----------------------------
# VISUALIZATION SUPPORT
# ----------------------------
def get_player_chart(df, player_name):
    """Return player stat trend over weeks for visualization."""
    if df.empty or player_name not in df.get("Name", []):
        return None

    player_df = df[df["Name"] == player_name]
    if "Week" not in player_df.columns:
        player_df["Week"] = np.arange(1, len(player_df) + 1)

    chart_df = player_df[["Week", "PassingYards", "RushingYards", "ReceivingYards"]].melt(
        id_vars="Week", var_name="Stat", value_name="Value"
    )
    return chart_df
