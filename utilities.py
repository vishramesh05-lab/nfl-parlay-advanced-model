from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import io
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

VERSION = "v1.2 — Team Odds (JSON/CSV uploads, matchup simulator)"
TRAINABLE_TARGETS = ["moneyline", "spread", "over_under"]

META_COLS = [
    "GlobalGameID","ScoreID","Date","DateTime","GameDate","Week","Season","SeasonType",
    "Stadium","PlayingSurface","HomeOrAway","Team","Opponent","home_team","away_team",
    "game_id","game_date",
]

# ==================== Ingest ====================

def _try_read(file) -> pd.DataFrame:
    name = (getattr(file, "name", "") or "").lower()
    if name.endswith(".json"):
        try:
            data = json.load(file)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # maybe an id->obj mapping
                try:
                    return pd.DataFrame(list(data.values()))
                except Exception:
                    return pd.json_normalize(data)
        except Exception:
            file.seek(0)
            text = file.read()
            try:
                data = json.loads(text)
                return pd.DataFrame(data if isinstance(data, list) else [data])
            except Exception:
                pass
    file.seek(0)
    # csv fallback
    try:
        return pd.read_csv(file)
    except Exception:
        file.seek(0)
        return pd.read_json(file, lines=True)

def ingest_files_to_games_df(files) -> tuple[pd.DataFrame, Dict[str, Any]]:
    frames = []
    notes = {}
    for f in files:
        try:
            df = _try_read(f)
            notes[getattr(f, "name", "file")] = {"rows": len(df), "cols": list(df.columns)[:20]}
            frames.append(df)
        except Exception as e:
            notes[getattr(f, "name", "file")] = {"error": str(e)}
    if not frames:
        return pd.DataFrame(), notes

    df = pd.concat(frames, ignore_index=True, sort=False)
    # canonicalize some names
    ren = {
        "TeamName":"Team",
        "Day":"Date",
        "OpponentTeam":"Opponent",
        "HomeAway":"HomeOrAway",
        "OverUnder":"OverUnder",
        "PointSpread":"PointSpread",
        "TotalScore":"TotalScore",
    }
    for k, v in ren.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    # Ensure basic expected columns if present in variations
    if "Team" not in df.columns and "team" in df.columns:
        df["Team"] = df["team"]
    if "Opponent" not in df.columns and "opponent" in df.columns:
        df["Opponent"] = df["opponent"]
    if "HomeOrAway" not in df.columns and "homeoraway" in df.columns:
        df["HomeOrAway"] = df["homeoraway"].str.upper()

    # numeric coerce for key stats if present
    for c in ["Score","OpponentScore","OverUnder","PointSpread","TotalScore"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, notes

# ==================== Transform to home rows ====================

def build_home_game_rows(df_team: pd.DataFrame) -> pd.DataFrame:
    df = df_team.copy()
    # Some sources include TotalScore; if not, compute it.
    if "TotalScore" not in df.columns and {"Score","OpponentScore"}.issubset(df.columns):
        df["TotalScore"] = df["Score"] + df["OpponentScore"]

    # Identify home records
    if "HomeOrAway" in df.columns:
        home_mask = df["HomeOrAway"].astype(str).str.upper().eq("HOME")
        # If we have a unique game id use it to pick the home
        if "GlobalGameID" in df.columns:
            home_rows = df[home_mask].copy()
            # If there are cases with no home row (only away), approximate by picking max ScoreID/first
            if home_rows.empty:
                # Fall back: try to create home rows by picking one per game and mapping columns
                home_rows = df.sort_values("ScoreID").groupby("GlobalGameID", as_index=False).first()
        else:
            # Fallback: rely on HomeOrAway
            home_rows = df[home_mask].copy()
    else:
        # If HomeOrAway missing, try to infer: treat first of (Team,Opponent,Date) as home (best effort)
        key_cols = [c for c in ["Date","DateTime","GameDate","Week","Season"] if c in df.columns]
        grp_cols = key_cols + ["Team","Opponent"]
        df["_gid"] = pd.factorize(df[grp_cols].astype(str).agg("|".join, axis=1))[0]
        home_rows = df.drop_duplicates(subset=["_gid"]).copy()

    # Rename teams
    home_rows["home_team"] = home_rows.get("Team")
    home_rows["away_team"] = home_rows.get("Opponent")

    # Targets will be computed later. Keep all numeric features.
    # Drop pure opponent team strings etc. Keep a wide table.
    return home_rows.reset_index(drop=True)

# ==================== Targets from lines/scores ====================

def compute_targets_from_lines(home_rows: pd.DataFrame) -> pd.DataFrame:
    df = home_rows.copy()
    # Moneyline: 1 if home won
    if {"Score","OpponentScore"}.issubset(df.columns):
        df["moneyline"] = (df["Score"] > df["OpponentScore"]).astype(int)

    # Spread (ATS): need PointSpread from the home perspective.
    # Many feeds give per-team spread (home rows should represent HOME team line).
    # We'll assume df["PointSpread"] is the **home team's** line (negative if favored).
    if {"Score","OpponentScore","PointSpread"}.issubset(df.columns):
        margin = df["Score"] - df["OpponentScore"]
        df["spread"] = (margin + df["PointSpread"]) > 0  # home + spread beats opponent
        df["spread"] = df["spread"].astype(int)

    # Over/Under: if OverUnder and TotalScore present
    if {"OverUnder","TotalScore"}.issubset(df.columns):
        df["over_under"] = (df["TotalScore"] > df["OverUnder"]).astype(int)

    return df

# ==================== Modeling ====================

def _numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    ex = set(META_COLS + TRAINABLE_TARGETS + ["Score","OpponentScore"])
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in ex]

def _preprocessor(numeric_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("num", PipelineSimple(), numeric_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

class PipelineSimple(SimpleImputer):
    # small helper: impute median then scale
    def __init__(self):
        super().__init__(strategy="median")
        self.scaler = StandardScaler()
    def fit(self, X, y=None):
        Z = super().fit_transform(X)
        self.scaler.fit(Z)
        return self
    def transform(self, X):
        Z = super().transform(X)
        return self.scaler.transform(Z)

def _make_clf(model_type: str):
    m = model_type.lower()
    if m == "log_reg":
        return LogisticRegression(max_iter=2000)
    if m == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample",
        )
    if m == "gbm":
        return GradientBoostingClassifier(random_state=42)
    raise ValueError(f"Unknown model_type: {model_type}")

def _cv_metrics(pipe, X, y, folds=5):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    def score(sc):
        try:
            return float(cross_val_score(pipe, X, y, cv=cv, scoring=sc).mean())
        except Exception:
            return np.nan
    return {
        "cv_accuracy": round(score("accuracy"), 4),
        "cv_auc": None if np.isnan(score("roc_auc")) else round(score("roc_auc"), 4),
        "cv_f1": round(score("f1"), 4),
        "cv_precision": round(score("precision"), 4),
        "cv_recall": round(score("recall"), 4),
    }

def train_model(df_home: pd.DataFrame, target: str, model_type="log_reg", cv_folds=5, calibrate=True):
    if target not in df_home.columns:
        raise ValueError(f"Target '{target}' not present.")
    y = df_home[target].astype(int)
    X_cols = _numeric_feature_cols(df_home)
    X = df_home[X_cols].copy()

    prep = _preprocessor(X_cols)
    base = _make_clf(model_type)
    if calibrate:
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    else:
        clf = base

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("prep", prep), ("clf", clf)])

    metrics = _cv_metrics(pipe, X, y, folds=cv_folds)
    pipe.fit(X, y)

    return {"pipeline": pipe, "metrics": metrics, "feature_cols": X_cols, "target": target}

def predict_proba_for_homerow(model_obj: Dict[str, Any], home_row: Dict[str, Any]) -> float:
    X_cols = model_obj["feature_cols"]
    x = pd.DataFrame([{c: home_row.get(c, np.nan) for c in X_cols}])
    proba = model_obj["pipeline"].predict_proba(x)[:, 1][0]
    return float(np.clip(proba, 0.0, 1.0))

# ==================== Odds helpers ====================

def american_to_implied_prob(o: Optional[float]) -> Optional[float]:
    if o is None: return None
    try:
        o = float(o)
    except Exception:
        return None
    if np.isnan(o): return None
    if o < 0:  # favorite
        return (-o) / ((-o) + 100.0)
    else:
        return 100.0 / (o + 100.0)

def expected_value_per_100(p: float, american_odds: float) -> float:
    o = float(american_odds)
    if o < 0:
        win_profit = 100.0 * (100.0 / abs(o))
    else:
        win_profit = o
    return p * win_profit - (1.0 - p) * 100.0

# ==================== Matchup synthesis ====================

def _safe_float(x):
    try:
        f = float(x)
        if np.isnan(f): return None
        return f
    except Exception:
        return None

def make_homerow_from_rolling_avgs(
    home_rows_all: pd.DataFrame,
    team_home: str,
    team_away: str,
    last_n: int = 4,
    spread_text: str = "",
    total_text: str = "",
    home_ml_text: str = "",
    away_ml_text: str = "",
    home_spread_odds_text: str = "",
    away_spread_odds_text: str = "",
    over_odds_text: str = "",
    under_odds_text: str = "",
) -> Dict[str, Any]:
    df = home_rows_all.copy()

    # Build per-team recent averages using the *home-row schema*. For the away team,
    # we will map home metrics <-> opponent metrics.
    def team_recent_avg(team_name: str):
        # consider games where team was home (home_rows) and also where it appeared as away
        # Construct a "team view": if the selected team was AWAY in a home_row, swap columns.
        hx = df[(df["home_team"] == team_name) | (df["away_team"] == team_name)].copy()
        if hx.empty:
            return {}

        # Re-map to a unified "team perspective": columns that start with "Opponent" should come from the opponent.
        # If the team is home in that row, keep as-is; if the team is away, we need to swap Team<->Opponent fields.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        rows = []
        for _, r in hx.iterrows():
            row = {}
            is_home = (r["home_team"] == team_name)
            for c in numeric_cols:
                if c in ["Score","OpponentScore"]:  # keep raw, swaps below if needed
                    pass
                v = r.get(c, np.nan)
                if is_home:
                    # team perspective: base metrics without "Opponent" prefix belong to team,
                    # and "Opponent..." belong to opponent — keep as-is
                    row[c] = v
                else:
                    # if away, swap: team metrics should use opponent-prefixed columns and vice-versa where sensible
                    if c.startswith("Opponent"):
                        # OpponentX for home row = TEAM X for away team's perspective
                        key = c[len("Opponent"):]
                        # find non-opponent name for same stat
                        alt = key if key in numeric_cols else c
                        row[c] = r.get(alt, v)
                    else:
                        # non-opponent metric: use the "opponent" version from row if it exists
                        oppc = "Opponent" + c
                        row[c] = r.get(oppc, v)
            rows.append(row)

        if not rows:
            return {}

        tdf = pd.DataFrame(rows)
        if tdf.empty:
            return {}

        # Average of last N rows
        tdf = tdf.tail(last_n)
        return tdf.mean(numeric_only=True).to_dict()

    avg_home = team_recent_avg(team_home)
    avg_away = team_recent_avg(team_away)

    # Construct one synthetic home-row dict
    synth = {}
    # For every numeric feature seen in training, set team/opponent sides from averages
    all_num = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in all_num:
        if c.startswith("Opponent"):
            base = c[len("Opponent"):]
            if base in avg_away:
                synth[c] = avg_away.get(base)
            else:
                synth[c] = avg_away.get(c, np.nan)
        else:
            synth[c] = avg_home.get(c, np.nan)

    # Set meta + betting inputs if provided
    synth["home_team"] = team_home
    synth["away_team"] = team_away

    # Betting lines (optional)
    synth["home_ml"] = _safe_float(home_ml_text)
    synth["away_ml"] = _safe_float(away_ml_text)
    synth["PointSpread"] = _safe_float(spread_text)  # home-line
    synth["OverUnder"] = _safe_float(total_text)
    synth["home_spread_odds"] = _safe_float(home_spread_odds_text)
    synth["away_spread_odds"] = _safe_float(away_spread_odds_text)
    synth["over_odds"] = _safe_float(over_odds_text)
    synth["under_odds"] = _safe_float(under_odds_text)

    return synth
