from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
)

VERSION_STRING = "v1.0.0 — Team Odds (Moneyline / ATS / O/U)"
TRAINABLE_TARGETS = ["moneyline", "spread", "over_under"]

# ----- Expected target columns in historical data -----
TARGET_COLUMN_MAP = {
    "moneyline": "result_home_win",     # 1 = home team wins, 0 = home loses
    "spread": "result_home_cover",      # 1 = home covers (consider spread), 0 = does not cover
    "over_under": "result_over",        # 1 = game goes over the total, 0 = under
}

# ----- Optional odds column names for upcoming data -----
ODDS_COLUMNS = {
    "moneyline": ["home_ml", "away_ml"],  # American odds (e.g., -120, +140)
    "spread": ["spread", "home_spread_odds", "away_spread_odds"],    # spread (home-line), and odds
    "over_under": ["total", "over_odds", "under_odds"],              # total line and odds
}

# ----- Helpful metadata (not required but improves UX) -----
META_COLUMNS = ["game_id", "game_date", "home_team", "away_team"]

# =====================================================================================
# Data Schema / Parsing
# =====================================================================================

@dataclass
class ParsedSchema:
    errors: List[str]
    numeric_feature_cols: List[str]
    available_targets: Dict[str, str]  # target -> column name present


def _infer_numeric_features(df: pd.DataFrame, exclude_cols: List[str]) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    return numeric_cols


def parse_and_validate_schema(
    df_hist: pd.DataFrame,
    expected_targets: List[str],
) -> ParsedSchema:
    errors = []
    available_targets = {}

    # Check targets exist
    for t in expected_targets:
        col = TARGET_COLUMN_MAP.get(t)
        if col is None:
            errors.append(f"Unknown target: {t}")
            continue
        if col not in df_hist.columns:
            errors.append(
                f"Missing target column for '{t}': expected `{col}` ∈ {{0,1}}"
            )
        else:
            # Basic validation: binary 0/1
            vals = set(pd.Series(df_hist[col]).dropna().unique().tolist())
            if not vals.issubset({0, 1}):
                errors.append(
                    f"Target column `{col}` for '{t}` must be binary 0/1. Found values: {sorted(vals)}"
                )
            else:
                available_targets[t] = col

    exclude = list(TARGET_COLUMN_MAP.values()) + META_COLUMNS
    numeric_feature_cols = _infer_numeric_features(df_hist, exclude_cols=exclude)
    if not numeric_feature_cols:
        errors.append(
            "No numeric features detected. Provide numeric columns (team stats, rolling averages, etc.)."
        )

    return ParsedSchema(
        errors=errors,
        numeric_feature_cols=numeric_feature_cols,
        available_targets=available_targets,
    )

# =====================================================================================
# Odds helpers
# =====================================================================================

def american_to_implied_prob(american_odds: Optional[float]) -> Optional[float]:
    """
    Convert American odds (e.g., -120, +145) to implied probability in [0,1].
    Returns None for NaN/None.
    """
    if american_odds is None:
        return None
    try:
        o = float(american_odds)
    except Exception:
        return None
    if np.isnan(o):
        return None
    if o < 0:
        return (-o) / ((-o) + 100.0)
    else:
        return 100.0 / (o + 100.0)


def expected_value_per_100(model_prob: float, american_odds: float) -> float:
    """
    EV for a $100 stake, assuming even settlement on win (no fees).
    EV = p * win_amount - (1-p) * 100
    where win_amount = 100 if +100, or for American odds:
        - positive odds +X -> profit = X on $100 stake
        - negative odds -Y -> profit = 100*100/Y on $100 stake? Careful:
          For -Y, to win $100 profit, you stake Y. Here we fix stake at $100:
          Profit if win = 100 * (100 / Y)
    """
    o = float(american_odds)
    if o >= 100:
        win_profit = o  # +X yields X profit on $100 stake
    elif o > 0:
        win_profit = o  # covers fractional oddities
    elif o < 0:
        win_profit = 100.0 * (100.0 / abs(o))
    else:
        win_profit = 0.0
    return model_prob * win_profit - (1.0 - model_prob) * 100.0

# =====================================================================================
# Modeling
# =====================================================================================

def _make_preprocessor(numeric_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _make_classifier(model_type: str) -> Any:
    model_type = model_type.lower()
    if model_type == "log_reg":
        return LogisticRegression(max_iter=2000, n_jobs=None)
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        )
    if model_type == "gbm":
        return GradientBoostingClassifier(random_state=42)
    raise ValueError(f"Unknown model_type: {model_type}")


def _evaluate_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, folds: int) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    # Use probability-based metrics where possible
    acc = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy").mean()
    try:
        auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc").mean()
    except Exception:
        auc = np.nan
    f1 = cross_val_score(pipeline, X, y, cv=cv, scoring="f1").mean()
    prec = cross_val_score(pipeline, X, y, cv=cv, scoring="precision").mean()
    rec = cross_val_score(pipeline, X, y, cv=cv, scoring="recall").mean()
    return {"cv_accuracy": round(float(acc), 4),
            "cv_auc": None if np.isnan(auc) else round(float(auc), 4),
            "cv_f1": round(float(f1), 4),
            "cv_precision": round(float(prec), 4),
            "cv_recall": round(float(rec), 4)}


def train_model(
    df_hist: pd.DataFrame,
    target: str,
    model_type: str = "log_reg",
    cv_folds: int = 5,
    calibrate: bool = True,
) -> Dict[str, Any]:
    """
    Trains a binary classifier for the given target.
    Returns dict with: pipeline, metrics, feature_cols, target_col.
    """
    target_col = TARGET_COLUMN_MAP[target]
    exclude_cols = list(TARGET_COLUMN_MAP.values()) + META_COLUMNS
    feature_cols = df_hist.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in exclude_cols]

    X = df_hist[feature_cols].copy()
    y = df_hist[target_col].astype(int).copy()

    pre = _make_preprocessor(feature_cols)
    base_clf = _make_classifier(model_type)

    if calibrate:
        # Wrap with calibration (Platt scaling)
        clf = CalibratedClassifierCV(base_estimator=base_clf, method="sigmoid", cv=3)
    else:
        clf = base_clf

    pipe = Pipeline(steps=[("prep", pre), ("clf", clf)])

    # CV
    metrics = _evaluate_cv(pipe, X, y, folds=cv_folds)

    # Fit on all data
    pipe.fit(X, y)

    return {
        "pipeline": pipe,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "model_type": model_type,
        "target": target,
    }

# =====================================================================================
# Prediction / EV Logic
# =====================================================================================

def _predict_prob_home(pipe: Pipeline, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].copy()
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[:, 1]
    else:
        # Some classifiers may not expose predict_proba in rare cases
        preds = pipe.predict(X)
        proba = preds.astype(float)
    return np.clip(proba, 0.0, 1.0)


def _safe_float(x):
    try:
        if x is None:
            return None
        f = float(x)
        return f
    except Exception:
        return None


def _add_moneyline_ev(row: pd.Series, model_prob_home: float) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    Return (book_implied_prob, edge, ev_per_$100, prediction_side)
    """
    home_ml = _safe_float(row.get("home_ml"))
    away_ml = _safe_float(row.get("away_ml"))

    if home_ml is None or away_ml is None:
        return None, None, None, "home" if model_prob_home >= 0.5 else "away"

    # Choose the side with bigger positive EV
    p_home = model_prob_home
    p_away = 1.0 - p_home

    book_p_home = american_to_implied_prob(home_ml)
    book_p_away = american_to_implied_prob(away_ml)

    ev_home = expected_value_per_100(p_home, home_ml)
    ev_away = expected_value_per_100(p_away, away_ml)

    if ev_home >= ev_away:
        return book_p_home, p_home - book_p_home, ev_home, "home"
    else:
        return book_p_away, p_away - book_p_away, ev_away, "away"


def _add_spread_ev(row: pd.Series, model_prob_home_cover: float) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    Spread line is from home perspective: negative means home favored.
    We only use the **odds** here to produce EV (not the numeric spread), since the classifier already models cover prob.
    """
    home_odds = _safe_float(row.get("home_spread_odds"))
    away_odds = _safe_float(row.get("away_spread_odds"))

    if home_odds is None or away_odds is None:
        side = "home_cover" if model_prob_home_cover >= 0.5 else "away_cover"
        return None, None, None, side

    p_home = model_prob_home_cover
    p_away = 1.0 - p_home

    book_p_home = american_to_implied_prob(home_odds)
    book_p_away = american_to_implied_prob(away_odds)

    ev_home = expected_value_per_100(p_home, home_odds)
    ev_away = expected_value_per_100(p_away, away_odds)

    if ev_home >= ev_away:
        return book_p_home, p_home - book_p_home, ev_home, "home_cover"
    else:
        return book_p_away, p_away - book_p_away, ev_away, "away_cover"


def _add_total_ev(row: pd.Series, model_prob_over: float) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    over_odds = _safe_float(row.get("over_odds"))
    under_odds = _safe_float(row.get("under_odds"))

    if over_odds is None or under_odds is None:
        side = "over" if model_prob_over >= 0.5 else "under"
        return None, None, None, side

    p_over = model_prob_over
    p_under = 1.0 - p_over

    book_p_over = american_to_implied_prob(over_odds)
    book_p_under = american_to_implied_prob(under_odds)

    ev_over = expected_value_per_100(p_over, over_odds)
    ev_under = expected_value_per_100(p_under, under_odds)

    if ev_over >= ev_under:
        return book_p_over, p_over - book_p_over, ev_over, "over"
    else:
        return book_p_under, p_under - book_p_under, ev_under, "under"


def predict_for_upcoming(model_obj: Dict[str, Any], df_future: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Produce predictions and EV for upcoming games for a given trained model/target.
    Returns a tidy DataFrame per target.
    """
    pipe = model_obj["pipeline"]
    feature_cols = model_obj["feature_cols"]

    # Attempt to align features; fill missing
    for col in feature_cols:
        if col not in df_future.columns:
            df_future[col] = np.nan

    # For display safety
    for c in META_COLUMNS:
        if c not in df_future.columns:
            df_future[c] = None

    # Predict probability for "home side" notion:
    # moneyline: prob home wins
    # spread: prob home covers
    # over_under: prob of OVER
    probs = _predict_prob_home(pipe, df_future, feature_cols)

    out = pd.DataFrame({
        "game_id": df_future.get("game_id", pd.Series([None]*len(df_future))),
        "game_date": df_future.get("game_date", pd.Series([None]*len(df_future))),
        "home_team": df_future.get("home_team", pd.Series([None]*len(df_future))),
        "away_team": df_future.get("away_team", pd.Series([None]*len(df_future))),
    })

    if target == "moneyline":
        out["model_prob"] = probs
        # Pick & EV
        bprob, edge, ev, side = [], [], [], []
        for _, r in df_future.iterrows():
            book_p, ed, ev100, pick = _add_moneyline_ev(r, model_prob_home=float(probs[_]))
            bprob.append(book_p)
            edge.append(ed)
            ev.append(ev100)
            side.append(pick)

        # Attach odds if present
        for c in ODDS_COLUMNS["moneyline"]:
            if c in df_future.columns:
                out[c] = df_future[c].values

    elif target == "spread":
        out["model_prob"] = probs
        bprob, edge, ev, side = [], [], [], []
        for _, r in df_future.iterrows():
            book_p, ed, ev100, pick = _add_spread_ev(r, model_prob_home_cover=float(probs[_]))
            bprob.append(book_p)
            edge.append(ed)
            ev.append(ev100)
            side.append(pick)

        # Useful to display spread & odds
        for c in ODDS_COLUMNS["spread"]:
            if c in df_future.columns:
                out[c] = df_future[c].values

    elif target == "over_under":
        out["model_prob"] = probs
        bprob, edge, ev, side = [], [], [], []
        for _, r in df_future.iterrows():
            book_p, ed, ev100, pick = _add_total_ev(r, model_prob_over=float(probs[_]))
            bprob.append(book_p)
            edge.append(ed)
            ev.append(ev100)
            side.append(pick)

        for c in ODDS_COLUMNS["over_under"]:
            if c in df_future.columns:
                out[c] = df_future[c].values
    else:
        raise ValueError(f"Unknown target: {target}")

    out["book_implied_prob"] = bprob
    out["edge"] = edge
    out["ev_per_$100"] = ev
    out["prediction_side"] = side

    return out
