import io
import re
import pandas as pd
import numpy as np

VERSION = "v0.9 – Player Parlay (Usage-Only, Single CSV)"

# ---------- Column normalization ----------

def _canon(s: str) -> str:
    s = re.sub(r"[\s/]+", "_", s.strip(), flags=re.I)
    s = re.sub(r"[^a-zA-Z0-9_]+", "", s, flags=re.I)
    return s.lower()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_canon(c) for c in df.columns]
    # common FantasyPros headings
    rename = {
        "player": "player",
        "team": "team",
        "tm": "team",
        "position": "pos",
        "pos": "pos",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df

# ---------- Schema detection ----------

def detect_schema(df: pd.DataFrame) -> dict:
    cols = df.columns.tolist()
    # Find week columns that look like snaps or snap%
    week_cols_snaps = [c for c in cols if re.search(r"^(w?k?|(week)?_?)\d+(_)?(snaps|off_snaps)$", c)]
    week_cols_pct   = [c for c in cols if re.search(r"^(w?k?|(week)?_?)\d+(_)?(snap_pct|off_snap_pct|off_snap_percent)$", c)]
    # Broad FantasyPros formats sometimes ship as "wk1_snaps", "wk1_snap_pct"
    if not week_cols_snaps:
        week_cols_snaps = [c for c in cols if re.search(r"^wk?\d+_snaps$", c)]
    if not week_cols_pct:
        week_cols_pct = [c for c in cols if re.search(r"^wk?\d+_snap_pct$", c)]

    # Optional columns
    routes_cols = [c for c in cols if re.search(r"(routes|routes_run)(_wk?\d+)?$", c)]
    targets_cols = [c for c in cols if re.search(r"(tgt|targets)(_wk?\d+)?$", c)]

    pos_col  = "pos"  if "pos" in cols else None
    team_col = "team" if "team" in cols else None
    name_col = "player" if "player" in cols else (cols[0] if cols else None)

    return {
        "name_col": name_col,
        "team_col": team_col,
        "pos_col": pos_col,
        "week_cols_snaps": sorted(week_cols_snaps, key=_week_key),
        "week_cols_pct":   sorted(week_cols_pct, key=_week_key),
        "routes_cols": sorted(routes_cols, key=_week_key) if routes_cols else [],
        "targets_cols": sorted(targets_cols, key=_week_key) if targets_cols else [],
    }

def _week_key(col: str) -> int:
    m = re.search(r"(\d+)", col)
    return int(m.group(1)) if m else 0

# ---------- Build weekly matrix ----------

def build_week_matrix(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    df = df.copy()
    name = schema["name_col"] or "player"
    team = schema["team_col"] or "team"
    pos  = schema["pos_col"] or "pos"

    # Ensure present
    if name not in df.columns:
        df[name] = np.arange(len(df)).astype(str)
    if team not in df.columns:
        df[team] = "UNK"
    if pos not in df.columns:
        df[pos] = "UNK"

    wk_cols_pct   = schema["week_cols_pct"]
    wk_cols_snaps = schema["week_cols_snaps"]
    routes_cols   = schema["routes_cols"]
    targets_cols  = schema["targets_cols"]

    # Prefer snap% if available; else derive from snaps by within-team max (approx)
    weeks = sorted({*_weeks_from_cols(wk_cols_pct), *_weeks_from_cols(wk_cols_snaps)})
    rows = []
    for _, r in df.iterrows():
        for w in weeks:
            snap_pct = _get_week_value(r, wk_cols_pct, w)
            snaps    = _get_week_value(r, wk_cols_snaps, w)
            routes   = _get_week_value(r, routes_cols, w)
            targets  = _get_week_value(r, targets_cols, w)
            rows.append({
                "player": r[name],
                "team": r[team],
                "pos": r[pos],
                "week": w,
                "snap_pct": snap_pct,
                "snaps": snaps,
                "routes": routes,
                "targets": targets,
            })
    out = pd.DataFrame(rows)

    # Clean types
    for c in ["snap_pct", "snaps", "routes", "targets"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop rows with no information at all
    keep = (~out["snap_pct"].isna()) | (~out["snaps"].isna()) | (~out["routes"].isna()) | (~out["targets"].isna())
    out = out.loc[keep].copy()

    # If snap_pct missing but snaps exist, convert to within-player relative (fallback)
    # (We avoid team normalization here to stay single-CSV)
    out["snap_pct"] = out.groupby("player")["snaps"].apply(lambda s: 100 * s / s.max() if s.max() and s.max() > 0 else s) \
                        .where(out["snap_pct"].isna(), out["snap_pct"])

    # Drop duplicates and sort
    out = out.drop_duplicates(subset=["player", "week"]).sort_values(["player", "week"]).reset_index(drop=True)
    return out

def _weeks_from_cols(cols):
    wk = []
    for c in cols:
        m = re.search(r"(\d+)", c)
        if m:
            wk.append(int(m.group(1)))
    return wk

def _get_week_value(row, cols, w):
    # find a column in 'cols' whose digits match week w
    for c in cols:
        m = re.search(r"(\d+)", c)
        if m and int(m.group(1)) == int(w):
            return row.get(c, np.nan)
    return np.nan

# ---------- Features & scoring ----------

def compute_usage_features(wk_df: pd.DataFrame, routes_weight=0.6, targets_weight=0.8, l3_weight=1.0, trend_weight=1.0):
    df = wk_df.copy()
    df["snap_pct"] = df["snap_pct"].clip(0, 100)

    # last week by player
    last_weeks = df.groupby("player")["week"].max().rename("week_last")
    t = df.merge(last_weeks, on="player", how="left")

    # last-3-weeks rolling stats
    t = t.sort_values(["player", "week"])
    for col in ["snap_pct", "routes", "targets"]:
        t[f"l3_{col}"] = t.groupby("player")[col].transform(lambda s: s.rolling(3, min_periods=1).mean())

    # week-over-week trend (delta of snap%)
    t["wow_snap_pct"] = t.groupby("player")["snap_pct"].diff().fillna(0.0)

    # keep only the latest row per player for scoring
    latest = t[t["week"] == t["week_last"]].copy()

    # games_count in dataset
    latest["games_count"] = t.groupby("player")["week"].transform("count").loc[latest.index]

    # usage score (bounded)
    # base: snap% last-3-weeks
    base = (latest["l3_snap_pct"].fillna(0) / 100.0) * l3_weight
    # trend: wow delta (percentage points -> scale)
    trend = (latest["wow_snap_pct"].fillna(0) / 50.0) * trend_weight  # ~±2.0 for ±100pp (guarded)
    # optional routes/targets
    rts = (latest["l3_routes"].fillna(0))
    tgts = (latest["l3_targets"].fillna(0))
    # normalize routes/targets to [0,1] per position to avoid bias
    for c, w in [("l3_routes", routes_weight), ("l3_targets", targets_weight)]:
        mx = latest.groupby("pos")[c].transform(lambda s: s.fillna(0).quantile(0.95) if (s.notna().any()) else 1.0)
        mx = mx.replace(0, 1.0)
        latest[f"norm_{c}"] = latest[c].fillna(0) / mx
    rts_n = latest["norm_l3_routes"]
    tgts_n = latest["norm_l3_targets"]

    usage = base + trend + routes_weight * rts_n + targets_weight * tgts_n
    latest["usage_score"] = usage.clip(lower=0)

    # friendly columns
    keep_cols = [
        "player","team","pos","week_last","games_count",
        "l3_snap_pct","wow_snap_pct","l3_routes","l3_targets","usage_score"
    ]
    return latest[keep_cols].sort_values("usage_score", ascending=False).reset_index(drop=True)

# ---------- Recommendation logic ----------

POS_ORDER = ["RB","WR","TE","QB","FB","HB"]

def _pos_norm(s):
    s = str(s).upper()
    if s.startswith("RB"): return "RB"
    if s.startswith("WR"): return "WR"
    if s.startswith("TE"): return "TE"
    if s.startswith("QB"): return "QB"
    return s

def rank_recommendations_by_pos(feat_df: pd.DataFrame, per_pos: int = 8):
    x = feat_df.copy()
    x["pos"] = x["pos"].map(_pos_norm)

    # Over candidates: highest usage scores within position
    overs = []
    for p in POS_ORDER:
        sub = x[x["pos"] == p].copy()
        if not sub.empty:
            sub = sub.sort_values(["usage_score","l3_snap_pct","wow_snap_pct"], ascending=[False, False, False]).head(per_pos)
            sub.insert(0, "pick", "OVER")
            overs.append(sub)
    over_df = pd.concat(overs, ignore_index=True) if overs else pd.DataFrame(columns=x.columns)

    # Under candidates: lowest usage scores (and low/negative trend)
    unders = []
    for p in POS_ORDER:
        sub = x[x["pos"] == p].copy()
        if not sub.empty:
            sub["trend_rank"] = sub["wow_snap_pct"].rank(method="first")
            sub = sub.sort_values(["usage_score","wow_snap_pct","l3_snap_pct"], ascending=[True, True, True]).head(per_pos)
            sub.insert(0, "pick", "UNDER")
            unders.append(sub.drop(columns=["trend_rank"], errors="ignore"))
    under_df = pd.concat(unders, ignore_index=True) if unders else pd.DataFrame(columns=x.columns)

    # Format
    show_cols = ["pick","player","team","pos","week_last","games_count","l3_snap_pct","wow_snap_pct","l3_routes","l3_targets","usage_score"]
    return {"over": over_df[show_cols], "under": under_df[show_cols]}

# ---------- Export ----------

def export_recommendations_csv(recs: dict) -> bytes:
    df = pd.concat([
        recs["over"].assign(list_type="over"),
        recs["under"].assign(list_type="under")
    ], ignore_index=True)
    out = io.StringIO()
    df.to_csv(out, index=False)
    return out.getvalue().encode("utf-8")
