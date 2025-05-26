"""
Feature-Engineering & Vorverarbeitung
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

NUM_FEATS = [
    "imp_home", "imp_draw", "imp_away",
    "form_last5", "goal_diff", "xg_diff", "h2h_home_winrate"
]

def add_implied_prob(df: pd.DataFrame) -> pd.DataFrame:
    for s in ("home", "draw", "away"):
        df[f"imp_{s}"] = 1 / df[f"odds_{s}"]
    total = df[[f"imp_{s}" for s in ("home", "draw", "away")]].sum(axis=1)
    for s in ("home", "draw", "away"):
        df[f"imp_{s}"] /= total
    return df

def add_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values("date")
    df["points_home"] = df["result"].map({"H": 3, "D": 1, "A": 0})
    df["points_away"] = df["result"].map({"H": 0, "D": 1, "A": 3})

    form_vals = []
    for team in pd.unique(df[["home_team", "away_team"]].values.ravel()):
        mask = (df.home_team == team) | (df.away_team == team)
        pts = np.where(df.home_team == team, df.points_home,
                       np.where(df.away_team == team, df.points_away, np.nan))
        form_vals.append(pd.Series(pts).rolling(window, min_periods=1).mean())
    df["form_last5"] = np.vstack(form_vals).max(axis=0)
    return df

def add_goal_xg_diff(df: pd.DataFrame) -> pd.DataFrame:
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    if {"xg_home", "xg_away"}.issubset(df.columns):
        df["xg_diff"] = df["xg_home"] - df["xg_away"]
    else:
        df["xg_diff"] = np.nan
    return df

def build_preprocessor():
    return ColumnTransformer([("num", "passthrough", NUM_FEATS)], remainder="drop")