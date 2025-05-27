"""
Feature-Engineering & Vorverarbeitung
"""
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

NUM_FEATS = [
    # Wettmarkt (implizite Wahrscheinlichkeit)
    "imp_home", "imp_draw", "imp_away",
    # Team-Form
    "form_last5",
    # Tore und xG Differenzen
    "goal_diff", "xg_diff",
    # Head-to-Head
    "h2h_home_winrate"
]

def add_implied_prob(df: pd.DataFrame) -> pd.DataFrame:
    # Falls Quoten fehlen → uniform verteilen, ansonsten implizite p berechnen
    for s in ("home", "draw", "away"):
        if f"odds_{s}" not in df.columns:
            df[f"imp_{s}"] = 1/3
        else:
            df[f"imp_{s}"] = 1 / df[f"odds_{s}"]
    total = df[[f"imp_{s}" for s in ("home", "draw", "away")]].sum(axis=1)
    for s in ("home", "draw", "away"):
        df[f"imp_{s}"] /= total.replace(0, np.nan)
    return df

def add_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    # Rolling-Average der letzten x Spiele
    df = df.sort_values("date")
    df["home_pts"] = df["result"].map({"H":3, "D":1, "A":0})
    df["away_pts"] = df["result"].map({"H":0, "D":1, "A":3})
    form = []
    for team in pd.unique(df[["home_team", "away_team"]].values.ravel()):
        mask = (df.home_team == team) | (df.away_team == team)
        pts  = np.where(df.home_team == team, df.home_pts,
                        np.where(df.away_team == team, df.away_pts, np.nan))
        form.append(pd.Series(pts).rolling(window, min_periods=1).mean())
    df["form_last5"] = np.vstack(form).max(axis=0)
    return df

def add_goal_xg_diff(df: pd.DataFrame) -> pd.DataFrame:
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    if {"xg_home", "xg_away"}.issubset(df.columns):
        df["xg_diff"] = df["xg_home"] - df["xg_away"]
    else:
        df["xg_diff"] = np.nan
    return df

def build_preprocessor():
    return ColumnTransformer(
        [("num", StandardScaler(), NUM_FEATS)],
        remainder="drop"
    )