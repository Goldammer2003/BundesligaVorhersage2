# src/features.py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# —> goal_diff hier entfernt
NUM_FEATS = [
    "form_last5",
    "xg_diff",
    "h2h_home_winrate",
    "avg_goals_home_last10", "avg_goals_away_last10",
    "winrate_last10_home",   "winrate_last10_away",
]

def add_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values("date")
    df["home_pts"] = df["result"].map({"H":3,"D":1,"A":0})
    df["away_pts"] = df["result"].map({"H":0,"D":1,"A":3})
    rolling = []
    for team in pd.unique(df[["home_team","away_team"]].values.ravel()):
        pts = np.where(df.home_team == team, df.home_pts,
                       np.where(df.away_team == team, df.away_pts, np.nan))
        rolling.append(pd.Series(pts).rolling(window, min_periods=1).mean())
    df["form_last5"] = np.vstack(rolling).max(axis=0)
    return df

def add_goal_xg_diff(df: pd.DataFrame) -> pd.DataFrame:
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    if {"xg_home","xg_away"}.issubset(df.columns):
        df["xg_diff"] = df["xg_home"] - df["xg_away"]
    else:
        df["xg_diff"] = np.nan
    return df

def add_rolling_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.sort_values("date")
    recs = []
    for _, r in df.iterrows():
        recs.append({"team":r.home_team,"date":r.date,
                     "goals_for":r.home_goals,"win":int(r.result=="H")})
        recs.append({"team":r.away_team,"date":r.date,
                     "goals_for":r.away_goals,"win":int(r.result=="A")})
    rec = pd.DataFrame(recs).sort_values(["team","date"])
    rec["avg_goals_last10"] = rec.groupby("team")["goals_for"] \
                                 .rolling(window, min_periods=1).mean() \
                                 .reset_index(0,drop=True)
    rec["winrate_last10"]   = rec.groupby("team")["win"] \
                                 .rolling(window, min_periods=1).mean() \
                                 .reset_index(0,drop=True)
    df = df.merge(
        rec[["team","date","avg_goals_last10","winrate_last10"]] \
           .rename(columns={
               "team":"home_team",
               "avg_goals_last10":"avg_goals_home_last10",
               "winrate_last10":"winrate_last10_home"
           }),
        on=["home_team","date"], how="left"
    ).merge(
        rec[["team","date","avg_goals_last10","winrate_last10"]] \
           .rename(columns={
               "team":"away_team",
               "avg_goals_last10":"avg_goals_away_last10",
               "winrate_last10":"winrate_last10_away"
           }),
        on=["away_team","date"], how="left"
    )
    return df

def build_preprocessor():
    return ColumnTransformer(
        [("num", StandardScaler(), NUM_FEATS)],
        remainder="drop"
    )