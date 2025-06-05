# src/features.py

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# ────────────────────────────────────────────────────────────────────────────
# Definition aller numerischen Features, die wir im Modell verwenden wollen.
# Darunter:
#  • form_last5             – Durchschnittliche Punktzahl des Teams aus den letzten 5 Spielen
#  • xg_diff                – Differenz (xG_home - xG_away) pro Spiel
#  • h2h_home_winrate       – Head-to-Head-Winrate des Heimteams gegen den Gast
#  • avg_goals_home_last10  – Durchschnitt Tore des Heimteams aus den letzten 10 Spielen
#  • avg_goals_away_last10  – Durchschnitt Tore des Auswärtsteams aus den letzten 10 Spielen
#  • winrate_last10_home    – prozentuale Siegquote des Heimteams aus den letzten 10 Spielen
#  • winrate_last10_away    – prozentuale Siegquote des Auswärtsteams aus den letzten 10 Spielen
# ────────────────────────────────────────────────────────────────────────────
NUM_FEATS = [
    "form_last5",
    "xg_diff",
    "h2h_home_winrate",
    "avg_goals_home_last10", "avg_goals_away_last10",
    "winrate_last10_home",   "winrate_last10_away",
]


def add_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Berechnet für jede Partie, welches Mittel der Punkte das jeweilige Team
    aus den letzten `window` Spielen erzielt hat.
    Speichert das Maximum von Home-/Away-Punkten in form_last5.
    """
    df = df.sort_values("date")
    df["home_pts"] = df["result"].map({"H": 3, "D": 1, "A": 0})
    df["away_pts"] = df["result"].map({"H": 0, "D": 1, "A": 3})
    rolling = []

    # Für jedes Team: erstelle eine Rolling-Mean-Kurve über die letzten `window` Spiele
    for team in pd.unique(df[["home_team", "away_team"]].values.ravel()):
        pts = np.where(
            df.home_team == team, df.home_pts,
            np.where(df.away_team == team, df.away_pts, np.nan)
        )
        rolling.append(pd.Series(pts).rolling(window, min_periods=1).mean())

    # Wir nehmen das Maximum, weil an einem Spieltag immer ein Team Heim- und Auswärtsspiel hat.
    df["form_last5"] = np.vstack(rolling).max(axis=0)
    return df


def add_goal_xg_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legt auf Basis der Spalten 'home_goals' und 'away_goals' ein Feature 'goal_diff' an
    und berechnet, falls vorhanden, xG-Differenz (xg_home - xg_away).
    """
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    if {"xg_home", "xg_away"}.issubset(df.columns):
        df["xg_diff"] = df["xg_home"] - df["xg_away"]
    else:
        df["xg_diff"] = np.nan
    return df


def add_rolling_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Berechnet rolling mean für Tore und Siegquote über `window` Spiele pro Team,
    und merged die resultierenden Werte als vier neue Spalten ins ursprüngliche df:
       • avg_goals_home_last10
       • winrate_last10_home
       • avg_goals_away_last10
       • winrate_last10_away
    """
    df = df.sort_values("date")
    recs = []

    # In rec speichern wir pro Spiel zwei Zeilen:
    # 1) Team = home_team, Datum, Tore, Sieg (1/0)
    # 2) Team = away_team, Datum, Tore, Sieg (1/0)
    for _, r in df.iterrows():
        recs.append({
            "team": r.home_team,
            "date": r.date,
            "goals_for": r.home_goals,
            "win": int(r.result == "H")
        })
        recs.append({
            "team": r.away_team,
            "date": r.date,
            "goals_for": r.away_goals,
            "win": int(r.result == "A")
        })

    # DataFrame rec mit allen Einträgen pro Team und Datum
    rec = pd.DataFrame(recs).sort_values(["team", "date"])

    # Rolling-Mittel von 'goals_for' und 'win'
    rec["avg_goals_last10"] = rec.groupby("team")["goals_for"] \
                                 .rolling(window, min_periods=1).mean() \
                                 .reset_index(0, drop=True)
    rec["winrate_last10"]   = rec.groupby("team")["win"] \
                                 .rolling(window, min_periods=1).mean() \
                                 .reset_index(0, drop=True)

    # Merge für Home-Teams
    df = df.merge(
        rec[["team", "date", "avg_goals_last10", "winrate_last10"]]
           .rename(columns={
               "team": "home_team",
               "avg_goals_last10": "avg_goals_home_last10",
               "winrate_last10": "winrate_last10_home"
           }),
        on=["home_team", "date"], how="left"
    )

    # Merge für Away-Teams
    df = df.merge(
        rec[["team", "date", "avg_goals_last10", "winrate_last10"]]
           .rename(columns={
               "team": "away_team",
               "avg_goals_last10": "avg_goals_away_last10",
               "winrate_last10": "winrate_last10_away"
           }),
        on=["away_team", "date"], how="left"
    )

    return df


def build_preprocessor():
    """
    Gibt einen ColumnTransformer zurück, der NUM_FEATS standardisiert (StandardScaler)
    und alle anderen Spalten droppt (remainder='drop').
    """
    return ColumnTransformer(
        [("num", StandardScaler(), NUM_FEATS)],
        remainder="drop"
    )