# src/simulate.py
"""
Monte-Carlo-Simulation der Meisterwahrscheinlichkeit
– entweder ab Saisonstart (champion_probs) oder
– ab einem fiktiven Tabellenstand (champion_probs_from_current_table).
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline

POINTS = {"H": 3, "D": 1, "A": 0}

def champion_probs(fixtures: pd.DataFrame, model: Pipeline, features: list,
                   n_iter: int = 10_000) -> dict:
    """
    Simulation ab Saisonstart:
    - fixtures: DataFrame, enthält Spieldaten (home_team, away_team, alle Input-Features).
    - model: trainiertes ML-Modell (Pipeline).
    - features: Liste der Feature-Spalten (z. B. NUM_FEATS).
    - n_iter: Anzahl der Monte-Carlo-Durchläufe.
    Rückgabe: Dict {team: Meisterwahrscheinlichkeit}.
    """
    clubs = pd.unique(fixtures[["home_team", "away_team"]].values.ravel())
    champs = dict.fromkeys(clubs, 0)

    X = fixtures[features]
    prob_mat = model.predict_proba(X)
    outcomes = np.array(["H", "D", "A"])

    for _ in tqdm(range(n_iter), desc="Simulations"):
        standings = dict.fromkeys(clubs, 0)
        rnd = outcomes[(np.random.rand(len(fixtures), 1) < prob_mat.cumsum(axis=1)).argmax(1)]
        for (idx, row), res in zip(fixtures.iterrows(), rnd):
            standings[row.home_team] += POINTS[res]
            standings[row.away_team] += POINTS[{"H":"A","A":"H","D":"D"}[res]]
        champ = max(standings, key=standings.get)
        champs[champ] += 1
    return {k: v / n_iter for k, v in champs.items()}


def champion_probs_from_current_table(remaining_fixtures: pd.DataFrame,
                                      model: Pipeline, features: list,
                                      current_points: dict,
                                      n_iter: int = 10_000) -> dict:
    """
    Simulation ab fiktivem Tabellenstand:
    - remaining_fixtures: DataFrame der noch zu spielenden Spiele (home_team, away_team, ...).
    - model: trainiertes ML-Modell.
    - features: Liste der Feature-Spalten (NUM_FEATS).
    - current_points: Dict {team: schon erreichte Punkte}.
    - n_iter: Anzahl der Durchläufe.

    Vorgehen:
      • Wir starten jedes Team mit current_points[team].
      • Für jedes Spiel in remaining_fixtures wählen wir anhand model.predict_proba
        zufällig ein Ergebnis (H/D/A).
      • Punkte werden addiert. Am Ende zählt, wer in jedem Simulationslauf vorne liegt.
    """
    clubs = list(current_points.keys())
    champs = dict.fromkeys(clubs, 0)
    X_rem = remaining_fixtures[features]
    prob_mat_rem = model.predict_proba(X_rem)
    outcomes = np.array(["H", "D", "A"])

    for _ in tqdm(range(n_iter), desc="Simulations (ab fiktivem Stand)"):
        # Starte mit dem fiktiven Punktebudget
        standings = current_points.copy()
        rnd = outcomes[(np.random.rand(len(remaining_fixtures), 1) < prob_mat_rem.cumsum(axis=1)).argmax(1)]
        for (idx, row), res in zip(remaining_fixtures.iterrows(), rnd):
            standings[row.home_team] += POINTS[res]
            standings[row.away_team] += POINTS[{"H":"A","A":"H","D":"D"}[res]]
        champ = max(standings, key=standings.get)
        champs[champ] += 1

    return {k: v / n_iter for k, v in champs.items()}