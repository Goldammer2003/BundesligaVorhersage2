"""
Monte-Carlo-Simulation der Meisterwahrscheinlichkeit
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline

POINTS = {"H": 3, "D": 1, "A": 0}

def champion_probs(fixtures: pd.DataFrame, model: Pipeline, features: list,
                   n_iter: int = 10_000) -> dict:
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

print ("Simulation gestartet")