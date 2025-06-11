import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tqdm import tqdm

POINTS = {"H": 3, "D": 1, "A": 0}

def champion_probs_from_current_table(fixtures, model, features, current_points, n_iter=10000):
    clubs = list(current_points.keys())
    champs = dict.fromkeys(clubs, 0)
    prob_mat = model.predict_proba(fixtures[features])
    outcomes = np.array(["H", "D", "A"])

    for _ in tqdm(range(n_iter), desc="Simulating Meister"):
        standings = current_points.copy()
        rnd = outcomes[(np.random.rand(len(fixtures), 1) < prob_mat.cumsum(axis=1)).argmax(1)]
        for idx, res in enumerate(rnd):
            home, away = fixtures.iloc[idx]["home_team"], fixtures.iloc[idx]["away_team"]
            standings[home] += POINTS[res]
            standings[away] += POINTS[{"H":"A","A":"H","D":"D"}[res]]

        champ = max(standings, key=standings.get)
        champs[champ] += 1

    return {k: v / n_iter for k, v in champs.items()}