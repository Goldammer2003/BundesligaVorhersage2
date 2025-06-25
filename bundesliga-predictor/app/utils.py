import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BEST_MODEL_PATH = Path("models/best_model.pkl")
POISSON_MODEL_PATH = Path("models/poisson_params.pkl")

best_model = joblib.load(BEST_MODEL_PATH)

try:
    poisson_params = joblib.load(POISSON_MODEL_PATH)
except FileNotFoundError:
    poisson_params = None

TEAMS = [
    "FC Bayern München", "Borussia Dortmund", "RB Leipzig", "Bayer 04 Leverkusen",
    "VfB Stuttgart", "Eintracht Frankfurt", "TSG Hoffenheim", "SC Freiburg",
    "1. FC Union Berlin", "Borussia Mönchengladbach", "VfL Wolfsburg",
    "1. FSV Mainz 05", "FC Augsburg", "Werder Bremen", "1. FC Heidenheim",
    "VfL Bochum", "Fortuna Düsseldorf", "Hamburger SV",
]

MAX_POINTS = 34 * 3
POINTS = {"H": 3, "D": 1, "A": 0}

def expected_goals(home: str, away: str):
    if poisson_params is None:
        return None, None
    ah, dh = poisson_params["attack"][home], poisson_params["defense"][home]
    aa, da = poisson_params["attack"][away], poisson_params["defense"][away]
    return ah * da, aa * dh

def generate_round_robin_fixtures(teams: list[str], total_matchdays: int = 34) -> pd.DataFrame:
    fixtures = []
    n_teams = len(teams)
    matchday = 1
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            fixtures.append({"matchday": matchday, "home_team": teams[i], "away_team": teams[j]})
            matchday = matchday + 1 if matchday < total_matchdays else 1
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            fixtures.append({"matchday": matchday, "home_team": teams[j], "away_team": teams[i]})
            matchday = matchday + 1 if matchday < total_matchdays else 1
    return pd.DataFrame(fixtures)

def simulate_championship(start_matchday, fixtures, model, features, current_points, n_sim=1500):
    outcomes = np.array(["H", "D", "A"])
    prob_mat = model.predict_proba(fixtures[features])
    placements = {team: np.zeros(len(TEAMS)) for team in TEAMS}

    for _ in range(n_sim):
        standings = current_points.copy()
        rnd_results = outcomes[(np.random.rand(len(fixtures), 1) < prob_mat.cumsum(axis=1)).argmax(1)]
        for idx, res in enumerate(rnd_results):
            home, away = fixtures.iloc[idx]["home_team"], fixtures.iloc[idx]["away_team"]
            standings[home] += POINTS[res]
            standings[away] += POINTS[{"H": "A", "A": "H", "D": "D"}[res]]

        sorted_teams = sorted(standings.items(), key=lambda x: x[1], reverse=True)
        for place, (team, _) in enumerate(sorted_teams):
            placements[team][place] += 1

    return {team: (placements[team] / n_sim).tolist() for team in TEAMS}