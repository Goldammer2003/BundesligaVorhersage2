import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Modelle & Parameter laden
# ---------------------------------------------------------------------
BEST_MODEL_PATH    = Path("models/best_model.pkl")
POISSON_MODEL_PATH = Path("models/poisson_params.pkl")

best_model = joblib.load(BEST_MODEL_PATH)

try:
    poisson_params = joblib.load(POISSON_MODEL_PATH)
except FileNotFoundError:
    poisson_params = None  # xG-Berechnung fällt dann sauber zurück

# ---------------------------------------------------------------------
# 2. Stammdaten
# ---------------------------------------------------------------------
TEAMS = [
    "FC Bayern München", "Borussia Dortmund", "RB Leipzig", "Bayer 04 Leverkusen",
    "VfB Stuttgart", "Eintracht Frankfurt", "TSG Hoffenheim", "SC Freiburg",
    "1. FC Union Berlin", "Borussia Mönchengladbach", "VfL Wolfsburg",
    "1. FSV Mainz 05", "FC Augsburg", "Werder Bremen", "1. FC Heidenheim",
    "VfL Bochum", "Fortuna Düsseldorf", "Hamburger SV",
]
MAX_POINTS = 34 * 3
POINTS     = {"H": 3, "D": 1, "A": 0}  # für die Simulation

# ---------------------------------------------------------------------
# 3. Hilfsfunktionen
# ---------------------------------------------------------------------
def expected_goals(home: str, away: str):
    """
    Liefert erwartete Tore (λ-Werte) auf Basis gespeicherter Poisson-Parameter.
    """
    if poisson_params is None:
        return None, None
    ah, dh = poisson_params["attack"][home],  poisson_params["defense"][home]
    aa, da = poisson_params["attack"][away], poisson_params["defense"][away]
    return ah * da, aa * dh

def generate_round_robin_fixtures(teams: list[str], total_matchdays: int = 34) -> pd.DataFrame:
    """
    Erstellt einen vollständigen Round-Robin-Spielplan (Hin- und Rückrunde)
    mit 18 Teams über 34 Spieltage.
    """
    fixtures = []
    n_teams = len(teams)
    matchday = 1

    # Einfachrunde: Jeder gegen jeden einmal
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            fixtures.append({
                "matchday": matchday,
                "home_team": teams[i],
                "away_team": teams[j]
            })
            matchday = matchday + 1 if matchday < total_matchdays else 1

    # Rückrunde: Seiten getauscht
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            fixtures.append({
                "matchday": matchday,
                "home_team": teams[j],
                "away_team": teams[i]
            })
            matchday = matchday + 1 if matchday < total_matchdays else 1

    return pd.DataFrame(fixtures)

def simulate_championship(start_matchday: int,
                          fixtures: pd.DataFrame,
                          model,
                          features: list,
                          current_points: dict,
                          n_sim: int = 2000) -> dict:
    """
    Monte-Carlo-Simulation aller noch offenen Spiele → Meisterschafts-Wahrscheinlichkeiten
    """
    outcomes = np.array(["H", "D", "A"])
    prob_mat = model.predict_proba(fixtures[features])
    champs   = dict.fromkeys(current_points.keys(), 0)

    for _ in range(n_sim):
        standings = current_points.copy()

        rnd_results = outcomes[
            (np.random.rand(len(fixtures), 1) < prob_mat.cumsum(axis=1)).argmax(1)
        ]

        for idx, res in enumerate(rnd_results):
            home = fixtures.iloc[idx]["home_team"]
            away = fixtures.iloc[idx]["away_team"]
            standings[home] += POINTS[res]
            standings[away] += POINTS[{"H": "A", "A": "H", "D": "D"}[res]]

        champion = max(standings, key=standings.get)
        champs[champion] += 1

    return {k: v / n_sim for k, v in champs.items()}