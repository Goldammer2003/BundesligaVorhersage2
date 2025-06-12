import numpy as np
import joblib
import os

# Modelle laden
BEST_MODEL_PATH = "models/best_model.pkl"
POISSON_MODEL_PATH = "models/poisson_params.pkl"

best_model = joblib.load(BEST_MODEL_PATH)
try:
    poisson_params = joblib.load(POISSON_MODEL_PATH)
except FileNotFoundError:
    poisson_params = None

# Erstligisten & Max-Punkte
TEAMS = [
    "FC Bayern München", "Borussia Dortmund", "RB Leipzig", "Bayer 04 Leverkusen",
    "VfB Stuttgart", "Eintracht Frankfurt", "TSG Hoffenheim", "SC Freiburg",
    "1. FC Union Berlin", "Borussia Mönchengladbach", "VfL Wolfsburg",
    "1. FSV Mainz 05", "FC Augsburg", "Werder Bremen", "1. FC Heidenheim",
    "VfL Bochum", "Fortuna Düsseldorf", "Hamburger SV",
]
MAX_POINTS = 34 * 3

# Hilfsfunktionen
def expected_goals(home: str, away: str):
    if poisson_params is None:
        return None, None
    ah = poisson_params["attack"][home]
    dh = poisson_params["defense"][home]
    aa = poisson_params["attack"][away]
    da = poisson_params["defense"][away]
    return ah * da, aa * dh

def simulate_championship(start_matchday: int, n_sim: int = 10000) -> dict:
    # TODO: echte Monte-Carlo-Logik implementieren
    return {}