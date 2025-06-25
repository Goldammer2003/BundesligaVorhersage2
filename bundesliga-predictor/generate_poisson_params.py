# generate_poisson_params.py
import joblib
from pathlib import Path
import numpy as np

TEAMS = [
    "FC Bayern München", "Borussia Dortmund", "RB Leipzig", "Bayer 04 Leverkusen",
    "VfB Stuttgart", "Eintracht Frankfurt", "TSG Hoffenheim", "SC Freiburg",
    "1. FC Union Berlin", "Borussia Mönchengladbach", "VfL Wolfsburg",
    "1. FSV Mainz 05", "FC Augsburg", "Werder Bremen", "1. FC Heidenheim",
    "VfL Bochum", "Fortuna Düsseldorf", "Hamburger SV",
]

poisson_params = {
    "attack": {team: np.random.uniform(0.8, 1.5) for team in TEAMS},
    "defense": {team: np.random.uniform(0.8, 1.5) for team in TEAMS},
}

Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(poisson_params, "models/poisson_params.pkl")

print("✅ poisson_params.pkl generiert.")