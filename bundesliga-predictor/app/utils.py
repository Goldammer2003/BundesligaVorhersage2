"""
Utility-Modul der Flask-App
---------------------------
• hält globale Konstanten (TEAMS, MAX_POINTS, POINTS)
• lädt das trainierte Modell & optionale Poisson-Parameter
• stellt Hilfsfunktionen bereit:
     - expected_goals
     - generate_round_robin_fixtures
     - simulate_championship (vektorisiert, schnell)
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# ────────────────────────── Konstanten ──────────────────────────
BEST_MODEL_PATH    = Path("models/best_model.pkl")
POISSON_MODEL_PATH = Path("models/poisson_params.pkl")

TEAMS = [
    "FC Bayern München", "Borussia Dortmund", "RB Leipzig", "Bayer 04 Leverkusen",
    "VfB Stuttgart", "Eintracht Frankfurt", "TSG Hoffenheim", "SC Freiburg",
    "1. FC Union Berlin", "Borussia Mönchengladbach", "VfL Wolfsburg",
    "1. FSV Mainz 05", "FC Augsburg", "Werder Bremen", "1. FC Heidenheim",
    "VfL Bochum", "Fortuna Düsseldorf", "Hamburger SV",
]

MAX_POINTS = 34 * 3                # theoretisches Maximum einer Saison
POINTS = {"H": 3, "D": 1, "A": 0}  # Punktvergabe nach Ergebnis

# ───────────────────── Modell & Poisson-Parameter ───────────────
best_model = joblib.load(BEST_MODEL_PATH)

try:
    poisson_params = joblib.load(POISSON_MODEL_PATH)
except FileNotFoundError:
    poisson_params = None   # xG-Berechnung dann deaktiviert

# ───────────────────────── Hilfsfunktionen ──────────────────────
def expected_goals(home: str, away: str):
    """Gibt erwartete Tore (λ_home, λ_away) zurück, falls Poisson-Parameter vorhanden."""
    if poisson_params is None:
        return None, None
    ah, dh = poisson_params["attack"][home],  poisson_params["defense"][home]
    aa, da = poisson_params["attack"][away],  poisson_params["defense"][away]
    return ah * da, aa * dh


def generate_round_robin_fixtures(teams: list[str], total_matchdays: int = 34) -> pd.DataFrame:
    """Erstellt einen Hin- und Rückrunden-Spielplan als DataFrame."""
    fixtures, matchday = [], 1
    n = len(teams)
    for i in range(n):
        for j in range(i + 1, n):
            fixtures.append({"matchday": matchday, "home_team": teams[i], "away_team": teams[j]})
            matchday = matchday + 1 if matchday < total_matchdays else 1
    for i in range(n):
        for j in range(i + 1, n):
            fixtures.append({"matchday": matchday, "home_team": teams[j], "away_team": teams[i]})
            matchday = matchday + 1 if matchday < total_matchdays else 1
    return pd.DataFrame(fixtures)


def simulate_championship(
    start_matchday: int,
    fixtures: pd.DataFrame,
    model,
    features: list[str],
    current_points: dict[str, int],
    n_sim: int = 800,        # genügt für stabile Resultate, <10 s Laufzeit
) -> dict[str, list[float]]:
    """
    Vektorisierte Saisonsimulation. Liefert pro Team eine Liste mit
    Wahrscheinlichkeiten, auf Rang 1 … 18 zu landen.
    """
    team_to_idx = {t: i for i, t in enumerate(TEAMS)}
    n_teams     = len(TEAMS)

    home_idx = fixtures["home_team"].map(team_to_idx).to_numpy()
    away_idx = fixtures["away_team"].map(team_to_idx).to_numpy()

    prob_mat = model.predict_proba(fixtures[features])        # Form (m,3)
    cum_prob = prob_mat.cumsum(axis=1)

    base_points = np.array([current_points[t] for t in TEAMS])
    placements  = np.zeros((n_teams, n_teams), dtype=np.int32)

    rng = np.random.default_rng()

    for _ in range(n_sim):
        rnd      = rng.random(len(fixtures))
        res_idx  = (rnd[:, None] < cum_prob).argmax(axis=1)   # 0=H,1=D,2=A
        pts      = base_points.copy()

        pts[home_idx] += np.where(res_idx == 0, 3, np.where(res_idx == 1, 1, 0))
        pts[away_idx] += np.where(res_idx == 2, 3, np.where(res_idx == 1, 1, 0))

        ranking = np.argsort(-pts)                            # absteigend sortiert
        placements[ranking, np.arange(n_teams)] += 1

    return {t: (placements[i] / n_sim).tolist() for i, t in enumerate(TEAMS)}