from . import predict_api_blueprint
from flask import render_template, request, flash
from ..forms import PredictionForm
from ..utils import (
    TEAMS, best_model, simulate_championship,
    expected_goals, generate_round_robin_fixtures
)
import pandas as pd
import numpy as np

POINTS = {"H": 3, "D": 1, "A": 0}

@predict_api_blueprint.route("/", methods=["GET", "POST"])
def index():
    form = PredictionForm()

    if request.method == "POST":
        if form.validate_on_submit():
            spieltag = form.spieltag.data
            home = form.home_team.data
            away = form.away_team.data

            points = {TEAMS[i]: form.table[i].points.data for i in range(len(TEAMS))}

            # Spielvorhersage (einzelnes Spiel)
            X = pd.DataFrame({
                "home_team": [home],
                "away_team": [away],
                "matchday": [spieltag],
            })

            def calc_winrate(team):
                played = min(spieltag, 10)
                return 0.0 if played == 0 else min(points[team] / (played * 3), 1)

            X["winrate_last10_home"] = calc_winrate(home)
            X["winrate_last10_away"] = calc_winrate(away)

            for feat in best_model.feature_names_in_:
                if feat not in X.columns:
                    X[feat] = 0

            proba = best_model.predict_proba(X[best_model.feature_names_in_])[0]
            result_proba = dict(zip(best_model.classes_, proba))

            # Expected Goals
            lambda_h, lambda_a = expected_goals(home, away)

            # Spielplan bis Saisonende generieren
            full_schedule = generate_round_robin_fixtures(TEAMS)
            remaining = full_schedule[full_schedule["matchday"] > spieltag].copy()

            remaining["winrate_last10_home"] = remaining["home_team"].apply(calc_winrate)
            remaining["winrate_last10_away"] = remaining["away_team"].apply(calc_winrate)

            for feat in best_model.feature_names_in_:
                if feat not in remaining.columns:
                    remaining[feat] = 0

            fixtures_df = remaining[["home_team", "away_team", *best_model.feature_names_in_]]

            # Meisterschafts-Simulation mit Platzierungen
            placement_probs = simulate_championship(
                start_matchday=spieltag,
                fixtures=fixtures_df,
                model=best_model,
                features=best_model.feature_names_in_,
                current_points=points,
                n_sim=1500  # Kompromiss: realistisch + schnell
            )

            return render_template(
                "predict.html",
                spieltag=spieltag,
                home_team=home,
                away_team=away,
                result_proba=result_proba,
                lambda_home=lambda_h,
                lambda_away=lambda_a,
                placement_probs=placement_probs,
            )

    for entry in form.table:
        entry.points.data = 0

    return render_template("index.html", form=form, teams=TEAMS)