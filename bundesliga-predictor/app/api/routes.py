from . import predict_api_blueprint
from flask import render_template, request, flash
from ..forms import PredictionForm
from ..utils import TEAMS, best_model, simulate_championship, expected_goals
import pandas as pd
import numpy as np

# API Routes
@predict_api_blueprint.route("/", methods=["GET", "POST"])
def index():
    form = PredictionForm()

    if request.method == "POST":

        if form.validate_on_submit():
            spieltag = form.spieltag.data
            home = form.home_team.data
            away = form.away_team.data

            points = {TEAMS[i]: form.table[i].points.data for i in range(len(TEAMS))}

            # Feature-DataFrame bauen
            X = pd.DataFrame({
                "home_team": [home],
                "away_team": [away],
                "matchday": [spieltag],
            })

            # Winrates last10
            def calc_winrate(team):
                played = min(spieltag, 10)
                pts = points[team]
                return min(pts / (played * 3), 1)

            X["winrate_last10_home"] = calc_winrate(home)
            X["winrate_last10_away"] = calc_winrate(away)

            # Prediction und Missing-Columns-Fallback
            try:
                proba = best_model.predict_proba(X)[0]
            except ValueError as e:
                msg = str(e)
                if "columns are missing" in msg:
                    miss = eval(msg.split("columns are missing:")[1])
                    for c in miss:
                        X[c] = 0
                    proba = best_model.predict_proba(X)[0]
                else:
                    raise
            result_proba = dict(zip(best_model.classes_, proba))

            lambda_h, lambda_a = expected_goals(home, away)
            champ_probs = simulate_championship(spieltag)

            return render_template(
                "predict.html",
                spieltag=spieltag,
                home_team=home,
                away_team=away,
                result_proba=result_proba,
                lambda_home=lambda_h,
                lambda_away=lambda_a,
                champ_probs=champ_probs,
            )
    
    for entry in form.table:
        entry.points.data = 0

    return render_template("index.html", form=form, teams=TEAMS)