from flask import Flask, render_template, request, flash
from flask_wtf import FlaskForm
from wtforms import SelectField, IntegerField, SubmitField, FormField, FieldList, Form
from wtforms.validators import DataRequired, NumberRange, ValidationError
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")

# ──────────────────────────────────────────────────────────────────────────────
# Wir registrieren Python's zip als Jinja-Variable, damit Templates zip()
# direkt aufrufen können:
app.jinja_env.globals.update(zip=zip)
# ──────────────────────────────────────────────────────────────────────────────

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

# Unterformular für Punkte je Team
class TableEntryForm(Form):
    points = IntegerField(
        "Punkte",
        validators=[DataRequired(), NumberRange(min=0, max=MAX_POINTS)],
        render_kw={"class": "form-control", "placeholder": "0"},
    )

# Hauptformular
class PredictionForm(FlaskForm):
    spieltag = IntegerField(
        "Spieltag (1 – 34)",
        validators=[DataRequired(), NumberRange(min=1, max=34)],
        render_kw={"class": "form-control", "placeholder": "z. B. 5"},
    )
    home_team = SelectField(
        "Heimteam",
        choices=[(t, t) for t in sorted(TEAMS)],
        validators=[DataRequired()],
        render_kw={"class": "form-select"},
    )
    away_team = SelectField(
        "Auswärtsteam",
        choices=[(t, t) for t in sorted(TEAMS)],
        validators=[DataRequired()],
        render_kw={"class": "form-select"},
    )
    table = FieldList(
        FormField(TableEntryForm),
        min_entries=len(TEAMS),
        max_entries=len(TEAMS),
    )
    submit = SubmitField(
        "Vorhersage berechnen",
        render_kw={"class": "btn btn-primary btn-lg w-100 mt-3"},
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for idx, entry in enumerate(self.table):
            entry.label = TEAMS[idx]

    def validate_away_team(self, field):
        if field.data == self.home_team.data:
            raise ValidationError("Heim- und Auswärtsteam müssen verschieden sein.")

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

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    form = PredictionForm()
    if request.method == "GET":
        for entry in form.table:
            entry.points.data = 0

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

    return render_template("index.html", form=form, teams=TEAMS)

if __name__ == "__main__":
    app.run(debug=True)