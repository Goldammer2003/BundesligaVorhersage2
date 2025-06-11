from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import pickle
from pathlib import Path

from src.config import SEASONS
from src.ingest import fetch_fixtures, load_fd, add_fbref_xg, add_h2h
from src.features import add_form, add_goal_xg_diff, add_rolling_stats, NUM_FEATS
from src.simulate import champion_probs_from_current_table

app = Flask(__name__)
app.secret_key = "supersecret123"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/simulate", methods=["POST"])
def simulate():
    spieltag = int(request.form["spieltag"])
    
    current_points = {}
    for team in request.form.getlist("team[]"):
        pts = request.form.get(f"points_{team}", type=int)
        if pts is None:
            flash(f"Punkte fehlen für {team}", "error")
            return redirect(url_for("index"))
        current_points[team] = pts

    model_path = Path("models/best_model.pkl")
    if not model_path.exists():
        flash("Kein trainiertes Modell gefunden.", "error")
        return redirect(url_for("index"))
    
    model = pickle.load(open(model_path, "rb"))

    fixtures = fetch_fixtures(SEASONS[-1])
    remaining = fixtures[fixtures["matchday"] > spieltag].copy()

    df_all = (load_fd(SEASONS[:-1] + [SEASONS[-1]])
              .pipe(add_fbref_xg)
              .pipe(add_h2h)
              .pipe(add_form)
              .pipe(add_goal_xg_diff)
              .pipe(add_rolling_stats, window=10)
              .fillna(0))

    remaining_feat = remaining.merge(df_all, how="left", on=["home_team", "away_team", "date"]).fillna(0)

    results = champion_probs_from_current_table(
        remaining_feat, model, NUM_FEATS, current_points, n_iter=10000
    )

    return render_template("result.html", spieltag=spieltag, results=results)

if __name__ == "__main__":
    app.run(debug=True)