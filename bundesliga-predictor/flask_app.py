# flask_app.py

from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import pickle
from pathlib import Path

from src.config import SEASONS
from src.ingest import fetch_fixtures, load_fd, add_fbref_xg, add_h2h
from src.features import add_form, add_goal_xg_diff, add_rolling_stats, NUM_FEATS
from src.simulate import champion_probs_from_current_table

app = Flask(__name__)
app.secret_key = "dein_geheimer_schluessel"  # Zum Beispiel „supersecret123“ – nur für Flash-Meldungen

# ----------------------------------------------------
# Route: Startseite mit Formular
# ----------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ----------------------------------------------------
# Route: Verarbeitung + Simulation
# ----------------------------------------------------
@app.route("/simulate", methods=["POST"])
def simulate():
    # 1) Spieltag validieren
    spieltag_raw = request.form.get("spieltag", "").strip()
    try:
        spieltag = int(spieltag_raw)
        if not (1 <= spieltag <= 34):
            raise ValueError("Spieltag muss zwischen 1 und 34 liegen.")
    except Exception as e:
        flash(f"Ungültiger Spieltag: {e}", "error")
        return redirect(url_for("index"))

    # 2) CSV-Datei einlesen
    file = request.files.get("csv_stand")
    if not file or file.filename == "":
        flash("Bitte eine CSV-Datei mit 'team' und 'points' hochladen.", "error")
        return redirect(url_for("index"))

    try:
        df_stand = pd.read_csv(file)
        if not {"team", "points"}.issubset(df_stand.columns):
            raise ValueError("CSV-Datei muss exakt die Spalten 'team' und 'points' enthalten.")
        # Punkte in ein Dict umwandeln
        current_points = dict(zip(df_stand["team"], df_stand["points"]))
    except Exception as e:
        flash(f"Fehler beim Einlesen der CSV: {e}", "error")
        return redirect(url_for("index"))

    # 3) Trainiertes Modell laden
    model_path = Path("models/best_model.pkl")
    if not model_path.exists():
        flash("Kein trainiertes Modell gefunden. Bitte zuerst 'python -m src.train' ausführen.", "error")
        return redirect(url_for("index"))
    model = pickle.load(open(model_path, "rb"))

    # 4) Verbleibende Fixtures laden
    current_season = SEASONS[-1]  # z.B. 2024
    fixtures = fetch_fixtures(current_season)  # DataFrame mit ['match_id','date','home_team','away_team','matchday']
    remaining = fixtures[fixtures["matchday"] > spieltag].copy()

    if remaining.empty:
        flash(f"Ab Spieltag {spieltag + 1} gibt es keine verbleibenden Spiele.", "error")
        return redirect(url_for("index"))

    # 5) Features für „remaining“ Partien berechnen
    try:
        # 5a) Alle Spiele der aktuellen Saison laden und bereiten
        df_all = (
            load_fd(SEASONS[:-1] + [current_season])  # Alle Saisons + aktuelle
            .pipe(add_fbref_xg)
            .pipe(add_h2h)
        )

        # 5b) Feature Engineering identisch zu train.py:
        df_all = (
            df_all
            .pipe(add_form)
            .pipe(add_goal_xg_diff)
            .pipe(add_rolling_stats, window=10)
        )
        df_all = df_all[df_all.result.isin(["H", "D", "A"])].fillna(0)

        # 5c) Wir nehmen die bereits absolvierten Spiele bis „spieltag“, um darauf aufbauend
        #     die Feature-Werte für Teams Weiterlesen zu können.
        #     In einer echten Version würde man Spieltag <→> Datum exakt mappen. 
        #     Hier approximieren wir: „erste spieltag * X Spiele“ aus df_all.
        #     (Das könnt ihr gegen echte Spieltag-Daten austauschen, wenn ihr sie habt.)
        played_all_season = df_all[df_all["Season"] == f"{current_season}/{str(current_season+1)[-2:]}"]
        # Approximation: In Bundesliga 18 Teams → 9 Spiele pro Spieltag. 
        # Ersten 'spieltag' × 9 Datensätze in 'played_subset' nehmen.
        played_subset = played_all_season.sort_values("date").iloc[: spieltag * 9]

        # 5d) „remaining“ Partien mit exakt denselben Spalten wie im Training versehen:
        #     Wir mergen home-team-spezifische Rolling-Stat-Werte und
        #     away-team-spezifische Rolling-Stat-Werte aus played_subset
        rem = remaining.copy()
        rem = rem.merge(
            played_subset[["home_team", "date", "avg_goals_home_last10", "winrate_last10_home"]],
            how="left", on=["home_team", "date"]
        ).merge(
            played_subset[["away_team", "date", "avg_goals_away_last10", "winrate_last10_away"]],
            how="left", on=["away_team", "date"]
        ).fillna(0)

        # 5e) Abschließend dieselben Pipes add_form, add_goal_xg_diff, add_rolling_stats
        #     anwenden, damit alle NUM_FEATS-Spalten vorhanden sind.
        rem = (
            rem
            .pipe(add_form)
            .pipe(add_goal_xg_diff)
            .pipe(add_rolling_stats, window=10)
        )
        remaining_feat = rem[["home_team", "away_team"] + NUM_FEATS]
    except Exception as e:
        flash(f"Fehler bei der Feature-Berechnung: {e}", "error")
        return redirect(url_for("index"))

    # 6) Meisterschafts-Simulation (10.000 Iterationen)
    results = champion_probs_from_current_table(
        remaining_feat,
        model,
        NUM_FEATS,
        current_points,
        n_iter=10_000
    )

    # 7) Result-Page anzeigen
    return render_template("result.html", spieltag=spieltag, results=results)


# ----------------------------------------------------
# Hauptprogramm (Debug-Modus)
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)