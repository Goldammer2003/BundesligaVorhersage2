import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
from datetime import datetime

import pandas as pd
import streamlit as st

from src import ingest as ing
from src import features as ft
from src import models as mdl
from src import simulate as sim
from src.config import SEASONS, MODEL_DIR
from src.features import NUM_FEATS

st.set_page_config(page_title="Bundesliga Predictor", layout="wide")
st.title("⚽️ Bundesliga Match- & Meister-Predictor")

# ----------------------------------------------------------------------
# 1) Daten laden / vorbereiten (nur 1× pro Session)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_prepare():
    df = ing.load_fd(SEASONS)
    df = ft.add_implied_prob(df)
    df = ft.add_form(df)
    df = ft.add_goal_xg_diff(df)
    return df

hist = load_and_prepare()
last_season = hist["Season"].iloc[-1]
train_df = hist[hist.Season != last_season]
test_df  = hist[hist.Season == last_season]

# ----------------------------------------------------------------------
# 2) Modelle trainieren / laden
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_best_model():
    trained = mdl.train(train_df)
    board   = mdl.evaluate(trained, test_df)
    best    = trained[board.index[0]]
    path    = MODEL_DIR / f"best_{board.index[0]}.pkl"
    pickle.dump(best, open(path, "wb"))
    return best, board

best_model, leaderboard = get_best_model()

st.subheader("📊 Modell-Leaderboard (Hold-out Saison)")
st.dataframe(leaderboard.style.format({"accuracy":"{:.3f}",
                                       "logloss":"{:.3f}",
                                       "brier":"{:.3f}"}))

# ----------------------------------------------------------------------
# 3) Live-Fixtures laden (OpenLigaDB) & Vorhersagen
# ----------------------------------------------------------------------
this_season = datetime.now().year if datetime.now().month >= 7 else datetime.now().year - 1
fixtures = ing.fetch_fixtures(this_season)

next_md = int(st.slider("Nächster Spieltag", 1, int(fixtures.matchday.max()), 1))
upcoming = fixtures[fixtures.matchday == next_md].copy()

# Dummy-Features (wenn Odds/Form live nicht verfügbar)
for s in ("home", "draw", "away"):
    upcoming[f"imp_{s}"] = 1/3
upcoming["form_last5"] = 1.3
upcoming["goal_diff"]  = 0
upcoming["xg_diff"]    = 0
upcoming["h2h_home_winrate"] = 0.5

probs = best_model.predict_proba(upcoming[NUM_FEATS])
tmp   = upcoming[["home_team", "away_team"]].copy()
tmp[["Heimsieg", "Remis", "Auswärtssieg"]] = probs
st.subheader("🔮 Sieg-/Remis-Wahrscheinlichkeiten")
st.dataframe(tmp.style.format({"Heimsieg":"{:.1%}",
                               "Remis":"{:.1%}",
                               "Auswärtssieg":"{:.1%}"}))

# ----------------------------------------------------------------------
# 4) Meister-Simulation
# ----------------------------------------------------------------------
st.subheader("🏆 Meister-Wahrscheinlichkeiten (Monte-Carlo, 10 000)")
champ = sim.champion_probs(upcoming, best_model, NUM_FEATS, 10_000)
champ_df = pd.DataFrame({"Team": champ.keys(), "Chance": champ.values()})\
             .sort_values("Chance", ascending=False)
st.bar_chart(champ_df.set_index("Team"))
st.caption("© 2025 – Daten: football-data.co.uk · OpenLigaDB · Understat · Bulibox")