# ──────────────────────────────────────────────────────────────
# src/app_streamlit.py
# Streamlit-Dashboard: Match- und Meister-Prognosen
# ──────────────────────────────────────────────────────────────
import os, sys, pickle
from datetime import datetime
import pandas as pd
import streamlit as st

# Projekt-Module erreichbar machen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import ingest as ing
from src import features as ft
from src import simulate as sim
from src.config import SEASONS, MODEL_DIR
from src.features import NUM_FEATS

st.set_page_config(page_title="Bundesliga Predictor",
                   page_icon="⚽️",
                   layout="wide")
st.title("⚽️ Bundesliga – Match- & Meister-Predictor")

# ──────────────────────────────────────────────────────────────
# 1) Daten laden  (Session-Cache)
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_hist(use_h2h=False, use_fbref_xg=True):
    df = ing.load_fd(SEASONS)
    if use_fbref_xg:
        df = ing.add_fbref_xg(df)
    if use_h2h:
        df = ing.add_h2h(df)
    else:
        df["h2h_home_winrate"] = 0.5

    df = (df.pipe(ft.add_implied_prob)
             .pipe(ft.add_form)
             .pipe(ft.add_goal_xg_diff))

    df = df[df["result"].isin(["H", "D", "A"])].fillna(0)
    return df

# Seitenleiste – Optionen
with st.sidebar:
    st.header("⚙️ Optionen")
    use_h2h  = st.checkbox("📈 Head-to-Head verwenden (Bulibox)", value=False)
    use_xg   = st.checkbox("🎯 FBref-xG verwenden", value=True)

hist = load_hist(use_h2h, use_xg)

# ──────────────────────────────────────────────────────────────
# 2) Modell laden
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    mdl_path = MODEL_DIR / "best_model.pkl"
    if not mdl_path.exists():
        st.error("Kein trainiertes Modell gefunden. Bitte zuerst `python -m src.train` ausführen.")
        st.stop()
    return pickle.load(open(mdl_path, "rb"))

model = load_model()

# ──────────────────────────────────────────────────────────────
# 3) Kommende Spiele & Match-Prognosen
# ──────────────────────────────────────────────────────────────
this_season = datetime.now().year if datetime.now().month >= 7 else datetime.now().year - 1
fixtures = ing.fetch_fixtures(this_season)

st.subheader("🔮 Match-Wahrscheinlichkeiten")
max_md = int(fixtures.matchday.max())
sel_md = st.slider("Spieltag wählen", 1, max_md, max_md)
upcoming = fixtures[fixtures.matchday == sel_md].copy()

# Dummy-Features (live-Odds/Form fehlen i. d. R.)
for s in ("home", "draw", "away"):
    upcoming[f"imp_{s}"] = 1/3
upcoming["form_last5"] = 1.3
upcoming["goal_diff"]  = 0
upcoming["xg_diff"]    = 0
upcoming["h2h_home_winrate"] = 0.5 if not use_h2h else upcoming["h2h_home_winrate"]

# Prognose
probs = model.predict_proba(upcoming[NUM_FEATS])
pred_df = pd.DataFrame(probs, columns=["Heimsieg", "Remis", "Auswärtssieg"])
pred_df.insert(0, "Heim", upcoming.home_team.values)
pred_df.insert(1, "Auswärts", upcoming.away_team.values)

st.dataframe(pred_df.style.format({col:"{:.1%}" for col in ["Heimsieg","Remis","Auswärtssieg"]}))

# ──────────────────────────────────────────────────────────────
# 4) Meister-Simulation
# ──────────────────────────────────────────────────────────────
st.subheader("🏆 Meister-Wahrscheinlichkeiten (Monte Carlo, 10 000)")
champ = sim.champion_probs(upcoming, model, NUM_FEATS, 10_000)
champ_df = (pd.DataFrame(champ.items(), columns=["Team","Chance"])
              .sort_values("Chance", ascending=False))
st.bar_chart(champ_df.set_index("Team"))

st.caption("© 2025 – Datenquellen: football-data.co.uk · FBref/StatsBomb · OpenLigaDB · Bulibox")