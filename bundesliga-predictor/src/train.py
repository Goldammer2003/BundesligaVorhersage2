"""
Training-Pipeline:  Daten einlesen  ➜  Features  ➜  Model Training
Speichert:
    • reports/model_metrics.csv
    • models/best_model.pkl
"""
from __future__ import annotations
import time, math, pickle, warnings
from pathlib import Path
from typing  import List, Dict

import pandas as pd, numpy as np
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, log_loss, brier_score_loss,
                             mean_squared_error)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm      import SVC

# ────────────────────────────────────────────────────────────────
# interne Module
# ────────────────────────────────────────────────────────────────
from src.config        import SEASONS, MODEL_DIR, RAW_DIR, FD_URL
from src.utils         import read_csv_cached, scrape_bulibox_h2h
from src.fbref_ingest  import load_fbref_xg
from src               import features                       # NUM_FEATS etc.

warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)

# ────────────────────────────────────────────────────────────────
# 0) Daten laden  ─ football-data
# ────────────────────────────────────────────────────────────────
def load_fd(seasons: List[int] = SEASONS) -> pd.DataFrame:
    frames = []
    for s in tqdm(seasons, desc="football-data"):
        url  = FD_URL.format(y1=s % 100, y2=(s+1) % 100)
        path = RAW_DIR / f"fd_D1_{s}.csv"
        df   = read_csv_cached(url, path)
        df["Season"] = f"{s}/{str(s+1)[-2:]}"
        frames.append(df)

    df = pd.concat(frames, ignore_index=True, copy=False)

    rename = {"Date":"date","HomeTeam":"home_team","AwayTeam":"away_team",
              "FTHG":"home_goals","FTAG":"away_goals","FTR":"result",
              "B365H":"odds_home","B365D":"odds_draw","B365A":"odds_away"}
    df = df.rename(columns=rename)[list(rename.values()) + ["Season"]]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["date"])

# ────────────────────────────────────────────────────────────────
# 1) Enrichment
# ────────────────────────────────────────────────────────────────
def add_fbref_xg(df: pd.DataFrame) -> pd.DataFrame:
    xg = load_fbref_xg(sorted(df.Season.str[:4].astype(int).unique()))
    if xg.empty:
        df["xg_home"] = df["xg_away"] = np.nan
        return df
    df = df.merge(xg, how="left",
                  left_on=["Season","home_team","away_team","date"],
                  right_on=["season","home_team","away_team","date"])\
           .drop(columns=["season"])
    return df

def add_h2h(df: pd.DataFrame) -> pd.DataFrame:
    rates = [scrape_bulibox_h2h(r.home_team, r.away_team)
             for _, r in tqdm(df.iterrows(), total=len(df), desc="Bulibox H2H")]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rates)], axis=1)

# ────────────────────────────────────────────────────────────────
# 2) Haupt-Routine
# ────────────────────────────────────────────────────────────────
def main() -> None:
    print("✅ train.py gestartet")

    # a) Daten
    df = load_fd()
    df = (df.pipe(add_fbref_xg)
            .pipe(add_h2h)
            .pipe(features.add_implied_prob)   # Achtung: Quoten = Leakage!
            .pipe(features.add_form)
            .pipe(features.add_goal_xg_diff))

    df = df[df["result"].isin(["H","D","A"])].fillna(0)
    print("Datensätze nach FE:", df.shape)

    # b) Split (letzte Saison als Test)
    last_season      = df.Season.iloc[-1]
    train_df, test_df = df[df.Season != last_season], df[df.Season == last_season]

    X_train, y_train = train_df[features.NUM_FEATS], train_df["result"]
    X_test,  y_test  = test_df [features.NUM_FEATS], test_df ["result"]

    # c) Modelle
    pipelines: Dict[str,Pipeline] = {
        "RF":  Pipeline([("clf", RandomForestClassifier(n_estimators=400, random_state=42))]),
        "LR":  Pipeline([("clf", LogisticRegression(max_iter=1500, multi_class="multinomial"))]),
        "SVC": Pipeline([("clf", SVC(C=10, probability=True))]),
    }

    label = {"H":0,"D":1,"A":2}
    results = []
    best_name, best_pipe, best_ll = None, None, np.inf

    for name, pipe in pipelines.items():
        t0      = time.time()
        pipe.fit(X_train, y_train)
        dur     = time.time() - t0
        prob    = pipe.predict_proba(X_test)
        pred    = pipe.predict(X_test)
        acc     = accuracy_score(y_test, pred)
        ll      = log_loss(y_test, prob)
        br      = np.mean([brier_score_loss((y_test==k).astype(int), prob[:,k]) for k in range(3)])
        rmse    = math.sqrt(mean_squared_error(pd.Series(y_test).map(label),
                                               pd.Series(pred).map(label)))
        results.append([name, acc, ll, br, rmse, dur])

        if ll < best_ll:
            best_name, best_pipe, best_ll = name, pipe, ll

        print(f"{name}:  acc={acc:.3f}  logloss={ll:.3f}  rmse={rmse:.3f}")

    # d) Reporting
    rep = pd.DataFrame(results, columns=["model","accuracy","logloss","brier","rmse","sec"])\
            .set_index("model").sort_values("logloss")
    Path("reports").mkdir(exist_ok=True)
    rep.to_csv("reports/model_metrics.csv")
    print("\n🏁  gespeicherte Metriken -> reports/model_metrics.csv")
    print(rep)

    # e) Bestes Modell persistieren
    MODEL_DIR.mkdir(exist_ok=True)
    pickle.dump(best_pipe, open(MODEL_DIR / "best_model.pkl","wb"))
    print(f"🚀  best_model.pkl ({best_name}) gespeichert")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()