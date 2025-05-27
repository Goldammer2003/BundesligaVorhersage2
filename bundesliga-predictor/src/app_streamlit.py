"""
Training-Pipeline: Daten laden ➜ Enrichment ➜ Features ➜ Modelle trainieren
Speichert:
  • reports/model_metrics.csv
  • models/best_model.pkl
"""
import time, math, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, brier_score_loss

from src.config       import SEASONS, MODEL_DIR
from src.ingest       import load_fd, add_h2h, add_fbref_xg
import src.features   as features  # NUM_FEATS, add_*, build_preprocessor

warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)

def main():
    print("✅ train.py gestartet")
    # ─── 0) Daten laden + Anreichern ─────────────────────────
    df = load_fd(SEASONS)
    df = add_fbref_xg(df)
    df = add_h2h(df)

    # ─── 1) Feature Engineering ──────────────────────────────
    df = (
        df
        .pipe(features.add_form)
        .pipe(features.add_goal_xg_diff)
        .pipe(features.add_rolling_stats, window=10)
    )
    df = df[df.result.isin(["H","D","A"])].fillna(0)
    print("Datensätze nach FE:", df.shape)

    # ─── 2) Split (letzte Saison als Hold-out) ───────────────
    last_season = df.Season.iloc[-1]
    train_df    = df[df.Season != last_season]
    test_df     = df[df.Season == last_season]
    X_train, y_train = train_df[features.NUM_FEATS], train_df.result
    X_test,  y_test  = test_df [features.NUM_FEATS], test_df.result

    # ─── 3) Modelle defininieren ────────────────────────────
    pipelines = {
        "RF": Pipeline([
            ("pre", features.build_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=400, random_state=42))
        ]),
        "LR": Pipeline([
            ("pre", features.build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1500, multi_class="multinomial"))
        ]),
        "SVC": Pipeline([
            ("pre", features.build_preprocessor()),
            ("clf", SVC(C=10, probability=True))
        ]),
    }

    label = {"H":0, "D":1, "A":2}
    results, best_ll, best_name, best_pipe = [], float("inf"), None, None

    for name, pipe in pipelines.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        dur  = time.time() - t0
        prob = pipe.predict_proba(X_test)
        pred = pipe.predict(X_test)

        acc  = accuracy_score(y_test, pred)
        ll   = log_loss(y_test, prob)
        br   = np.mean([brier_score_loss((y_test==k).astype(int), prob[:,k]) for k in range(3)])
        rmse = math.sqrt(mean_squared_error(pd.Series(y_test).map(label),
                                            pd.Series(pred ).map(label)))

        results.append([name, acc, ll, br, rmse, dur])
        print(f"{name}:  acc={acc:.3f}  logloss={ll:.3f}  rmse={rmse:.3f}")

        if ll < best_ll:
            best_ll, best_name, best_pipe = ll, name, pipe

    # ─── 4) Report & Speichern ───────────────────────────────
    cols = ["model","accuracy","logloss","brier","rmse","sec"]
    rep  = pd.DataFrame(results, columns=cols) \
             .set_index("model") \
             .sort_values("logloss")

    Path("reports").mkdir(exist_ok=True)
    rep.to_csv("reports/model_metrics.csv")
    print("\n🏁 Metrics → reports/model_metrics.csv")
    print(rep)

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_pipe, f)
    print(f"🚀 best_model.pkl ({best_name}) gespeichert")

if __name__ == "__main__":
    main()