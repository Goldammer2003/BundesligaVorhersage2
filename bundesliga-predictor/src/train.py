# src/train.py
"""
Training-Pipeline: Daten laden ➜ Enrichment ➜ Features ➜ Modelle trainieren
Speichert:
  • reports/model_metrics.csv
  • reports/correlation_matrix.png
  • models/best_model.pkl
"""
import time
import math
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, log_loss,
    mean_squared_error, brier_score_loss
)

from src.config     import SEASONS, MODEL_DIR
from src.ingest     import load_fd, add_h2h, add_fbref_xg
import src.features as features   # NUM_FEATS, add_*, build_preprocessor

warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)

def main():
    print("✅ train.py gestartet")

    # ─── 0) Daten laden + Enrichment ─────────────────────────
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

    # ─── 1b) Korrelationsmatrix numerischer Features ─────────
    corr = df[features.NUM_FEATS].corr()
    plt.figure(figsize=(10,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Korrelationsmatrix numerischer Features")
    Path("reports").mkdir(exist_ok=True)
    plt.savefig("reports/correlation_matrix.png", dpi=150)
    plt.close()
    print("📊 reports/correlation_matrix.png gespeichert")

    # ─── 2) Split (letzte 3 Saisons als Hold-out) ────────────
    df["SeasonYear"] = df.Season.str[:4].astype(int)
    years = sorted(df.SeasonYear.unique())
    train_years, test_years = years[:-3], years[-3:]
    train_df = df[df.SeasonYear.isin(train_years)]
    test_df  = df[df.SeasonYear.isin(test_years)]
    print(f"  → Train Saisons: {train_years}")
    print(f"  → Test  Saisons: {test_years}")

    X_train = train_df[features.NUM_FEATS]
    y_train = train_df.result
    X_test  = test_df [features.NUM_FEATS]
    y_test  = test_df .result
    print("Train/Test Shapes:", X_train.shape, X_test.shape)

    # ─── 3) Modelle definieren ──────────────────────────────
    pipelines = {
        "RF": Pipeline([
            ("pre", features.build_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=400, random_state=42))
        ]),
        "LR": Pipeline([
            ("pre", features.build_preprocessor()),
            ("clf", LogisticRegression(
                C=1.0, max_iter=1500, multi_class="multinomial"))
        ]),
        "SVC": Pipeline([
            ("pre", features.build_preprocessor()),
            ("clf", SVC(C=0.5, probability=True, random_state=42))
        ]),
    }

    label = {"H":0, "D":1, "A":2}
    results, best_ll, best_name, best_pipe = [], float("inf"), None, None

    # ─── 4) Training & Evaluation ───────────────────────────
    for name, pipe in pipelines.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        dur  = time.time() - t0
        prob = pipe.predict_proba(X_test)
        pred = pipe.predict(X_test)

        acc  = accuracy_score(y_test, pred)
        ll   = log_loss(y_test, prob)
        br   = np.mean([
            brier_score_loss((y_test==k).astype(int), prob[:,k])
            for k in range(3)
        ])
        rmse = math.sqrt(mean_squared_error(
            pd.Series(y_test).map(label),
            pd.Series(pred).map(label)
        ))

        results.append([name, acc, ll, br, rmse, dur])
        print(f"{name}:  acc={acc:.3f}  logloss={ll:.3f}  rmse={rmse:.3f}")

        if ll < best_ll:
            best_ll, best_name, best_pipe = ll, name, pipe

    # ─── 5) Reporting & Speichern ───────────────────────────
    cols = ["model","accuracy","logloss","brier","rmse","sec"]
    rep  = pd.DataFrame(results, columns=cols) \
             .set_index("model") \
             .sort_values("logloss")
    rep.to_csv("reports/model_metrics.csv")
    print("\n🏁 reports/model_metrics.csv gespeichert")
    print(rep)

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "best_model.pkl","wb") as f:
        pickle.dump(best_pipe, f)
    print(f"🚀 models/best_model.pkl ({best_name}) gespeichert")

if __name__=="__main__":
    main()