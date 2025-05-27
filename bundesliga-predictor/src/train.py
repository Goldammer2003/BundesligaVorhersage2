# ──────────────────────────────────────────────────────────────
# src/train.py
"""
Training-Pipeline · Daten  ➜  Enrichment ➜  FE ➜  Auto-Feature-Selektion
                                ➜  Modelltraining ➜  Reporting
Speichert:
  • reports/model_metrics.csv
  • reports/correlation_matrix.png
  • models/best_model.pkl
"""
import time, math, pickle, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm      import SVC
from sklearn.metrics  import (accuracy_score, log_loss,
                              mean_squared_error, brier_score_loss)
from sklearn.pipeline import Pipeline

from src.config   import SEASONS, MODEL_DIR
from src.ingest   import load_fd, add_h2h, add_fbref_xg
import src.features as feats              # NUM_FEATS, add_* utils

warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)

# ──────────────────────────────────────────────────────────────
def correlation_filter(df: pd.DataFrame,
                       columns: list[str],
                       threshold: float = 0.90) -> list[str]:
    """
    entfernt hoch korrelierte Features (|ρ| ≥ threshold)
    behält immer das *erste* Feature einer korrelierten Gruppe.
    """
    corr = df[columns].corr().abs()
    keep, dropped = [], set()
    for c in corr.columns:
        if c in dropped:
            continue
        keep.append(c)
        # alle weiteren Features, die mit c hoch korrelieren, markieren
        dropped |= set(corr.index[(corr[c] >= threshold) & (corr.index != c)])
    return keep

# ──────────────────────────────────────────────────────────────
def main() -> None:
    print("✅ train.py gestartet")

    # 0) Daten laden + Enrichment
    df = (load_fd(SEASONS)
            .pipe(add_fbref_xg)
            .pipe(add_h2h)
            .pipe(feats.add_form)
            .pipe(feats.add_goal_xg_diff)
            .pipe(feats.add_rolling_stats, window=10))
    df = df[df.result.isin(["H","D","A"])].fillna(0)
    print("Datensätze nach FE:", df.shape)

    # 1) Korrelationsmatrix & Stufe-1 Selektion
    base_feats = feats.NUM_FEATS
    corr_feats = correlation_filter(df, base_feats, threshold=0.90)
    corr_mat   = df[corr_feats].corr()
    Path("reports").mkdir(exist_ok=True)
    plt.figure(figsize=(10,6))
    sns.heatmap(corr_mat, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, cbar_kws={"label": "ρ"})
    plt.title("Korrelationsmatrix numerischer Features")
    plt.tight_layout()
    plt.savefig("reports/correlation_matrix.png", dpi=150)
    plt.close()
    print("📊 reports/correlation_matrix.png gespeichert")
    print("🪄 Nach Korrelations-Filter behalten:", corr_feats)

    # 2) Train/Valid-Split (letzte 3 Saisons = Hold-out)
    df["SeasonYear"] = df.Season.str[:4].astype(int)
    years       = sorted(df.SeasonYear.unique())
    train_years = years[:-3];  test_years = years[-3:]
    train_df    = df[df.SeasonYear.isin(train_years)]
    test_df     = df[df.SeasonYear.isin(test_years)]
    y_train, y_test = train_df.result, test_df.result
    print(f"  → Train Saisons: {train_years}")
    print(f"  → Test  Saisons: {test_years}")

    # 3) Stufe-2 Selektion (L1-LogReg)  ───────────────────────
    l1_selector = SelectFromModel(
        LogisticRegression(penalty="l1", C=1.0, solver="liblinear",
                           max_iter=1000, random_state=42),
        threshold="mean"          # behält Features mit |coef| ≥ Durchschnitt
    ).fit(train_df[corr_feats], y_train)
    final_feats = [f for f, keep in zip(corr_feats, l1_selector.get_support())
                   if keep]
    print("✨ Finale Feature-Auswahl:", final_feats)

    # Helper: Preprocessor auf Basis final_feats
    def make_pre():
        return ColumnTransformer(
            [("num", StandardScaler(), final_feats)],
            remainder="drop"
        )

    # 4) Modell-Pipelines
    pipelines = {
        "RF":  Pipeline([("pre", make_pre()),
                         ("clf", RandomForestClassifier(
                             n_estimators=400, random_state=42))]),
        "LR":  Pipeline([("pre", make_pre()),
                         ("clf", LogisticRegression(
                             C=1.0, max_iter=1500,
                             multi_class="multinomial"))]),
        "SVC": Pipeline([("pre", make_pre()),
                         ("clf", SVC(C=0.5, probability=True,
                                     random_state=42))]),
    }

    X_train, X_test = train_df[final_feats], test_df[final_feats]
    label_map = {"H":0, "D":1, "A":2}
    results, best_ll = [], float("inf")

    # 5) Training & Evaluation
    for name, pipe in pipelines.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        dur  = time.time() - t0
        prob = pipe.predict_proba(X_test)
        pred = pipe.predict(X_test)

        acc  = accuracy_score(y_test, pred)
        ll   = log_loss(y_test, prob)
        br   = np.mean([brier_score_loss((y_test==k).astype(int), prob[:,k])
                        for k in range(3)])
        rmse = math.sqrt(mean_squared_error(
            y_test.map(label_map), pd.Series(pred).map(label_map)))

        results.append([name, acc, ll, br, rmse, dur])
        print(f"{name}:  acc={acc:.3f}  logloss={ll:.3f}  rmse={rmse:.3f}")

        if ll < best_ll:
            best_ll, best_name, best_pipe = ll, name, pipe

    # 6) Reporting
    rep_cols = ["model","accuracy","logloss","brier","rmse","sec"]
    rep = pd.DataFrame(results, columns=rep_cols)\
            .set_index("model").sort_values("logloss")
    rep.to_csv("reports/model_metrics.csv")
    print("\n🏁 reports/model_metrics.csv gespeichert")
    print(rep)

    MODEL_DIR.mkdir(exist_ok=True)
    pickle.dump(best_pipe, open(MODEL_DIR / "best_model.pkl","wb"))
    print(f"🚀 models/best_model.pkl ({best_name}) gespeichert")

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()