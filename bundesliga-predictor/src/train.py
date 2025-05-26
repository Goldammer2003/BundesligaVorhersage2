"""
Training, Modellvergleich und Speicherung des besten Bundesliga-Vorhersagemodells.
Danach (optional) Monte-Carlo-Simulation der Meisterwahrscheinlichkeiten.
"""
# -------------------------------------------------- Basics
import warnings, pickle, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, log_loss, classification_report,
    mean_absolute_error, mean_squared_error, brier_score_loss
)

from src import ingest, features     # eigene Module

warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)
print("train.py gestartet ✅")

# -------------------------------------------------- 0) Daten laden
print("Lade Daten …")
df = ingest.load_fd()

# -------------------------------------------------- 1) Feature-Engineering
print("Wende Feature Engineering an …")
df = (df
      .pipe(features.add_implied_prob)
      .pipe(features.add_form)
      .pipe(features.add_goal_xg_diff))

USE_H2H = False      # ⚡ True dauert ~15 min, False setzt neutrales 0.5-Default
if USE_H2H:
    df = ingest.add_h2h(df)
else:
    df["h2h_home_winrate"] = 0.5

for col in features.NUM_FEATS:
    df[col] = df.get(col, 0).fillna(0)

df = df[df["result"].isin(["H", "D", "A"])].copy()
print("Shape nach FE:", df.shape)

# -------------------------------------------------- 2) Korrelation
# -------------------------------------------------- 2) Korrelation prüfen
corr_df = df[features.NUM_FEATS].copy()

# Warnung ausgeben, falls Feature-Spalten konstant sind (z. B. nur 0)
dropped_cols = corr_df.columns[corr_df.nunique() <= 1]
if len(dropped_cols) > 0:
    print("⚠️  Folgende Features wurden nicht in die Korrelationsmatrix aufgenommen (nur konstante Werte):")
    for col in dropped_cols:
        print(f"   - {col}")

# Nur valide Spalten behalten
corr_df = corr_df.drop(columns=dropped_cols)

# Korrelation berechnen und speichern
corr = corr_df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Korrelationsmatrix numerischer Features")
plt.tight_layout()
Path("reports").mkdir(exist_ok=True)
plt.savefig("reports/correlation_matrix.png")
plt.close()
print("Korrelationsmatrix gespeichert → reports/correlation_matrix.png")
# -------------------------------------------------- 3) Zeitlicher Train/Test-Split
df = df.sort_values("date")
cut = int(len(df)*0.8)
train_df, test_df = df.iloc[:cut], df.iloc[cut:]

X_train, y_train = train_df[features.NUM_FEATS], train_df["result"]
X_test,  y_test  = test_df[features.NUM_FEATS],  test_df["result"]

# -------------------------------------------------- 4) Pipelines
pipelines = {
    "RandomForest": Pipeline([
        ("pre", features.build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=400, random_state=42))
    ]),
    "LogisticRegression": Pipeline([
        ("pre", features.build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial"))
    ]),
    "SVC": Pipeline([
        ("pre", features.build_preprocessor()),
        ("clf", SVC(probability=True))
    ]),
}

# -------------------------------------------------- 5) Metrik-Helper
label_map = {"H":0,"D":1,"A":2}
y_train_enc = y_train.map(label_map)
y_test_enc  = y_test.map(label_map)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true),1e-9,None)))

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2*np.abs(y_pred - y_true) /
                                    (np.abs(y_true)+np.abs(y_pred)+1e-9))

def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true))+1e-9)

def brier_multiclass(y_true_enc, prob_mat, n_classes=3):
    """mittlerer one-vs-all-Brier-Score"""
    scores=[]
    for k in range(n_classes):
        binary_true = (y_true_enc==k).astype(int)
        scores.append(brier_score_loss(binary_true, prob_mat[:,k]))
    return np.mean(scores)

# -------------------------------------------------- 6) Training & Evaluation
results=[]
best_name,best_model,best_ll=None,None,np.inf
print("Starte Training & Evaluation …")
for name,pipe in pipelines.items():
    t0=time.time(); pipe.fit(X_train,y_train); dur=time.time()-t0
    prob = pipe.predict_proba(X_test)
    pred = pipe.predict(X_test)
    pred_enc = pd.Series(pred).map(label_map)

    acc   = accuracy_score(y_test,pred)
    ll    = log_loss(y_test,prob)
    br    = brier_multiclass(y_test_enc,prob)          # ✅ multiclass-Brier
    mae   = mean_absolute_error(y_test_enc,pred_enc)
    mse   = mean_squared_error(y_test_enc,pred_enc)
    rmse  = math.sqrt(mse)
    mape_ = mape(y_test_enc,pred_enc)
    smape_= smape(y_test_enc,pred_enc)
    wape_ = wape(y_test_enc,pred_enc)

    results.append([name,acc,ll,br,mae,mse,rmse,mape_,smape_,wape_,dur])

    print(f"\n{name}: Acc {acc:.3f} | LogLoss {ll:.3f} | RMSE {rmse:.3f}")
    print(classification_report(y_test,pred))

    if ll < best_ll:
        best_ll,best_name,best_model = ll,name,pipe

# -------------------------------------------------- 7) Ergebnis-Tabelle
cols=["model","accuracy","log_loss","brier","mae","mse","rmse",
      "mape","smape","wape","train_sec"]
res_df = pd.DataFrame(results,columns=cols).set_index("model")
res_df.to_csv("reports/model_metrics.csv")
print("\nGesamtergebnis → reports/model_metrics.csv"); print(res_df)

# -------------------------------------------------- 8) Bestes Modell speichern
Path("models").mkdir(exist_ok=True)
with open("models/best_model.pkl","wb") as f:
    pickle.dump(best_model,f)
print(f"\n🚀  Bestes Modell ({best_name}) gespeichert → models/best_model.pkl")