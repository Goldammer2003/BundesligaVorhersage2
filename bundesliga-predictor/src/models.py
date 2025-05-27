"""
Training, Hyperparameter-Tuning & Evaluation (für Streamlit-App)
"""
from typing import Dict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, classification_report
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .features import NUM_FEATS, build_preprocessor

TARGET = "result"

def _pipelines():
    pre = build_preprocessor()
    return {
        "rf": Pipeline([("pre", pre),
                        ("clf", RandomForestClassifier(random_state=42, n_estimators=500))]),
        "xgb": Pipeline([("pre", pre),
                         ("clf", XGBClassifier(random_state=42, n_estimators=500,
                                               objective="multi:softprob",
                                               eval_metric="mlogloss"))]),
        "glm": Pipeline([("pre", pre),
                         ("clf", LogisticRegression(max_iter=300, multi_class="multinomial"))]),
    }

def _param_grids():
    return {
        "rf":  {"clf__max_depth": [None, 10, 20]},
        "xgb": {"clf__max_depth": [4, 6, 8]},
        "glm": {"clf__C": [0.1, 1, 10]},
    }

def train(train_df: pd.DataFrame) -> Dict[str, Pipeline]:
    X, y = train_df[NUM_FEATS], train_df[TARGET]
    cv    = TimeSeriesSplit(n_splits=5)
    best  = {}
    for name, pipe in _pipelines().items():
        gs = GridSearchCV(pipe, _param_grids()[name], cv=cv,
                          scoring="neg_log_loss", n_jobs=-1)
        gs.fit(X, y)
        best[name] = gs.best_estimator_
        print(f"{name}: {-gs.best_score_:.4f}   (best log-loss)")
    return best

def evaluate(models: Dict[str, Pipeline], test_df: pd.DataFrame) -> pd.DataFrame:
    X, y = test_df[NUM_FEATS], test_df[TARGET]
    rows = []
    for n, m in models.items():
        prob = m.predict_proba(X)
        pred = m.predict(X)
        rows.append({
            "model": n,
            "accuracy": accuracy_score(y, pred),
            "logloss": log_loss(y, prob),
            "brier":  brier_score_loss(y.map({"H":0,"D":1,"A":2}), prob.max(axis=1)),
        })
        print("\n", n.upper(), classification_report(y, pred))
    return pd.DataFrame(rows).set_index("model").sort_values("logloss")