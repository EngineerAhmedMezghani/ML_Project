"""
Churn Prediction - MLflow Basic Template
=========================================
A clean starting point to track experiments with MLflow.
Model choice is TBD — plug in any sklearn-compatible estimator.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────
EXPERIMENT_NAME = "churn-prediction"
DATA_PATH       = "data/processed/retail_customers_processed.csv"   # ← point to your file
TARGET_COLUMN   = "Churn"                    # ← your label column (0/1 or True/False)
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# ──────────────────────────────────────────────
# 2. LOAD & SPLIT DATA
# ──────────────────────────────────────────────
def load_data(path: str, target: str):
    df = pd.read_csv(path)

    # Basic preprocessing (adapt to your schema)
    df = df.dropna()
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    # Encode categoricals if any
    X = pd.get_dummies(X, drop_first=True)

    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)


# ──────────────────────────────────────────────
# 3. TRAINING FUNCTION  (model-agnostic)
# ──────────────────────────────────────────────
def train(model, model_name: str, params: dict,
          X_train, X_test, y_train, y_test):
    """
    Trains any sklearn-compatible model and logs everything to MLflow.
    Just swap `model` to change the algorithm.
    """

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):

        # ── Scaling (optional — remove for tree-based models)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # ── Fit
        model.fit(X_train_sc, y_train)
        y_pred      = model.predict(X_test_sc)
        y_pred_proba = model.predict_proba(X_test_sc)[:, 1] \
                       if hasattr(model, "predict_proba") else None

        # ── Metrics
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred)
        auc    = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

        # ── Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        if auc:
            mlflow.log_metric("roc_auc", auc)

        # Log full classification report as artifact
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # Log the model itself
        mlflow.sklearn.log_model(model, artifact_path="model",
                                 registered_model_name=f"churn_{model_name}")

        # ── Console summary
        print(f"\n{'='*50}")
        print(f"  Run  : {model_name}")
        print(f"  ACC  : {acc:.4f}")
        print(f"  F1   : {f1:.4f}")
        print(f"  AUC  : {auc:.4f}" if auc else "  AUC  : N/A")
        print(f"{'='*50}")
        print(report)

    return model


# ──────────────────────────────────────────────
# 4. MAIN — plug your model here
# ──────────────────────────────────────────────
if __name__ == "__main__":

    # Load data
    X_train, X_test, y_train, y_test = load_data(DATA_PATH, TARGET_COLUMN)

    # ── ↓↓ SWAP THIS BLOCK with the model we decide on ↓↓ ──
    from sklearn.dummy import DummyClassifier          # placeholder
    model  = DummyClassifier(strategy="most_frequent") # replace me
    params = {"strategy": "most_frequent"}             # replace me
    # ── ↑↑ ─────────────────────────────────────────── ↑↑ ──

    train(
        model      = model,
        model_name = "baseline_dummy",
        params     = params,
        X_train    = X_train,
        X_test     = X_test,
        y_train    = y_train,
        y_test     = y_test,
    )

    # Launch UI with: mlflow ui  →  http://localhost:5000