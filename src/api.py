import os
import sys
import json
import uuid
from datetime import datetime

# Ensure src/ is on the path so local imports work when run from project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np

# MLflow exactement comme Train_Model.py
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("churn_prediction")

import feature_selector

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

API_PORT = int(os.environ.get("API_PORT", 5000))

# -----------------------------------------------------------------------------
# Helper : charger le modèle depuis MLflow ou fallback local
# -----------------------------------------------------------------------------
def _load_model_from_mlflow():
    """Tente de charger le dernier modèle loggé dans MLflow, sinon fallback local."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("churn_prediction")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            if runs:
                run_id = runs[0].info.run_id
                model_uri = f"runs:/{run_id}/model"
                model = mlflow.sklearn.load_model(model_uri)
                print(f"[API] Modèle chargé depuis MLflow (run_id: {run_id})")
                return model, run_id
    except Exception as e:
        print(f"[API] Impossible de charger depuis MLflow: {e}")
    
    # Fallback local
    local_path = "models/model.pkl"
    if os.path.exists(local_path):
        model = joblib.load(local_path)
        print(f"[API] Modèle chargé depuis {local_path}")
        return model, None
    return None, None


def _ensure_model():
    """Train feature-selector model if artefacts are missing."""
    artefacts = [
        "models/feature_selector_model.pkl",
        "models/feature_selector_encoders.pkl",
        "models/feature_selector_target_encoders.pkl",
        "models/feature_selector_targets.pkl",
        "models/feature_selector_cat_targets.pkl",
        "models/feature_selector_cat_inputs.pkl",
    ]
    missing = [a for a in artefacts if not os.path.exists(a)]
    if missing:
        print("[API] Missing model artefacts, training now...")
        os.makedirs("models", exist_ok=True)
        feature_selector.train()
        print("[API] Training complete.")


# -----------------------------------------------------------------------------
# Endpoints existants (inchangés + health)
# -----------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mlflow_tracking_uri": mlflow.get_tracking_uri()})


@app.route("/api/inputs", methods=["GET"])
def get_inputs():
    """Return metadata about the 9 input features."""
    df = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
    inputs_meta = []
    for col in feature_selector.INPUT_FEATURES:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        meta = {
            "name": col,
            "type": "categorical" if series.dtype == object else "numeric",
        }
        if series.dtype == object:
            meta["options"] = sorted(series.unique().tolist())
        else:
            meta["min"] = float(series.min())
            meta["max"] = float(series.max())
            meta["mean"] = float(series.mean())
        inputs_meta.append(meta)
    return jsonify({"inputs": inputs_meta})


# -----------------------------------------------------------------------------
# Prédiction avec logging MLflow exactement comme Train_Model.py
# -----------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    _ensure_model()
    data = request.get_json(force=True)

    # Validate required fields
    missing = [f for f in feature_selector.INPUT_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing input fields: {missing}"}), 400

    # --- Logging MLflow pour cette prédiction ---
    prediction_id = str(uuid.uuid4())[:8]
    with mlflow.start_run(run_name=f"api_prediction_{prediction_id}"):
        # Log les inputs
        mlflow.log_param("prediction_id", prediction_id)
        mlflow.log_param("timestamp", datetime.now().isoformat())
        for k, v in data.items():
            # On log les valeurs catégoriques/string comme paramètres MLflow
            if isinstance(v, (int, float, bool, str)):
                mlflow.log_param(f"input_{k}", v)
        
        try:
            predictions = feature_selector.predict_features(data)
        except Exception as e:
            mlflow.log_param("error", str(e))
            return jsonify({"error": str(e)}), 500

        # Separate churn from the rest
        churn_val = predictions.pop("Churn", None)
        churn_result = None
        if churn_val is not None:
            prob = float(churn_val)
            churn_result = {
                "probability": prob,
                "prediction": 0 if prob >= 0.5 else 1,
                "label": "loyal" if prob >= 0.5 else "churn",
                "risk_level": "high" if prob > 0.7 else "medium" if prob > 0.3 else "low",
            }
            # Log metrics MLflow exactement comme Train_Model.py
            mlflow.log_metric("churn_probability", prob)
            mlflow.log_metric("churn_prediction", churn_result["prediction"])
            mlflow.log_metric("risk_score", prob)  # alias utile pour le dashboard

        # Log toutes les autres prédictions
        for k, v in predictions.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"pred_{k}", float(v))

    # Categorise remaining predictions for the UI
    categories = {
        "Monetary": [k for k in predictions if "Monetary" in k or "Spending" in k or "Basket" in k],
        "Behaviour": [k for k in predictions if any(x in k for x in ["Quantity", "Transaction", "Frequency", "Recency", "Return", "Cancel"])],
        "Preferences": [k for k in predictions if any(x in k for x in ["Preferred", "Favorite", "Season", "Time", "Weekend"])],
        "Profile": [k for k in predictions if any(x in k for x in ["Age", "Gender", "Country", "Region", "Account", "Loyalty", "RFM", "CustomerType", "ChurnRisk", "ProductDiversity"])],
        "Other": [],
    }
    assigned = set()
    for cat_list in categories.values():
        assigned.update(cat_list)
    categories["Other"] = [k for k in predictions if k not in assigned]

    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v}

    return jsonify({
        "inputs": data,
        "churn": churn_result,
        "predictions": predictions,
        "categories": categories,
        "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None,
    })


# -----------------------------------------------------------------------------
# Batch predict avec logging MLflow
# -----------------------------------------------------------------------------
@app.route("/api/batch", methods=["POST"])
def batch_predict():
    _ensure_model()
    data = request.get_json(force=True)
    customers = data.get("customers", [])
    if not customers:
        return jsonify({"error": "No customers provided"}), 400

    batch_id = str(uuid.uuid4())[:8]
    results = []
    
    with mlflow.start_run(run_name=f"api_batch_{batch_id}"):
        mlflow.log_param("batch_id", batch_id)
        mlflow.log_param("batch_size", len(customers))
        mlflow.log_param("timestamp", datetime.now().isoformat())

        for idx, cust in enumerate(customers):
            missing = [f for f in feature_selector.INPUT_FEATURES if f not in cust]
            if missing:
                results.append({"error": f"Missing fields: {missing}"})
                mlflow.log_param(f"cust_{idx}_error", f"Missing: {missing}")
                continue
            try:
                preds = feature_selector.predict_features(cust)
                churn_val = preds.pop("Churn", None)
                churn = None
                if churn_val is not None:
                    prob = float(churn_val)
                    churn = {
                        "probability": prob,
                        "prediction": 1 if prob >= 0.5 else 0,
                        "label": "Churn" if prob >= 0.5 else "Loyal",
                        "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low",
                    }
                    # Log metrics pour chaque customer dans le batch
                    mlflow.log_metric(f"cust_{idx}_churn_prob", prob)
                
                results.append({"predictions": preds, "churn": churn})
            except Exception as e:
                results.append({"error": str(e)})
                mlflow.log_param(f"cust_{idx}_error", str(e))

        # Log métriques agrégées du batch
        churn_probs = [r["churn"]["probability"] for r in results if r.get("churn")]
        if churn_probs:
            mlflow.log_metric("batch_mean_churn_prob", float(np.mean(churn_probs)))
            mlflow.log_metric("batch_max_churn_prob", float(np.max(churn_probs)))
            mlflow.log_metric("batch_high_risk_count", sum(1 for p in churn_probs if p > 0.7))

    return jsonify({
        "results": results,
        "batch_id": batch_id,
        "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None,
    })


# -----------------------------------------------------------------------------
# Endpoints stats (inchangés)
# -----------------------------------------------------------------------------
@app.route("/api/dataset_stats", methods=["GET"])
def dataset_stats():
    """Return real statistics computed from the raw dataset."""
    df = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
    total = len(df)
    churn_rate = float(df["Churn"].mean() * 100) if "Churn" in df.columns else 0.0
    high_risk = int(df["Churn"].sum()) if "Churn" in df.columns else 0

    return jsonify({
        "total_customers": total,
        "churn_rate": round(churn_rate, 2),
        "high_risk_customers": high_risk,
        "loyal_customers": total - high_risk,
    })


@app.route("/api/churn_summary", methods=["GET"])
def churn_summary():
    """Return churn distribution for the doughnut chart."""
    df = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
    counts = df["Churn"].value_counts().to_dict()
    return jsonify({
        "labels": ["Loyal", "Churn"],
        "data": [counts.get(0, 0), counts.get(1, 0)],
    })


@app.route("/api/risk_distribution", methods=["GET"])
def risk_distribution():
    """Run model on a random sample and return predicted risk distribution."""
    _ensure_model()
    df_raw = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
    sample = df_raw.sample(n=min(200, len(df_raw)), random_state=42)

    low = medium = high = 0
    for _, row in sample.iterrows():
        inputs = {k: row[k] for k in feature_selector.INPUT_FEATURES}
        try:
            preds = feature_selector.predict_features(inputs)
            prob = float(preds.get("Churn", 0))
            if prob > 0.7:
                high += 1
            elif prob > 0.3:
                medium += 1
            else:
                low += 1
        except Exception:
            continue

    return jsonify({
        "labels": ["Low Risk", "Medium Risk", "High Risk"],
        "data": [low, medium, high],
    })


@app.route("/api/sample_predictions", methods=["GET"])
def sample_predictions():
    """Return the first 10 rows with their churn predictions for the recent table."""
    _ensure_model()
    df_raw = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv").head(10)

    results = []
    for idx, row in df_raw.iterrows():
        inputs = {k: row[k] for k in feature_selector.INPUT_FEATURES}
        try:
            preds = feature_selector.predict_features(inputs)
            prob = float(preds.get("Churn", 0))
            results.append({
                "id": int(idx),
                "customerId": str(row.get("CustomerID", f"CUST-{idx}")),
                "region": str(row.get("Region", "")),
                "age": int(row.get("Age", 0)) if pd.notna(row.get("Age")) else None,
                "riskLevel": "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low",
                "probability": prob,
                "predictedLabel": "Churn" if prob >= 0.5 else "Loyal",
                "actualLabel": "Churn" if row.get("Churn") == 1 else "Loyal",
            })
        except Exception as e:
            results.append({
                "id": int(idx),
                "customerId": str(row.get("CustomerID", f"CUST-{idx}")),
                "error": str(e),
            })

    return jsonify({"predictions": results})


@app.route("/api/region_churn", methods=["GET"])
def region_churn():
    """Return churn rate per region for the bar chart."""
    df = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
    if "Region" not in df.columns or "Churn" not in df.columns:
        return jsonify({"error": "Missing columns"}), 400

    grouped = df.groupby("Region")["Churn"].agg(["mean", "count"]).reset_index()
    grouped["mean"] = (grouped["mean"] * 100).round(2)

    return jsonify({
        "labels": grouped["Region"].tolist(),
        "churn_rates": grouped["mean"].tolist(),
        "counts": grouped["count"].tolist(),
    })


# -----------------------------------------------------------------------------
# NOUVEAU : Endpoint /api/train identique à Train_Model.py
# -----------------------------------------------------------------------------
@app.route("/api/train", methods=["POST"])
def train_models():
    """
    Réentraîne les 3 modèles (LogisticRegression, RandomForest, GradientBoosting)
    exactement comme Train_Model.py et logge tout dans MLflow.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report, roc_auc_score, confusion_matrix,
        accuracy_score, precision_score, recall_score, f1_score
    )

    # Load PCA-reduced data
    try:
        X_train = pd.read_csv("data/train_test/X_train_pca.csv")
        X_test = pd.read_csv("data/train_test/X_test_pca.csv")
        y_train = pd.read_csv("data/train_test/y_train.csv").values.ravel()
        y_test = pd.read_csv("data/train_test/y_test.csv").values.ravel()
    except FileNotFoundError as e:
        return jsonify({"error": f"Training data not found: {e}"}), 400

    results = {}

    def log_model(model, model_name, params):
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(params)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("test_samples", X_test.shape[0])

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.sklearn.log_model(model, artifact_path="model")

            cm = confusion_matrix(y_test, y_pred)
            return {
                "model": model_name,
                "accuracy": round(accuracy, 3),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "roc_auc": round(roc_auc, 3),
                "confusion_matrix": {
                    "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
                    "FN": int(cm[1, 0]), "TP": int(cm[1, 1])
                },
                "mlflow_run_id": mlflow.active_run().info.run_id,
            }

    # Model 1: Logistic Regression
    lr_params = {"class_weight": "balanced", "max_iter": 1000, "random_state": 42}
    lr = LogisticRegression(**lr_params)
    lr_result = log_model(lr, "LogisticRegression", lr_params)
    results['LogisticRegression'] = lr_result

    # Model 2: Random Forest
    rf_params = {"n_estimators": 100, "class_weight": "balanced", "max_depth": 10, "min_samples_split": 5, "random_state": 42}
    rf = RandomForestClassifier(**rf_params)
    rf_result = log_model(rf, "RandomForest", rf_params)
    results['RandomForest'] = rf_result

    # Model 3: Gradient Boosting
    gb_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}
    gb = GradientBoostingClassifier(**gb_params)
    gb_result = log_model(gb, "GradientBoosting", gb_params)
    results['GradientBoosting'] = gb_result

    # Select best model
    best_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = {'LogisticRegression': lr, 'RandomForest': rf, 'GradientBoosting': gb}[best_name]

    # Save locally
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.pkl")

    # Save summary
    os.makedirs("reports", exist_ok=True)
    with open("reports/model_results.txt", "w") as f:
        f.write("MODEL COMPARISON RESULTS (logged to MLflow via API)\n")
        f.write("=" * 50 + "\n")
        for name, res in sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True):
            marker = " <- DEPLOYED" if name == best_name else ""
            f.write(f"{name}: ROC-AUC = {res['roc_auc']}{marker}\n")

    return jsonify({
        "status": "training_complete",
        "best_model": best_name,
        "results": results,
    })


# -----------------------------------------------------------------------------
# NOUVEAU : Charger un modèle spécifique depuis MLflow pour prédire
# -----------------------------------------------------------------------------
@app.route("/api/predict_with_mlflow_model", methods=["POST"])
def predict_with_mlflow_model():
    """
    Prédit en utilisant le modèle sklearn loggé dans MLflow (comme Train_Model.py).
    Nécessite que les données soient au format PCA (mêmes features que X_train_pca).
    """
    data = request.get_json(force=True)
    features = data.get("features")
    
    if not features:
        return jsonify({"error": "Missing 'features' array"}), 400

    model, run_id = _load_model_from_mlflow()
    if model is None:
        return jsonify({"error": "No model available in MLflow or local"}), 500

    try:
        X = pd.DataFrame([features]) if isinstance(features, list) else pd.DataFrame(features)
        prediction = model.predict(X)
        probability = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

        # Log cette prédiction dans MLflow
        with mlflow.start_run(run_name="api_mlflow_model_prediction"):
            mlflow.log_param("source_run_id", run_id)
            mlflow.log_param("n_samples", len(X))
            if probability is not None:
                mlflow.log_metric("predicted_probability", float(probability[0]))
            mlflow.log_metric("predicted_class", int(prediction[0]))

        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(probability[0]) if probability is not None else None,
            "label": "Churn" if prediction[0] == 1 else "Loyal",
            "mlflow_source_run_id": run_id,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    _ensure_model()
    print(f"[API] Starting Flask server on http://127.0.0.1:{API_PORT}")
    print(f"[API] MLflow tracking URI: {mlflow.get_tracking_uri()}")
    app.run(host="127.0.0.1", port=API_PORT, debug=True)