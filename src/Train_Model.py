import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("churn_prediction")

def log_model_with_mlflow(model, model_name, X_train, X_test, y_train, y_test, params):
    """Train, evaluate and log model with MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)
        
        # Print results
        print(f"\n{'='*50}")
        print(f"  {model_name} RESULTS")
        print('='*50)
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-Score:  {f1:.3f}")
        print(f"ROC-AUC:   {roc_auc:.3f}")
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
        
        print(f"\n✅ Logged to MLflow (run_id: {mlflow.active_run().info.run_id})")
        
        return model, roc_auc

def train():
    # Load PCA-reduced data
    X_train = pd.read_csv("data/train_test/X_train_pca.csv")
    X_test = pd.read_csv("data/train_test/X_test_pca.csv")
    y_train = pd.read_csv("data/train_test/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/train_test/y_test.csv").values.ravel()
    
    print(f"Training with {X_train.shape[1]} PCA features")
    print(f"Churn rate in train: {y_train.mean():.2%}")
    
    results = {}
    
    # Model 1: Logistic Regression
    print("\n--- Training Logistic Regression ---")
    lr_params = {"class_weight": "balanced", "max_iter": 1000, "random_state": 42}
    lr = LogisticRegression(**lr_params)
    _, lr_roc = log_model_with_mlflow(lr, "LogisticRegression", X_train, X_test, y_train, y_test, lr_params)
    results['LogisticRegression'] = lr_roc
    
    # Model 2: Random Forest
    print("\n--- Training Random Forest ---")
    rf_params = {"n_estimators": 100, "class_weight": "balanced", "max_depth": 10, "min_samples_split": 5, "random_state": 42}
    rf = RandomForestClassifier(**rf_params)
    _, rf_roc = log_model_with_mlflow(rf, "RandomForest", X_train, X_test, y_train, y_test, rf_params)
    results['RandomForest'] = rf_roc
    
    # Model 3: Gradient Boosting
    print("\n--- Training Gradient Boosting ---")
    gb_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}
    gb = GradientBoostingClassifier(**gb_params)
    _, gb_roc = log_model_with_mlflow(gb, "GradientBoosting", X_train, X_test, y_train, y_test, gb_params)
    results['GradientBoosting'] = gb_roc
    
    # Print comparison
    print(f"\n{'='*50}")
    print("  MODEL COMPARISON SUMMARY")
    print('='*50)
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:<20} ROC-AUC: {score:.3f}")
    
    # Select best model for deployment
    best_model_name = max(results, key=results.get)
    best_model = {'LogisticRegression': lr, 'RandomForest': rf, 'GradientBoosting': gb}[best_model_name]
    
    # Save best model locally for prediction script
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.pkl")
    print(f"\n✅ Best model ({best_model_name}) saved to models/model.pkl for deployment")
    
    # Save results summary
    with open("reports/model_results.txt", "w") as f:
        f.write("MODEL COMPARISON RESULTS (logged to MLflow)\n")
        f.write("="*40 + "\n")
        for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            marker = " <- DEPLOYED" if name == best_model_name else ""
            f.write(f"{name}: ROC-AUC = {score:.3f}{marker}\n")
    
    print("\n📊 View MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")

if __name__ == "__main__":
    train()