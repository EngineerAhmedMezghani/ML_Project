import pandas as pd
import joblib
import numpy as np

def predict_churn(customer_data):
    """
    Predict churn probability for customer(s).
    
    Args:
        customer_data: DataFrame with same features as training data (pre-PCA)
    
    Returns:
        predictions: array of 0 (loyal) or 1 (churn)
        probabilities: array of churn probabilities
    """
    # Load transformers and model
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")
    model = joblib.load("models/model.pkl")
    
    # Transform: Scale → PCA → Predict
    X_scaled = scaler.transform(customer_data)
    X_pca = pca.transform(X_scaled)
    
    predictions = model.predict(X_pca)
    probabilities = model.predict_proba(X_pca)[:, 1]
    
    return predictions, probabilities

def predict_single_customer(customer_dict):
    """Predict for a single customer from dictionary."""
    df = pd.DataFrame([customer_dict])
    pred, prob = predict_churn(df)
    return {
        'churn_prediction': int(pred[0]),
        'churn_probability': float(prob[0]),
        'risk_level': 'High' if prob[0] > 0.7 else 'Medium' if prob[0] > 0.3 else 'Low'
    }

if __name__ == "__main__":
    # Example: Load a sample from test set and predict
    X_test = pd.read_csv("data/train_test/X_test.csv")
    y_test = pd.read_csv("data/train_test/y_test.csv").values.ravel()
    
    # Take first 5 samples
    sample = X_test.head(5)
    preds, probs = predict_churn(sample)
    
    print("\nSample Predictions:")
    print("-" * 50)
    for i, (pred, prob, actual) in enumerate(zip(preds, probs, y_test[:5])):
        status = "Churn" if pred == 1 else "Loyal"
        actual_status = "Churn" if actual == 1 else "Loyal"
        print(f"Customer {i+1}: Predicted={status} (prob={prob:.2%}) | Actual={actual_status}")
