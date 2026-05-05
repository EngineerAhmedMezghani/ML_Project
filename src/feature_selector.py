import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from preprocessing import (
    fix_age_column,
    fix_support_tickets_and_satisfaction,
    fix_registration_date,
    fix_newsletter_subscribed,
    fix_AvgDaysBetweenPurchases,
    fix_account_status_and_churn,
    extract_ip_features,
)

# These 9 features exist in the RAW CSV and will be our inputs
INPUT_FEATURES = [
    "Recency",
    "Age",
    "Region",
    "MonetaryTotal",
    "Frequency",
    "SatisfactionScore",
    "LoyaltyLevel",
    "CustomerType",
    "AccountStatus",
]

DROP_FROM_TARGETS = ["CustomerID"]


def prepare_raw_data(df_raw):
    """
    Apply ONLY cleaning steps from preprocessing.py.
    Do NOT apply encoding() — we'll use LabelEncoder for categoricals.
    This preserves the 9 input features as single columns.
    """
    df = df_raw.copy()
    
    # Cleaning only (no encoding)
    df = fix_age_column(df)
    df = fix_support_tickets_and_satisfaction(df)
    df = fix_registration_date(df)
    df = fix_newsletter_subscribed(df)
    df = fix_AvgDaysBetweenPurchases(df)
    df = fix_account_status_and_churn(df)
    df = extract_ip_features(df, ip_col="LastLoginIP")
    
    # Drop columns that are not useful as targets either
    # (NewsletterSubscribed was already dropped in fix_newsletter_subscribed)
    # (AvgDaysBetweenPurchases was already dropped)
    
    return df


def train():
    # Load raw data
    df_raw = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
    
    # Apply cleaning only (preserve original column structure)
    df = prepare_raw_data(df_raw)
    
    # Build encoders for categorical INPUT_FEATURES
    encoders = {}
    categorical_input_cols = []
    
    for col in INPUT_FEATURES:
        if col not in df.columns:
            raise KeyError(f"Input feature '{col}' not found in dataframe. Available: {list(df.columns)}")
            
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            le.fit(df[col])
            df[col] = le.transform(df[col])
            encoders[col] = le
            categorical_input_cols.append(col)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Determine target columns (everything except inputs and drops)
    target_cols = [c for c in df.columns if c not in INPUT_FEATURES + DROP_FROM_TARGETS]
    
    # Identify and encode categorical targets
    categorical_targets = []
    target_encoders = {}
    
    for col in target_cols:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            # Handle missing values
            df[col] = df[col].fillna("Unknown")
            le.fit(df[col])
            df[col] = le.transform(df[col])
            target_encoders[col] = le
            categorical_targets.append(col)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any remaining NaN
    df = df.fillna(df.median(numeric_only=True))
    
    X = df[INPUT_FEATURES]
    y = df[target_cols]
    
    print(f"\nInput features ({len(INPUT_FEATURES)}): {INPUT_FEATURES}")
    print(f"Categorical inputs: {categorical_input_cols}")
    print(f"Target features ({len(target_cols)}): {target_cols[:5]}... (showing first 5)")
    print(f"Categorical targets: {categorical_targets}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    )
    model.fit(X_train, y_train)

    # --- Evaluation ---
    preds = model.predict(X_test)
    pred_df = pd.DataFrame(preds, columns=target_cols)

    print("\n" + "=" * 60)
    print("  FEATURE SELECTOR – Test-Set Evaluation")
    print("=" * 60)
    for col in target_cols:
        rmse = np.sqrt(np.mean((y_test[col].values - pred_df[col].values) ** 2))
        print(f"  {col:35s} RMSE = {rmse:.4f}")

    # Save artefacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/feature_selector_model.pkl")
    joblib.dump(encoders, "models/feature_selector_encoders.pkl")
    joblib.dump(target_encoders, "models/feature_selector_target_encoders.pkl")
    joblib.dump(target_cols, "models/feature_selector_targets.pkl")
    joblib.dump(categorical_targets, "models/feature_selector_cat_targets.pkl")
    joblib.dump(categorical_input_cols, "models/feature_selector_cat_inputs.pkl")

    print("\n  Model saved to: models/feature_selector_model.pkl")
    print("=" * 60)


def predict_features(input_dict: dict) -> dict:
    """
    Predict all non-input features from the 9 known features.
    """
    model = joblib.load("models/feature_selector_model.pkl")
    encoders = joblib.load("models/feature_selector_encoders.pkl")
    target_encoders = joblib.load("models/feature_selector_target_encoders.pkl")
    target_cols = joblib.load("models/feature_selector_targets.pkl")
    cat_targets = joblib.load("models/feature_selector_cat_targets.pkl")
    cat_inputs = joblib.load("models/feature_selector_cat_inputs.pkl")

    input_df = pd.DataFrame([input_dict])

    # Encode inputs
    for col in INPUT_FEATURES:
        if col in cat_inputs and col in encoders:
            le = encoders[col]
            val = str(input_df[col].iloc[0])
            input_df[col] = le.transform([val])[0] if val in le.classes_ else -1
        else:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

    input_df = input_df[INPUT_FEATURES]
    
    # Predict
    preds = model.predict(input_df)
    pred_df = pd.DataFrame(preds, columns=target_cols)

    # Decode categorical targets
    for col in cat_targets:
        if col in target_encoders:
            le = target_encoders[col]
            encoded = pred_df[col].round().astype(int).clip(0, len(le.classes_) - 1)
            pred_df[col] = le.inverse_transform(encoded)

    # Clean types for JSON
    result = {}
    for k, v in pred_df.iloc[0].items():
        if isinstance(v, (np.integer, np.floating)):
            result[k] = int(v) if isinstance(v, np.integer) else float(v)
        else:
            result[k] = v
    
    return result


def demo():
    """Run a quick demo using the first row of the raw dataset as input."""
    raw = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
    sample = raw.iloc[0]
    inputs = {k: sample[k] for k in INPUT_FEATURES}

    print("\n" + "=" * 60)
    print("  FEATURE SELECTOR – Demo Prediction")
    print("=" * 60)
    print("\n  Inputs:")
    for k, v in inputs.items():
        print(f"    {k:25s}: {v}")

    predictions = predict_features(inputs)

    print("\n  Predicted outputs (sample):")
    for k, v in list(predictions.items())[:10]:
        print(f"    {k:25s}: {v}")
    print(f"    ... ({len(predictions)} total predicted features)")
    print("=" * 60)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train()
    demo()