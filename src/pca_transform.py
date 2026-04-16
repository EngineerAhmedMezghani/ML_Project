import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

def apply_pca():
    """Apply PCA to reduce dimensions and save transformed data."""
    # Load train/test data
    X_train = pd.read_csv("data/train_test/X_train.csv")
    X_test = pd.read_csv("data/train_test/X_test.csv")
    
    print(f"Original features: {X_train.shape[1]}")
    
    # Select only numeric columns (exclude IP addresses and other non-numeric data)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric features used for PCA: {len(numeric_cols)}")
    
    X_train = X_train[numeric_cols]
    X_test = X_test[numeric_cols]
    
    # Standardize features (important for PCA)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA - keep 95% variance
    n_components2=0.4
    pca = PCA(n_components2, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # xj′​=​(xj​−μj​​) / σj

    print(f"PCA reduced to {X_train_pca.shape[1]} components ({n_components2}% variance)")
    print(f"Explained variance: {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")
    
    # Save transformed data
    pd.DataFrame(X_train_pca).to_csv("data/train_test/X_train_pca.csv", index=False)
    pd.DataFrame(X_test_pca).to_csv("data/train_test/X_test_pca.csv", index=False)
    
    loadings = pd.DataFrame(
        pca.components_,
        columns=numeric_cols,
        index=[f"PC{i}" for i in range(pca.n_components_)]
    )
    print(loadings)

    # Save transformers for later use
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(pca, "models/pca.pkl")
    
    print("✅ Saved: X_train_pca.csv, X_test_pca.csv, scaler.pkl, pca.pkl")
    
    return X_train_pca, X_test_pca

if __name__ == "__main__":
    apply_pca()
