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
    
    # Select numeric and boolean columns (exclude strings/objects)
    feature_cols = X_train.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    print(f"Features used for PCA: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")
    
    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]
    
    # Standardize features (important for PCA)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA - keep 95% variance
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"PCA reduced to {X_train_pca.shape[1]} components (kept 95% variance)")
    print(f"Explained variance: {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")
    
    # Build loadings DataFrame
    pc_names = [f"PC{i+1}" for i in range(pca.n_components_)]
    loadings = pd.DataFrame(
        pca.components_,
        columns=feature_cols,
        index=pc_names
    )
    
    # Save transformed datasets with proper column names
    os.makedirs("data/train_test", exist_ok=True)
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=pc_names)
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=pc_names)
    X_train_pca_df.to_csv("data/train_test/X_train_pca.csv", index=False)
    X_test_pca_df.to_csv("data/train_test/X_test_pca.csv", index=False)
    
    # Save transformers and loadings
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(pca, "models/pca.pkl")
    loadings.to_csv("models/pca_feature_loadings.csv")
    pd.Series(feature_cols).to_csv("models/pca_input_features.csv", index=False, header=['feature'])
    
    print("\n=== Final PCA Features to work on ===")
    print(f"Count: {len(pc_names)}")
    print(f"Names: {pc_names}")
    print("\n=== Explained Variance ===")
    print(f"Cumulative explained variance: {pca.explained_variance_ratio_.cumsum()[-1]:.4f}")
    
    print("\n✅ Saved artifacts:")
    print("   - data/train_test/X_train_pca.csv")
    print("   - data/train_test/X_test_pca.csv")
    print("   - models/scaler.pkl")
    print("   - models/pca.pkl")
    print("   - models/pca_feature_loadings.csv")
    print("   - models/pca_input_features.csv")
    
    return X_train_pca, X_test_pca

if __name__ == "__main__":
    apply_pca()
