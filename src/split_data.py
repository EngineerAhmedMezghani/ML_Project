import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data():
    # Load processed data
    df = pd.read_csv("data/processed/retail_customers_processed.csv")
    
    print(f"Dataset shape: {df.shape}")
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Stratified split (IMPORTANT for imbalanced Churn!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Create directory
    os.makedirs("data/train_test", exist_ok=True)
    
    # Save
    X_train.to_csv("data/train_test/X_train.csv", index=False)
    X_test.to_csv("data/train_test/X_test.csv", index=False)
    y_train.to_csv("data/train_test/y_train.csv", index=False)
    y_test.to_csv("data/train_test/y_test.csv", index=False)
    
    print(f"✅ Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Churn rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

if __name__ == "__main__":
    split_data()