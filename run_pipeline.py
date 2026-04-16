#!/usr/bin/env python
"""
Simple ML Pipeline Runner
Run this script to execute the complete pipeline from preprocessing to model training.
"""

import subprocess
import sys
import os

def run_step(script_name, description):
    """Run a pipeline step and handle errors."""
    print(f"\n{'='*60}")
    print(f"  STEP: {description}")
    print('='*60)
    
    result = subprocess.run([sys.executable, f"src/{script_name}"], capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ ERROR: {script_name} failed")
        sys.exit(1)
    
    print(f"✅ {script_name} completed successfully")
    return True

def main():
    print("\n" + "="*60)
    print("  ML PIPELINE - Customer Churn Prediction")
    print("  Atelier Machine Learning - GI2")
    print("="*60)
    
    # Ensure directories exist
    for d in ["data/processed", "data/train_test", "models", "reports", "outputs"]:
        os.makedirs(d, exist_ok=True)
    
    steps = [
        ("preprocessing.py", "Data Preprocessing & Feature Engineering"),
        ("split_data.py", "Train/Test Split"),
        ("utils.py", "Correlation Analysis"),
        ("pca_transform.py", "PCA Dimensionality Reduction"),
        ("Train_Model.py", "Model Training & Evaluation"),
    ]
    
    for script, desc in steps:
        run_step(script, desc)
    
    print("\n" + "="*60)
    print("  PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  - Processed data: data/processed/retail_customers_processed.csv")
    print("  - Train/test splits: data/train_test/")
    print("  - Correlation plots: outputs/correlation_*.png")
    print("  - Trained model: models/model.pkl")
    print("  - Scaler & PCA: models/scaler.pkl, models/pca.pkl")
    print("  - Results report: reports/model_results.txt")
    print("\nTo make predictions, use: python src/predict.py")

if __name__ == "__main__":
    main()
