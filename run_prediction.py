import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.models.predict_model import predict_from_csv
from src.features.feature_engineering import build_feature_pipeline

def main():
    df = pd.read_csv("data/processed/cleaned_transactions.csv")

    # Build and fit the pipeline on the full dataset
    pipeline = build_feature_pipeline()
    pipeline.fit(df.drop(columns=["is_high_risk", "TransactionId"], errors="ignore"))

    # Run predictions
    results = predict_from_csv(
        input_csv="data/processed/cleaned_transactions.csv",
        model_path="models/final_cv_model.pkl",
        pipeline=pipeline
    )

    print(results.head())
    results.to_csv("data/predictions/predictions.csv", index=False)
    print("✅ Predictions saved to data/predictions.csv")

if __name__ == "__main__":
    main()
