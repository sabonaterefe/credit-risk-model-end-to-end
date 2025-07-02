import os
import sys
import pandas as pd
import joblib
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.features.feature_engineering import build_feature_pipeline
from src.features.rfm_target import create_rfm_features, assign_risk_label
from src.models.train_model import get_model, evaluate_model

def main():
    # 📥 Load dataset
    df = pd.read_csv("data/processed/cleaned_transactions.csv")

    # 🧠 Generate RFM-based risk labels
    snapshot_date = pd.to_datetime(df["TransactionStartTime"]).max().tz_localize(None)
    rfm = create_rfm_features(df, snapshot_date)
    risk_labels = assign_risk_label(rfm)
    df = df.merge(risk_labels, on="CustomerId", how="left")

    # 🎯 Define features and target
    y = df["is_high_risk"]
    X = df.drop(columns=["is_high_risk", "TransactionId"], errors="ignore")

    # 🛠️ Build and fit the feature pipeline
    pipeline = build_feature_pipeline()
    X_transformed = pipeline.fit_transform(X)

    # 🤖 Train and evaluate the model
    model = get_model()
    evaluate_model(pd.DataFrame(X_transformed), y, model)

    # 💾 Save the fitted pipeline
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/fitted_pipeline.pkl")
    print("✅ Fitted pipeline saved to models/fitted_pipeline.pkl")

if __name__ == "__main__":
    main()
