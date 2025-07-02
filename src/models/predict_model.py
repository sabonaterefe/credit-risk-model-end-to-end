import pandas as pd
import joblib

# 🔍 Load a trained model from disk
def load_model(path: str):
    return joblib.load(path)

# 🔮 Predict risk for a single transformed input
def predict_risk(model, X):
    proba = model.predict_proba(X)[:, 1]
    label = (proba > 0.5).astype(int)
    return label, proba

# 📂 Predict from a CSV file (used in CLI or scripts)
def predict_from_csv(input_csv: str, model_path: str, pipeline):
    df = pd.read_csv(input_csv)
    X = df.drop(columns=["is_high_risk", "TransactionId"], errors="ignore")
    X_transformed = pipeline.transform(X)

    model = load_model(model_path)
    predictions = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed)[:, 1]

    df["predicted_label"] = predictions
    df["risk_probability"] = probabilities
    return df[["CustomerId", "predicted_label", "risk_probability"]]

# 📊 Batch prediction from a DataFrame (used in API)
def predict_batch(df: pd.DataFrame, model, pipeline):
    X = df.drop(columns=["is_high_risk", "TransactionId"], errors="ignore")
    X_transformed = pipeline.transform(X)

    predictions = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed)[:, 1]

    df["predicted_label"] = predictions
    df["risk_probability"] = probabilities
    return df
