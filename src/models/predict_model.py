import os
import sys
import pandas as pd
import joblib
import shap

# ✅ Ensure project root is in sys.path so 'src' is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 🔧 Load the trained pipeline
def load_pipeline(path: str = "models/fitted_pipeline.pkl"):
    return joblib.load(path)

# 🧠 Classify risk level based on probability
def classify_risk_band(probability: float) -> str:
    if probability >= 0.6:
        return "High"
    elif probability >= 0.2:
        return "Medium"
    else:
        return "Low"

# 🧮 Feature engineering to match training pipeline
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TransactionStartTime" in df.columns:
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce")
        df["Hour"] = df["TransactionStartTime"].dt.hour
        df["DayOfWeek"] = df["TransactionStartTime"].dt.dayofweek
        df.drop(columns=["TransactionStartTime"], inplace=True)
    df["AmountToValueRatio"] = df["Amount"] / (df["Value"] + 1)
    df["IsNightTransaction"] = df["Hour"].apply(lambda h: 1 if h < 5 else 0)
    return df

# 🔮 Predict risk and explain with SHAP
def predict_risk(pipeline, input_df: pd.DataFrame):
    df = engineer_features(input_df)

    # Predict
    proba = pipeline.predict_proba(df)[:, 1]
    label = int(proba[0] > 0.5)
    risk_band = classify_risk_band(proba[0])

    # SHAP explanation
    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(df)

    # ✅ Recover feature names
    num_features = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1]
    cat_features = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
    feature_names = list(num_features) + list(cat_features)

    explainer = shap.Explainer(model, feature_names=feature_names)
    shap_values = explainer(X_transformed)

    shap_array = shap_values[0].values
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_array
    }).sort_values(by="shap_value", key=abs, ascending=False)

    top_features = shap_df.head(3).to_dict(orient="records")

    return label, proba[0], risk_band, top_features

# ▶️ CLI test entry point
if __name__ == "__main__":
    print("🔍 Loading pipeline and running test prediction...")

    try:
        pipeline = load_pipeline()

        # Sample high-risk input
        sample_input = pd.DataFrame([{
            "Amount": 95000.0,
            "Value": 10.0,
            "ProductCategory": "loan",
            "ChannelId": "ChannelId_2",
            "ProviderId": "ProviderId_3",
            "CustomerId": "CustomerId_1999",
            "TransactionStartTime": "2018-11-15 03:12:00+00:00"
        }])

        label, proba, risk_band, top_features = predict_risk(pipeline, sample_input)

        print("\n✅ Prediction Result:")
        print(f"Predicted Label     : {label}")
        print(f"Risk Probability    : {proba:.6f}")
        print(f"Risk Band           : {risk_band}")
        print("\n🔍 Top Contributing Features:")
        for feat in top_features:
            print(f"- {feat['feature']}: SHAP = {feat['shap_value']:.4f}")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
