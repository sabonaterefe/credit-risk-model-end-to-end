import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    confusion_matrix, roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# 🔧 Build preprocessing + model pipeline
def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0,
        reg_lambda=0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

# 🧪 Train and evaluate
def train_and_evaluate(data_path="data/processed/cleaned_transactions.csv"):
    df = pd.read_csv(data_path)

    # ✅ Define binary target robustly
    if df["FraudResult"].dtype == object:
        df["is_high_risk"] = (df["FraudResult"].str.lower() == "fraud").astype(int)
    else:
        df["is_high_risk"] = df["FraudResult"].astype(int)

    # 🔍 Confirm class distribution
    class_counts = df["is_high_risk"].value_counts()
    print("📊 Class distribution before SMOTE:", class_counts.to_dict())

    if class_counts.nunique() < 2:
        raise ValueError("❌ Not enough positive samples to train. Check your labels.")

    # 🎯 Feature engineering
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce")
    df["Hour"] = df["TransactionStartTime"].dt.hour
    df["DayOfWeek"] = df["TransactionStartTime"].dt.dayofweek
    df["AmountToValueRatio"] = df["Amount"] / (df["Value"] + 1)
    df["IsNightTransaction"] = df["Hour"].apply(lambda h: 1 if h < 5 else 0)
    df = df.drop(columns=["TransactionStartTime"])

    numeric_features = ["Amount", "Value", "Hour", "DayOfWeek", "AmountToValueRatio", "IsNightTransaction"]
    categorical_features = ["ProductCategory", "ChannelId", "ProviderId", "CustomerId"]
    features = numeric_features + categorical_features
    target = "is_high_risk"

    # Reset index to avoid alignment issues
    df = df.reset_index(drop=True)
    X = df[features]
    y = df[target]

    # One-hot encode for SMOTE
    X_encoded = pd.get_dummies(X, columns=categorical_features)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

    print("✅ Class distribution after SMOTE:", pd.Series(y_resampled).value_counts().to_dict())

    # Train-test split on resampled data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train pipeline on original features
    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    print("📈 Probability range:", np.min(y_proba), "to", np.max(y_proba))
    print("📈 Sample probabilities:", y_proba[:10])

    metrics = {
        "roc_auc": roc_auc_score(y_val, y_proba),
        "f1": f1_score(y_val, y_pred),
        "accuracy": accuracy_score(y_val, y_pred)
    }

    print("\n📊 Validation Metrics:")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    os.makedirs("models", exist_ok=True)

    # 📊 Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    # 📉 ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/roc_curve.png")
    plt.close()

    # 📈 SHAP Summary
    print("📈 Generating SHAP summary plot...")
    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    X_val_transformed = preprocessor.transform(X_val)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_val_transformed)

    shap.summary_plot(shap_values, X_val_transformed, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png")
    plt.close()

    # 💾 Save pipeline
    joblib.dump(pipeline, "models/fitted_pipeline.pkl")
    print("✅ Full pipeline saved to models/fitted_pipeline.pkl")

    return pipeline, metrics

# 🏁 Run training
if __name__ == "__main__":
    train_and_evaluate()
