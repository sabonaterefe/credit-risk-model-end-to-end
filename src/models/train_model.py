import os
import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    confusion_matrix, roc_curve
)
from xgboost import XGBClassifier

def get_model():
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42
    )

def evaluate_model(X, y, model, log_to_mlflow=True, save_model=True):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_val, y_proba),
        "f1": f1_score(y_val, y_pred),
        "accuracy": accuracy_score(y_val, y_pred)
    }

    print("\n📊 Validation Metrics:")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    # ROC Curve
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

    if log_to_mlflow:
        mlflow.set_experiment("credit-risk-model")
        with mlflow.start_run():
            mlflow.log_params(model.get_params())
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_artifact("models/confusion_matrix.png")
            mlflow.log_artifact("models/roc_curve.png")
            mlflow.xgboost.log_model(model, "xgb_model")

    if save_model:
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/final_cv_model.pkl")
        print("✅ Final model saved to models/final_cv_model.pkl")

    return model, metrics
