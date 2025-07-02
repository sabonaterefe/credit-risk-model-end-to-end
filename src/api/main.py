from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Header
from fastapi.responses import JSONResponse
import os
import sys
import pandas as pd
import joblib
from src.api.pydantic_models import CustomerInput, RiskPrediction
from src.models.predict_model import load_model, predict_risk, predict_batch

# 📦 Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ✅ Load trained model and pipeline
MODEL_PATH = "models/final_cv_model.pkl"
PIPELINE_PATH = "models/fitted_pipeline.pkl"
LOG_PATH = "logs/predictions_log.csv"
API_KEY = "supersecretkey"  # 🔐 Replace with env var in production

if not os.path.exists(MODEL_PATH) or not os.path.exists(PIPELINE_PATH):
    raise FileNotFoundError("Model or pipeline not found. Please run training first.")

model = load_model(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

# 🔀 Create API router
router = APIRouter()

# 🔐 Auth helper
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# 🏠 Root endpoint
@router.get("/")
def root(x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    return {"message": "🚀 Credit Risk API is running. Visit /api/docs to test the model."}

# 🔮 Single prediction
@router.post("/predict", response_model=RiskPrediction)
def predict(data: CustomerInput, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    try:
        df = data.to_df()
        X = pipeline.transform(df)
        _, proba = predict_risk(model, X)

        # 📝 Log prediction
        log_df = df.copy()
        log_df["risk_probability"] = proba[0]
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        log_df.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)

        return RiskPrediction(risk_probability=proba[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# 📂 Batch prediction
@router.post("/predict_batch")
def predict_batch_endpoint(file: UploadFile = File(...), x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    try:
        df = pd.read_csv(file.file)
        results = predict_batch(df, model, pipeline)

        # 📝 Log batch predictions
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        results.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)

        return {
            "message": "✅ Batch predictions completed",
            "rows": len(results),
            "columns": list(results.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# 🚀 Create FastAPI app and mount router
app = FastAPI(title="Credit Risk API", version="1.0")
app.include_router(router, prefix="/api")
