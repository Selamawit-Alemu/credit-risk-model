# src/api/main.py

import os
import logging
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import PredictionRequest, PredictionResponse

# Allow relative import from src/
sys.path.append("src")

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize API
app = FastAPI(
    title="Credit Risk Prediction API",
    description="A production-grade API that predicts customer credit risk using a trained ML model and preprocessing pipeline.",
    version="1.0.0"
)

# Constants for file paths
MODEL_PATH = os.getenv("MODEL_PATH", "models/credit_model.pkl")
PIPELINE_PATH = os.getenv("PIPELINE_PATH", "models/preprocessing_pipeline.pkl")

# ---------------- Load Model ----------------
try:
    model = joblib.load(MODEL_PATH)
    logging.info(f"✅ Loaded model from '{MODEL_PATH}'")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    model = None

# ---------------- Load Preprocessing Pipeline ----------------
try:
    preprocessing_pipeline = joblib.load(PIPELINE_PATH)
    logging.info(f"✅ Loaded preprocessing pipeline from '{PIPELINE_PATH}'")
except Exception as e:
    logging.error(f"❌ Failed to load preprocessing pipeline: {e}")
    preprocessing_pipeline = None

# ---------------- Health Check Endpoint ----------------
@app.get("/")
def health_check():
    return {"status": "OK", "message": "Credit Risk API is running 🚀"}

# ---------------- Prediction Endpoint ----------------
@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    if preprocessing_pipeline is None:
        raise HTTPException(status_code=500, detail="Preprocessing pipeline is not loaded")

    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        logging.info(f"📥 Received input: {input_df.to_dict(orient='records')[0]}")

        # Drop any ID columns if present
        if "CustomerId" in input_df.columns:
            input_df.drop(columns=["CustomerId"], inplace=True)

        # Preprocess
        X_processed = preprocessing_pipeline.transform(input_df)

        # Predict probability and label
        risk_prob = model.predict_proba(X_processed)[0][1]
        is_high_risk = int(risk_prob > 0.5)

        logging.info(f"✅ Prediction: Risk Probability={risk_prob:.4f}, High Risk={is_high_risk}")

        return PredictionResponse(
            is_high_risk=is_high_risk,
            risk_probability=round(risk_prob, 4)
        )

    except Exception as e:
        logging.exception("❌ Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
