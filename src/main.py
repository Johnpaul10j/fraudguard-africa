from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
from datetime import datetime
import logging
import os

# ====================== LOGGING SETUP ======================
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ====================== APP SETUP ======================
app = FastAPI(
    title="FraudGuard Africa",
    description="Real-time Mobile Money Fraud Detection API",
    version="1.1.0"
)

# Load Model
try:
    model = joblib.load("models/fraudguard_xgboost_full.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# ====================== INPUT VALIDATION ======================
class Transaction(BaseModel):
    step: int = Field(..., ge=1, description="Time step of the transaction")
    type: str = Field(..., description="Type of transaction")
    amount: float = Field(..., gt=0, description="Transaction amount")
    nameOrig: str = Field(..., min_length=1)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    nameDest: str = Field(..., min_length=1)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)

    @validator("type")
    def validate_type(cls, v):
        allowed = ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"]
        if v.upper() not in allowed:
            raise ValueError(f"type must be one of {allowed}")
        return v.upper()

# ====================== PREDICTION FUNCTION ======================
def predict_fraud(data: dict):
    df = pd.DataFrame([data])

    # Feature Engineering
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["oldbalanceDest"] - df["newbalanceDest"]
    df["amount_to_oldbalance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1e-8)
    df["amount_to_newbalance_ratio"] = df["amount"] / (df["newbalanceOrig"] + 1e-8)
    df["hour"] = df["step"] % 24
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)

    df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=True)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols]

    probability = float(model.predict_proba(df)[0][1])
    is_fraud = bool(model.predict(df)[0])

    risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"

    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(probability, 6),
        "risk_level": risk_level,
        "confidence_percent": round(probability * 100, 2),
        "timestamp": datetime.now().isoformat()
    }

# ====================== ROUTES ======================

@app.get("/")
def home():
    return {
        "message": "FraudGuard Africa API is running",
        "version": "1.1.0",
        "status": "active"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        result = predict_fraud(transaction.dict())

        # Log the prediction
        logging.info(f"Prediction: {result} | Input: {transaction.dict()}")

        return result

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))