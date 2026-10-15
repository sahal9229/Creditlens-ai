"""
main.py
-------
FastAPI backend for Credit Card Default Prediction.

Run:  uvicorn main:app --reload --port 8000
"""

import os
import subprocess
import sys
import warnings

# Suppress CUDA check errors on CPU-only cloud instances (Render)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model


# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "credit_ann_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_model.py")


# ── Pydantic request schema ──────────────────────────────────────────
class CreditFeatures(BaseModel):
    """23 input features matching the UCI dataset columns."""
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


# ── App initialisation ───────────────────────────────────────────────
app = FastAPI(
    title="Credit Card Default Prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global references (loaded once at startup)
model = None
scaler = None


@app.on_event("startup")
async def startup_event():
    """Load model artifacts; train first if they don't exist."""
    global model, scaler

    # Smart fallback: auto-train if artifacts are missing
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("[!] Model artifacts not found -- running train_model.py ...")
        subprocess.run(
            [sys.executable, TRAIN_SCRIPT],
            check=True,
            cwd=BASE_DIR,
        )

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[OK] Model and scaler loaded successfully.")


# ── Prediction endpoint ──────────────────────────────────────────────
@app.post("/predict")
async def predict(features: CreditFeatures):
    """
    Accept 23 credit features, return default probability & classification.
    """
    # Build the feature array in the exact column order the model expects
    feature_array = np.array([[
        features.LIMIT_BAL,
        features.SEX,
        features.EDUCATION,
        features.MARRIAGE,
        features.AGE,
        features.PAY_0,
        features.PAY_2,
        features.PAY_3,
        features.PAY_4,
        features.PAY_5,
        features.PAY_6,
        features.BILL_AMT1,
        features.BILL_AMT2,
        features.BILL_AMT3,
        features.BILL_AMT4,
        features.BILL_AMT5,
        features.BILL_AMT6,
        features.PAY_AMT1,
        features.PAY_AMT2,
        features.PAY_AMT3,
        features.PAY_AMT4,
        features.PAY_AMT5,
        features.PAY_AMT6,
    ]])

    # Scale & predict
    scaled = scaler.transform(feature_array)
    probability = float(model.predict(scaled, verbose=0)[0][0])

    prediction = 1 if probability >= 0.5 else 0
    status = "High Risk (Default Likely)" if prediction == 1 else "Low Risk (Creditworthy)"

    return {
        "default_probability": round(probability, 6),
        "default_probability_pct": f"{probability * 100:.2f}%",
        "prediction": prediction,
        "status": status,
    }


# ── Serve frontend ───────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.api_route("/", methods=["GET", "HEAD"])
async def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
