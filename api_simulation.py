"""
AgriTrust AI – API Simulation Layer
=====================================
Simulates the FastAPI microservice contract so the Streamlit app can
call predict() through the same interface it will use in production.

Production Deployment Plan:
  • Replace this module with a real FastAPI app (api_server.py)
  • Deploy behind an AWS API Gateway or GCP API Gateway
  • Horizontal scaling via Kubernetes + HPA on CPU / RPS metrics
  • Model artifact pulled from S3 on container startup (versioned)
  • Auth: API key header + rate-limiting middleware
  • Async endpoints (async def) with connection pooling

  Endpoint contract (matches this simulation exactly):

    POST /predict
    Headers: X-API-Key: <key>
    Body:
      {
        "farm_size":      float,
        "soil_score":     int,
        "rainfall":       float,
        "previous_loans": int,
        "yield_amount":   float,
        #"crop_diversity": int          (optional, default 2)
      }
    Response 200:
      {
        "trust_score":    float,       // 0–100
        "risk_category":  str,         // "Low Risk" | "Moderate Risk" | "High Risk" | "Very High Risk"
        "approved":       bool,
        "model_version":  str,
        "latency_ms":     float
      }

Regional Retraining Strategy:
  • Each state / agro-climatic zone maintains its own fine-tuned model
  • Base model retrained quarterly on fresh NABARD / FCI data
  • Regional deltas applied via transfer-learning adapters
  • A/B testing framework selects the champion model per zone

# TODO: Replace simulation with real FastAPI microservice deployment
"""

import time
import os
import joblib
import numpy as np
from dataclasses import dataclass, field

MODEL_PATH    = os.path.join("model", "credit_model.pkl")
MODEL_VERSION = "1.3.0-agritrust"

# Feature order must match training
FEATURE_ORDER = [
    "farm_size", "soil_score", "rainfall",
    "previous_loans", "yield_amount", #crop_diversity
]


@dataclass
class PredictRequest:
    farm_size:      float
    soil_score:     int
    rainfall:       float
    previous_loans: int
    yield_amount:   float
    #crop_diversity: int = field(default=2)


@dataclass
class PredictResponse:
    trust_score:   float
    risk_category: str
    approved:      bool
    model_version: str
    latency_ms:    float


def _load_model():
    """Load (and cache) the XGBoost model from disk."""
    if not hasattr(_load_model, "_model"):
        _load_model._model = joblib.load(MODEL_PATH)
    return _load_model._model


def _score_to_category(score: float) -> str:
    if score >= 70:
        return "Low Risk"
    elif score >= 50:
        return "Moderate Risk"
    elif score >= 30:
        return "High Risk"
    else:
        return "Very High Risk"


def predict(request: PredictRequest) -> PredictResponse:
    """
    Simulate POST /predict.
    Loads model, runs inference, converts raw probability → Trust Score.

    Trust Score formula:
      raw_prob (0–1) from XGBoost → scaled to 0–100
      with a soft S-curve to spread scores across the full range.

    # ================= FUTURE FEATURE =================
    # Add ensemble voting: XGBoost + LightGBM + CatBoost
    # Confidence interval via bootstrapped predictions
    # ===================================================
    """
    t0 = time.perf_counter()

    model = _load_model()

    X = np.array([[
        request.farm_size,
        request.soil_score,
        request.rainfall,
        request.previous_loans,
        request.yield_amount,
#request.crop_diversity,
    ]])

    prob = float(model.predict_proba(X)[0][1])

    # S-curve scaling: maps [0,1] → [0,100] with natural spread
    trust_score = round(100 / (1 + np.exp(-10 * (prob - 0.5))), 1)

    risk_category = _score_to_category(trust_score)
    approved      = trust_score >= 50
    latency_ms    = round((time.perf_counter() - t0) * 1000, 2)

    return PredictResponse(
        trust_score   = trust_score,
        risk_category = risk_category,
        approved      = approved,
        model_version = MODEL_VERSION,
        latency_ms    = latency_ms,
    )
