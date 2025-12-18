from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "global_best_model_optuna.pkl"

app = FastAPI(
    title="Accident Severity Prediction API",
    description="FastAPI service for predicting severity of accidents",
    version="1.0.0",
)

def load_model(path: Path):
    """Load the trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    print(f"Loading model from: {path}")
    m = joblib.load(path)
    print("✓ Model loaded successfully!")
    print(f"  Model type: {type(m).__name__}")
    if hasattr(m, "named_steps"):
        print(f"  Pipeline steps: {list(m.named_steps.keys())}")
    return m


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"✗ ERROR: Failed to load model from {MODEL_PATH}")
    print(f"  Error: {e}")
    raise RuntimeError(f"Failed to load model: {e}")


class PredictRequest(BaseModel):
    """
    Prediction request with list of instances (dicts of features).
    """
    instances: List[Dict[str, Any]]

    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    {
                        "hour": 22,
                        "month": 1,
                        "is_weekend": 0,
                        "is_night": 1,

                        "state": "CA",
                        "latitude": 34.05,
                        "longitude": -118.24,

                        "temperature_f": 55.0,
                        "visibility_mi": 10.0,
                        "wind_speed_mph": 5.0,
                        "precipitation_in": 0.0,
                        "weather_condition": "Clear",

                        "junction": 0,
                        "traffic_signal": 1,
                        "crossing": 0,
                        "stop": 0,
                        "railway": 0,
                        "roundabout": 0,
                        "bump": 0
                    }
                ]
            }
        }


class PredictResponse(BaseModel):
    predictions: List[float]
    count: int

    class Config:
        schema_extra = {
            "example": {
                "predictions": [452600.0],
                "count": 1,
            }
        }


@app.get("/")
def root():
    return {
        "name": "Accident Severity Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": str(model is not None),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(
            status_code=400,
            detail="No instances provided.",
        )

    try:
        X = pd.DataFrame(request.instances)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input format: {e}",
        )
    try:
        preds = model.predict(X) + 1
        probs = model.predict_proba(X)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}",
        )

    return {
        "predictions": preds.tolist(),
        "probabilities": probs.tolist(),
        "count": len(preds),
    }


@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 80)
    print("Housing Price Prediction API - Starting Up")
    print("=" * 80)
    print(f"Model path: {MODEL_PATH}")
    print(f"Model loaded: {model is not None}")
    print("API is ready to accept requests!")
    print("=" * 80 + "\n")