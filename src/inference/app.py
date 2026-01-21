from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pathlib import Path

from src.inference.schema import EngineFeatures, PredictionResponse

MODEL_PATH = Path("models/model.joblib")

app = FastAPI(
    title="Predictive Maintenance API",
    description="Failure prediction for industrial machines",
    version="1.0.0",
)


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError("Model file not found. Train the model first.")
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: EngineFeatures):
    try:
        X = np.array(payload.features).reshape(1, -1)
        prob = model.predict_proba(X)[0][1]
        prediction = int(prob >= 0.5)

        return PredictionResponse(
            will_fail_soon=prediction,
            probability=float(prob),
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
