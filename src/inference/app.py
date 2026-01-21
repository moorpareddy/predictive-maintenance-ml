from fastapi import FastAPI, HTTPException
from pathlib import Path
import joblib
import pandas as pd

from src.inference.schema import EngineFeatures, PredictionResponse

# ------------------------------------------------------------------
# Model & Feature Configuration
# ------------------------------------------------------------------

MODEL_PATH = Path("models/model.joblib")

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
FEATURE_COLUMNS = []

for sensor in SENSOR_COLUMNS:
    FEATURE_COLUMNS.append(f"{sensor}_mean")
    FEATURE_COLUMNS.append(f"{sensor}_std")

EXPECTED_FEATURE_COUNT = len(FEATURE_COLUMNS)

# ------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------

app = FastAPI(
    title="Predictive Maintenance API",
    description="Production-grade failure prediction service for industrial machines",
    version="1.0.0",
)

# ------------------------------------------------------------------
# Startup: Load Model
# ------------------------------------------------------------------

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model file not found. Please train the model before starting the API."
        )
    model = joblib.load(MODEL_PATH)

# ------------------------------------------------------------------
# Health Check
# ------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ------------------------------------------------------------------
# Prediction Endpoint
# ------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: EngineFeatures):
    # Strict feature length validation
    if len(payload.features) != EXPECTED_FEATURE_COUNT:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {EXPECTED_FEATURE_COUNT} features, "
                f"but received {len(payload.features)}"
            ),
        )

    try:
        # Convert input to DataFrame with correct feature names
        X = pd.DataFrame(
            [payload.features],
            columns=FEATURE_COLUMNS
        )

        # Predict probability of failure
        probability = model.predict_proba(X)[0][1]
        prediction = int(probability >= 0.5)

        return PredictionResponse(
            will_fail_soon=prediction,
            probability=float(probability),
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
