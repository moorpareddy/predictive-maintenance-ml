import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("logs/inference_logs.jsonl")


def log_inference(features, prediction, probability):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "features": features,
        "prediction": int(prediction),
        "probability": float(probability),
    }

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
