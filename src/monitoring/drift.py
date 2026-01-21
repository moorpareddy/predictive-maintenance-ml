import json
from pathlib import Path
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


TRAIN_DATA_PATH = Path("data/processed/engine_features.csv")
INFERENCE_LOG_PATH = Path("logs/inference_logs.jsonl")
REPORT_PATH = Path("reports/data_drift.html")


def load_inference_data():
    records = []
    with open(INFERENCE_LOG_PATH, "r") as f:
        for line in f:
            entry = json.loads(line)
            records.append(entry["features"])
    return pd.DataFrame(records)


def main():
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    feature_cols = [
        c for c in train_df.columns
        if c.endswith("_mean") or c.endswith("_std")
    ]
    train_df = train_df[feature_cols]

    print("Loading inference data...")
    prod_df = load_inference_data()
    prod_df.columns = feature_cols

    print("Running drift detection...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=train_df,
        current_data=prod_df,
    )

    REPORT_PATH.parent.mkdir(exist_ok=True)
    report.save_html(str(REPORT_PATH))
    print(f"✅ Drift report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
