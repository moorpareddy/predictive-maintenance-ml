import pandas as pd
import requests

df = pd.read_csv("data/processed/engine_features.csv")

FEATURE_COLUMNS = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]

sample = df.iloc[0][FEATURE_COLUMNS].tolist()

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"features": sample}
)

print(response.json())
