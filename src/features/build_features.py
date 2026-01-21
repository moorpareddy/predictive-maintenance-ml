import pandas as pd
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed/engine_data.csv")
FEATURE_DATA_PATH = Path("data/processed/engine_features.csv")

ROLLING_WINDOW = 10
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["engine_id", "cycle"])

    feature_frames = []

    for engine_id, engine_df in df.groupby("engine_id"):
        engine_df = engine_df.copy()

        for sensor in SENSOR_COLUMNS:
            engine_df[f"{sensor}_mean"] = (
                engine_df[sensor]
                .rolling(window=ROLLING_WINDOW, min_periods=1)
                .mean()
            )

            engine_df[f"{sensor}_std"] = (
                engine_df[sensor]
                .rolling(window=ROLLING_WINDOW, min_periods=1)
                .std()
                .fillna(0.0)
            )

        feature_frames.append(engine_df)

    return pd.concat(feature_frames, axis=0)


def main():
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    print("Building rolling window features...")
    df_features = build_rolling_features(df)

    FEATURE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(FEATURE_DATA_PATH, index=False)

    print(f"Feature data saved to {FEATURE_DATA_PATH}")
    print(df_features.head())


if __name__ == "__main__":
    main()
