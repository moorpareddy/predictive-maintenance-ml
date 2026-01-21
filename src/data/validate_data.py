import pandas as pd
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed/engine_data.csv")


def validate_schema(df: pd.DataFrame):
    required_columns = {
        "engine_id",
        "cycle",
        "RUL",
        "will_fail_soon"
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_missing_values(df: pd.DataFrame):
    missing_ratio = df.isnull().mean()
    if missing_ratio.max() > 0.05:
        raise ValueError("Too many missing values in dataset")


def validate_ranges(df: pd.DataFrame):
    if (df["cycle"] <= 0).any():
        raise ValueError("Invalid cycle values detected")

    if (df["RUL"] < 0).any():
        raise ValueError("Negative RUL detected")

    if not set(df["will_fail_soon"].unique()).issubset({0, 1}):
        raise ValueError("Invalid target labels detected")


def main():
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    print("Running data validation checks...")
    validate_schema(df)
    validate_missing_values(df)
    validate_ranges(df)

    print("✅ Data validation passed successfully.")


if __name__ == "__main__":
    main()
