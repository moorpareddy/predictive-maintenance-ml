import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/train_FD001.txt")
PROCESSED_DATA_PATH = Path("data/processed/engine_data.csv")


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw CMAPSS data from text file."""
    df = pd.read_csv(
        path,
        sep=" ",
        header=None
    )

    # Drop empty columns caused by extra spaces
    df = df.dropna(axis=1)

    return df


def add_column_names(df: pd.DataFrame) -> pd.DataFrame:
    columns = (
        ["engine_id", "cycle"]
        + [f"op_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )
    df.columns = columns
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add Remaining Useful Life (RUL) column."""
    max_cycles = df.groupby("engine_id")["cycle"].max()
    df["RUL"] = df.apply(
        lambda row: max_cycles[row["engine_id"]] - row["cycle"],
        axis=1
    )
    return df


def add_failure_label(df: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
    """Binary label: failure within N cycles."""
    df["will_fail_soon"] = (df["RUL"] <= threshold).astype(int)
    return df


def main():
    print("Loading raw data...")
    df = load_raw_data(RAW_DATA_PATH)
    df = add_column_names(df)

    print("Adding RUL and failure label...")
    df = add_rul(df)
    df = add_failure_label(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed data saved to {PROCESSED_DATA_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
