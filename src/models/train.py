import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import joblib
import mlflow
import mlflow.sklearn

FEATURE_DATA_PATH = Path("data/processed/engine_features.csv")
MODEL_PATH = Path("models/model.joblib")

TARGET_COL = "will_fail_soon"
DROP_COLS = ["engine_id", "cycle", "RUL"]


def load_data():
    return pd.read_csv(FEATURE_DATA_PATH)


def split_by_engine(df, test_size=0.2, random_state=42):
    engine_ids = df["engine_id"].unique()
    train_engines, test_engines = train_test_split(
        engine_ids, test_size=test_size, random_state=random_state
    )
    return (
        df[df["engine_id"].isin(train_engines)],
        df[df["engine_id"].isin(test_engines)],
    )


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def main():
    mlflow.set_experiment("predictive_maintenance")

    with mlflow.start_run():
        print("Loading data...")
        df = load_data()

        train_df, test_df = split_by_engine(df)

        X_train = train_df.drop(columns=DROP_COLS + [TARGET_COL])
        y_train = train_df[TARGET_COL]

        X_test = test_df.drop(columns=DROP_COLS + [TARGET_COL])
        y_test = test_df[TARGET_COL]

        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("model_type", "RandomForest")

        print("Training model...")
        model = train_model(X_train, y_train)

        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred)

        mlflow.log_metric("recall_failure", recall)

        mlflow.sklearn.log_model(model, "model")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print(f"Recall (failure class): {recall:.4f}")
        print("Model and metrics logged to MLflow")


if __name__ == "__main__":
    main()
