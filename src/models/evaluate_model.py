"""Evaluate the trained model on the test set and save predictions and scores."""

import json

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

PROCESSED_DATA_DIR = "data/processed_data"
MODELS_DIR = "models"
DATA_DIR = "data"
METRICS_DIR = "metrics"


def load_test_data(data_dir):
    """Return the scaled test features and the test target."""
    X_test = pd.read_csv(f"{data_dir}/X_test_scaled.csv")
    # One-column DataFrame -> Series, expected by scikit-learn
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()

    return X_test, y_test


def compute_scores(y_test, predictions):
    """Return the MSE and the R2 score of the predictions."""
    return {
        "mse": mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }


if __name__ == "__main__":
    X_test, y_test = load_test_data(PROCESSED_DATA_DIR)
    model = joblib.load(f"{MODELS_DIR}/random_forest_model.pkl")

    predictions = model.predict(X_test)
    scores = compute_scores(y_test, predictions)
    print(f"Test scores: {scores}")

    predictions_path = f"{DATA_DIR}/predictions.csv"
    # Merge ground truth and predictions into one DataFrame
    predictions_df = pd.DataFrame({"actual": y_test, "predicted": predictions})
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    scores_path = f"{METRICS_DIR}/scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=4)
    print(f"Scores saved to {scores_path}")
