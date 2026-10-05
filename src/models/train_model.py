"""Train the Random Forest with the best parameters and save the model."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DATA_DIR = "data/processed_data"
MODELS_DIR = "models"


def load_training_data(data_dir):
    """Return the scaled training features and the training target."""
    X_train = pd.read_csv(f"{data_dir}/X_train_scaled.csv")
    # One-column DataFrame -> Series, expected by scikit-learn
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()

    return X_train, y_train


def train_model(X_train, y_train, best_params):
    """Train the Random Forest with the given parameters on the training data."""
    model = RandomForestRegressor(**best_params, random_state=42)  # Unpack params
    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":
    X_train, y_train = load_training_data(PROCESSED_DATA_DIR)
    best_params = joblib.load(f"{MODELS_DIR}/best_params.pkl")
    print(f"Training Random Forest with parameters: {best_params}")

    model = train_model(X_train, y_train, best_params)

    model_path = f"{MODELS_DIR}/random_forest_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
