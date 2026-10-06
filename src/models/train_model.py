"""Train the Random Forest with the best parameters and save the model."""

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DATA_DIR = "data/processed_data"
MODELS_DIR = "models"
PARAMS_PATH = "params.yaml"


def load_training_data(data_dir):
    """Return the scaled training features and the training target."""
    X_train = pd.read_csv(f"{data_dir}/X_train_scaled.csv")
    # One-column DataFrame -> Series, expected by scikit-learn
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()

    return X_train, y_train


def load_params(path, section):
    """Return one section of the parameter file."""
    with open(path) as f:
        return yaml.safe_load(f)[section]


def train_model(X_train, y_train, best_params, random_state):
    """Train the Random Forest with the given parameters on the training data."""
    model = RandomForestRegressor(
        **best_params,  # Unpack params
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":
    X_train, y_train = load_training_data(PROCESSED_DATA_DIR)
    best_params = joblib.load(f"{MODELS_DIR}/best_params.pkl")
    random_state = load_params(PARAMS_PATH, "random_state")
    print(f"Training Random Forest with parameters: {best_params}")

    model = train_model(X_train, y_train, best_params, random_state)

    model_path = f"{MODELS_DIR}/random_forest_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
