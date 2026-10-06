"""Search the best Random Forest parameters with cross-validated grid search."""

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

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


def find_best_params(X_train, y_train, param_grid, cv, random_state):
    """Perform grid search to find the best Random Forest parameters."""
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=random_state),
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_


if __name__ == "__main__":
    X_train, y_train = load_training_data(PROCESSED_DATA_DIR)
    random_state = load_params(PARAMS_PATH, "random_state")
    params = load_params(PARAMS_PATH, "grid_search")

    best_params = find_best_params(
        X_train,
        y_train,
        param_grid=params["param_grid"],
        cv=params["cv"],
        random_state=random_state,
    )
    print(f"Best parameters found: {best_params}")

    params_path = f"{MODELS_DIR}/best_params.pkl"
    joblib.dump(best_params, params_path)
    print(f"Best parameters saved to {params_path}")
