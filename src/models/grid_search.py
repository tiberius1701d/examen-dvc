"""Search the best Random Forest parameters with cross-validated grid search."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

PROCESSED_DATA_DIR = "data/processed_data"
MODELS_DIR = "models"

# Grid search parameters for Random Forest (common starting point)
PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    # The minimum number of training rows that must end up in a leaf.
    # Higher values make the trees coarser and less prone to memorising noise.
    "min_samples_leaf": [1, 2, 4],
}


def load_training_data(data_dir):
    """Return the scaled training features and the training target."""
    X_train = pd.read_csv(f"{data_dir}/X_train_scaled.csv")
    # One-column DataFrame -> Series, expected by scikit-learn
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()

    return X_train, y_train


def find_best_params(X_train, y_train):
    """Perform grid search to find the best Random Forest parameters."""
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=PARAM_GRID,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_


if __name__ == "__main__":
    X_train, y_train = load_training_data(PROCESSED_DATA_DIR)

    best_params = find_best_params(X_train, y_train)
    print(f"Best parameters found: {best_params}")

    params_path = f"{MODELS_DIR}/best_params.pkl"
    joblib.dump(best_params, params_path)
    print(f"Best parameters saved to {params_path}")
