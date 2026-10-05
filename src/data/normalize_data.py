"""Standardize the training and test features to mean 0 and standard deviation 1."""

import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_DATA_DIR = "data/processed_data"


def load_data(path):
    """Load data from a CSV file and return a pandas DataFrame."""
    return pd.read_csv(path)


def standardize_data(X_train, X_test):
    """Fit the scaler on the training data and apply it to both sets."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_scaled, X_test_scaled


if __name__ == "__main__":
    X_train = load_data(f"{PROCESSED_DATA_DIR}/X_train.csv")
    X_test = load_data(f"{PROCESSED_DATA_DIR}/X_test.csv")

    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    X_train_scaled.to_csv(f"{PROCESSED_DATA_DIR}/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(f"{PROCESSED_DATA_DIR}/X_test_scaled.csv", index=False)
    print(f"Scaled features saved to {PROCESSED_DATA_DIR}")
