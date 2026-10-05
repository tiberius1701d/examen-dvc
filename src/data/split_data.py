"""Split the raw data into training and test sets."""

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = "data/raw_data"
PROCESSED_DATA_DIR = "data/processed_data"


def load_data(path):
    """Load data from a CSV file and return a pandas DataFrame."""
    return pd.read_csv(path)


def check_data(df):
    """Stop the pipeline if the raw data is not as expected."""
    assert not df.isna().any().any(), "Data contains missing values"
    assert not df.duplicated().any(), "Data contains duplicate rows"


def get_train_test_split_data(df, test_size=0.2, random_state=42):
    """Split the DataFrame into training and test sets."""
    X = df.drop(["date", "silica_concentrate"], axis=1)
    y = df["silica_concentrate"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    df = load_data(f"{RAW_DATA_DIR}/raw.csv")
    check_data(df)

    X_train, X_test, y_train, y_test = get_train_test_split_data(df)
    print(
        f"Split {len(df)} rows into {len(X_train)} training and {len(X_test)} test rows"
    )

    X_train.to_csv(f"{PROCESSED_DATA_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DATA_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DATA_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DATA_DIR}/y_test.csv", index=False)
    print(f"Training and test sets saved to {PROCESSED_DATA_DIR}")
