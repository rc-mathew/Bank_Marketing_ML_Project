"""
Feature engineering utilities for the Bank Marketing ML project.

Key decisions:
- Remove duplicate observations.
- Convert the target y from yes/no to 1/0.
- Treat pdays=999 as "client was not previously contacted".
- Create an explicit previous_contact indicator.
- Allow duration to be removed for deployment-safe modelling.
"""

import pandas as pd


def clean_bank_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Bank Marketing data."""

    df = df.copy()

    # Remove duplicate observations
    df = df.drop_duplicates().reset_index(drop=True)

    # Convert target to binary
    if "y" in df.columns:
        df["y"] = df["y"].map({"no": 0, "yes": 1})

    # pdays=999 means the client was not previously contacted
    if "pdays" in df.columns:
        df["previously_contacted"] = (df["pdays"] != 999).astype(int)

        # Replace sentinel value with missing value
        df["pdays"] = df["pdays"].replace(999, pd.NA)
        # Normalize categorical missing values for sklearn compatibility
        categorical_cols = df.select_dtypes(include=["object", "string"]).columns

        for col in categorical_cols:
            df[col] = df[col].astype("object")
            df[col] = df[col].where(df[col].notna(), "unknown")
        # Ensure categorical columns have consistent string dtype
        categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns

        for col in categorical_cols:
           df[col] = df[col].astype(str)

    return df


def split_features_target(
    df: pd.DataFrame,
    drop_duration: bool = True
):
    """
    Separate predictors and target.

    duration is unavailable before a marketing call is completed,
    so it is excluded by default from the deployment-safe model.
    """

    if "y" not in df.columns:
        raise ValueError("Target column 'y' not found.")

    X = df.drop(columns=["y"]).copy()
    y = df["y"].copy()

    if drop_duration and "duration" in X.columns:
        X = X.drop(columns=["duration"])

    return X, y


def get_feature_groups(X: pd.DataFrame):
    """Return categorical and numerical feature names."""

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    return categorical_features, numerical_features