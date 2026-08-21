"""Decision-threshold optimization for Bank Marketing classification."""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import (
    clean_bank_data,
    split_features_target,
)


RANDOM_STATE = 42
DATA_PATH = "data/raw/bank-additional-full.csv"


def build_model(X):
    """Build balanced Random Forest pipeline."""

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_thresholds(y_true, probabilities):
    """Evaluate precision/recall trade-off across thresholds."""

    rows = []

    for threshold in np.arange(0.20, 0.81, 0.05):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        rows.append(
            {
                "Threshold": round(float(threshold), 2),
                "Accuracy": accuracy_score(
                    y_true, predictions
                ),
                "Precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "F1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(rows)


def main():

    print("Loading Bank Marketing dataset...")

    df = pd.read_csv(DATA_PATH, sep=";")
    df = clean_bank_data(df)

    X, y = split_features_target(
        df,
        drop_duration=True,
    )

    # Train / validation / test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    print(f"Training samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_val)}")
    print(f"Test samples       : {len(X_test)}")

    pipeline = build_model(X_train)

    print("\nTraining balanced Random Forest...")
    pipeline.fit(X_train, y_train)

    val_probabilities = pipeline.predict_proba(
        X_val
    )[:, 1]

    threshold_results = evaluate_thresholds(
        y_val,
        val_probabilities,
    )

    print("\nVALIDATION THRESHOLD COMPARISON")
    print("=" * 70)
    print(
        threshold_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    best_row = threshold_results.loc[
        threshold_results["F1"].idxmax()
    ]

    best_threshold = float(
        best_row["Threshold"]
    )

    print(
        f"\nBest validation threshold by F1: "
        f"{best_threshold:.2f}"
    )

    # Evaluate selected threshold once on untouched test data.
    test_probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    print("\nFINAL TEST PERFORMANCE")
    print("=" * 70)
    print(
        f"Threshold : {best_threshold:.2f}"
    )
    print(
        f"Accuracy  : "
        f"{accuracy_score(y_test, test_predictions):.4f}"
    )
    print(
        f"Precision : "
        f"{precision_score(y_test, test_predictions):.4f}"
    )
    print(
        f"Recall    : "
        f"{recall_score(y_test, test_predictions):.4f}"
    )
    print(
        f"F1        : "
        f"{f1_score(y_test, test_predictions):.4f}"
    )


if __name__ == "__main__":
    main()