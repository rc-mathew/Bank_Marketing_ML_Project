"""
Cross-validation experiment for the Bank Marketing ML project.

Uses Stratified K-Fold cross-validation to evaluate model stability
while preserving the target-class distribution in each fold.
"""

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import (
    clean_bank_data,
    split_features_target,
)


DATA_PATH = "data/raw/bank-additional-full.csv"
N_SPLITS = 5
RANDOM_STATE = 42


def load_data():
    """Load cleaned, leakage-safe model data."""

    print("Loading Bank Marketing dataset...")

    df = pd.read_csv(DATA_PATH, sep=";")
    df = clean_bank_data(df)

    X, y = split_features_target(
        df,
        drop_duration=True,
    )

    return X, y

def build_pipeline(X):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
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
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=3,
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


def run_cross_validation(X, y):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    print(f"\nRunning {N_SPLITS}-fold Stratified Cross-Validation...\n")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        pipeline = build_pipeline(X)

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_val)[:, 1]

        # Use threshold selected in the previous experiment
        predictions = (probabilities >= 0.55).astype(int)

        fold_results = {
            "Fold": fold,
            "Accuracy": accuracy_score(y_val, predictions),
            "ROC-AUC": roc_auc_score(y_val, probabilities),
            "PR-AUC": average_precision_score(y_val, probabilities),
            "Recall": recall_score(y_val, predictions),
            "F1": f1_score(y_val, predictions),
        }

        results.append(fold_results)

        print(
            f"Fold {fold}: "
            f"ROC-AUC={fold_results['ROC-AUC']:.4f} | "
            f"PR-AUC={fold_results['PR-AUC']:.4f} | "
            f"Recall={fold_results['Recall']:.4f} | "
            f"F1={fold_results['F1']:.4f}"
        )

    return pd.DataFrame(results)


def summarize_results(results):
    print("\n" + "=" * 65)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 65)

    print(results.to_string(index=False))

    print("\n" + "=" * 65)
    print("MEAN ± STANDARD DEVIATION")
    print("=" * 65)

    metrics = ["Accuracy", "ROC-AUC", "PR-AUC", "Recall", "F1"]

    for metric in metrics:
        mean = results[metric].mean()
        std = results[metric].std()

        print(f"{metric:<10}: {mean:.4f} ± {std:.4f}")

    print("\nModel stability check:")
    print(f"ROC-AUC range: {results['ROC-AUC'].min():.4f} - "
          f"{results['ROC-AUC'].max():.4f}")

    print(f"PR-AUC range : {results['PR-AUC'].min():.4f} - "
          f"{results['PR-AUC'].max():.4f}")


def main():
    X, y = load_data()

    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Positive rate: {y.mean():.4f}")

    results = run_cross_validation(X, y)
    summarize_results(results)


if __name__ == "__main__":
    main()