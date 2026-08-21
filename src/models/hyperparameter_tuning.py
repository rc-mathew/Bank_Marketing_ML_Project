"""
Hyperparameter tuning for the Bank Marketing ML project.

Uses RandomizedSearchCV with Stratified K-Fold cross-validation
and Average Precision (PR-AUC) as the primary optimization metric.
"""

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import (
    clean_bank_data,
    split_features_target,
)


DATA_PATH = "data/raw/bank-additional-full.csv"
RANDOM_STATE = 42


def build_pipeline(X):
    """Create preprocessing and Random Forest pipeline."""

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
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
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(
    name,
    model,
    X_test,
    y_test,
    threshold=0.50,
):
    """Evaluate model using imbalance-aware metrics."""

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print(
        f"Accuracy  : "
        f"{accuracy_score(y_test, predictions):.4f}"
    )
    print(
        f"Precision : "
        f"{precision_score(y_test, predictions):.4f}"
    )
    print(
        f"Recall    : "
        f"{recall_score(y_test, predictions):.4f}"
    )
    print(
        f"F1        : "
        f"{f1_score(y_test, predictions):.4f}"
    )
    print(
        f"ROC-AUC   : "
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )
    print(
        f"PR-AUC    : "
        f"{average_precision_score(y_test, probabilities):.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )


def main():

    print("Loading Bank Marketing dataset...")

    df = pd.read_csv(
        DATA_PATH,
        sep=";",
    )

    df = clean_bank_data(df)

    X, y = split_features_target(
        df,
        drop_duration=True,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline(X_train)

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    param_distributions = {
        "model__n_estimators": [
            100,
            150,
            200,
            250,
        ],
        "model__max_depth": [
            8,
            10,
            12,
            15,
            None,
        ],
        "model__min_samples_split": [
            2,
            5,
            10,
        ],
        "model__min_samples_leaf": [
            2,
            3,
            5,
            8,
        ],
        "model__max_features": [
            "sqrt",
            "log2",
            0.5,
        ],
        "model__class_weight": [
            "balanced",
            "balanced_subsample",
        ],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=8,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        refit=True,
    )

    print("\nRunning RandomizedSearchCV...")
    print("Optimization metric: PR-AUC")
    print("Search iterations: 8")
    print("Cross-validation folds: 3\n")

    search.fit(
        X_train,
        y_train,
    )

    print("\n" + "=" * 65)
    print("BEST HYPERPARAMETERS")
    print("=" * 65)

    for parameter, value in search.best_params_.items():
        print(f"{parameter}: {value}")

    print(
        f"\nBest CV PR-AUC: "
        f"{search.best_score_:.4f}"
    )

    best_model = search.best_estimator_

    evaluate_model(
        "Tuned Random Forest",
        best_model,
        X_test,
        y_test,
        threshold=0.55,
    )


if __name__ == "__main__":
    main()