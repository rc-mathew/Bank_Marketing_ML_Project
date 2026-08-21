"""
Ensemble model training for the Bank Marketing ML project.

Models:
- Random Forest
- Gradient Boosting

The models are evaluated using ROC-AUC and PR-AUC,
which are useful metrics for imbalanced classification.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import (
    clean_bank_data,
    split_features_target,
)


DATA_PATH = "data/raw/bank-additional-full.csv"
RANDOM_STATE = 42


def load_data():
    """Load and clean the Bank Marketing dataset."""
    df = pd.read_csv(DATA_PATH, sep=";")
    df = clean_bank_data(df)
    return df


def prepare_data(df):
    """Create leakage-safe features and perform stratified split."""

    X, y = split_features_target(
        df,
        drop_duration=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test

    


def build_preprocessor(X):
    """Build preprocessing pipeline for numeric and categorical data."""

    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
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
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def evaluate_model(name, model, X_test, y_test):
    """Evaluate a fitted classification model."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
    }


def main():

    print("Loading Bank Marketing dataset...")

    df = load_data()

    X_train, X_test, y_train, y_test = prepare_data(df)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    results = []

    for name, classifier in models.items():

        print(f"\nTraining {name}...")

        preprocessor = build_preprocessor(X_train)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        result = evaluate_model(
            name,
            pipeline,
            X_test,
            y_test,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print("\nMODEL COMPARISON")
    print("=" * 60)
    print(
        results_df.sort_values(
            by="ROC-AUC",
            ascending=False,
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()