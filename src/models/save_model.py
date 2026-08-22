"""
Train and save the final Bank Marketing classification model.

This module:
1. Loads and cleans the Bank Marketing dataset.
2. Splits the data using stratified train/test sampling.
3. Builds the preprocessing and classification pipeline.
4. Trains a balanced Random Forest model.
5. Evaluates the model on the held-out test set.
6. Saves the complete fitted pipeline for future inference.
"""
from src.features.build_features import (
    clean_bank_data,
    split_features_target,
)
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
TEST_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-additional-full.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "bank_marketing_model.joblib"


def load_data():
    """Load cleaned and leakage-safe Bank Marketing data."""

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

    return X, y

def build_pipeline(X):
    """Create preprocessing and Random Forest pipeline."""

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             categorical_columns),
            ("numeric", "passthrough", numeric_columns),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test):
    """Evaluate the fitted model."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nFINAL MODEL PERFORMANCE")
    print("=" * 55)

    print(f"Accuracy  : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision : {precision_score(y_test, predictions):.4f}")
    print(f"Recall    : {recall_score(y_test, predictions):.4f}")
    print(f"F1        : {f1_score(y_test, predictions):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, probabilities):.4f}")
    print(f"PR-AUC    : {average_precision_score(y_test, probabilities):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


def main():
    """Train, evaluate and persist the final model."""

    X, y = load_data()

    print(f"Samples       : {len(X)}")
    print(f"Features      : {X.shape[1]}")
    print(f"Positive rate : {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")

    pipeline = build_pipeline(X)

    print("\nTraining final Random Forest model...")
    pipeline.fit(X_train, y_train)

    evaluate_model(
        pipeline,
        X_test,
        y_test,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    print("\nMODEL SAVED SUCCESSFULLY")
    print("=" * 55)
    print(f"Saved to: {MODEL_PATH}")
    print("The saved artifact contains preprocessing + model.")


if __name__ == "__main__":
    main()