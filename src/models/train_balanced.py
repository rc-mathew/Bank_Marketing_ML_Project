"""Class imbalance experiments for Bank Marketing classification."""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    roc_auc_score,
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


def build_pipeline(class_weight=None):
    """Build preprocessing and Random Forest pipeline."""

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

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return numeric_pipeline, categorical_pipeline, model


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    """Train model and report imbalance-aware metrics."""

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

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

    df = pd.read_csv(DATA_PATH, sep=";")
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

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Positive rate: {y.mean():.4f}")

    results = []

    for name, class_weight in [
        ("Standard Random Forest", None),
        ("Balanced Random Forest", "balanced"),
    ]:

        numeric_pipeline, categorical_pipeline, model = build_pipeline(
            class_weight
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        results.append(
            evaluate_model(
                name,
                pipeline,
                X_train,
                X_test,
                y_train,
                y_test,
            )
        )

    results_df = pd.DataFrame(results)

    print("\nMODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()