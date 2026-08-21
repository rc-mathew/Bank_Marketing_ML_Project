"""
Model explainability analysis for the Bank Marketing project.

Trains the tuned Random Forest model and extracts feature
importances after preprocessing.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import clean_bank_data, split_features_target


DATA_PATH = "data/raw/bank-additional-full.csv"
RANDOM_STATE = 42


def build_model(X):
    """Build tuned Random Forest pipeline."""

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
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
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
        n_jobs=2,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


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

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples    : {len(X_test)}")

    model = build_model(X_train)

    print("\nTraining tuned Random Forest...")
    model.fit(X_train, y_train)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    importances = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("=" * 70)

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE SUMMARY")
    print("=" * 70)

    print(
        f"Total transformed features: "
        f"{len(importance_df)}"
    )

    print(
        f"Top 10 cumulative importance: "
        f"{importance_df.head(10)['importance'].sum():.4f}"
    )

    print(
        f"Top 20 cumulative importance: "
        f"{importance_df.head(20)['importance'].sum():.4f}"
    )


if __name__ == "__main__":
    main()