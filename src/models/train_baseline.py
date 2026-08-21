import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

from src.features.build_features import (
    clean_bank_data,
    split_features_target,
    get_feature_groups,
)


DATA_PATH = "data/raw/bank-additional-full.csv"
RANDOM_STATE = 42


def build_preprocessor(X):
    categorical_features, numerical_features = get_feature_groups(X)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ])


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print("=" * 60)
    print(classification_report(y_test, pred, zero_division=0))
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
    print("PR-AUC :", round(average_precision_score(y_test, proba), 4))


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, sep=";")
    df = clean_bank_data(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X_train)

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE
            ))
        ]),

        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ])
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        evaluate(name, model, X_test, y_test)