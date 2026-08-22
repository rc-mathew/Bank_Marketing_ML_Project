"""CI-safe automated tests for the Bank Marketing ML pipeline."""

import numpy as np
import pandas as pd

from src.models.save_model import build_pipeline


def make_sample_data():
    """Create a small representative Bank Marketing dataset."""

    X = pd.DataFrame(
        {
            "age": [30, 45, 37, 52, 29, 61, 41, 34],
            "job": [
                "admin.",
                "technician",
                "services",
                "management",
                "student",
                "retired",
                "blue-collar",
                "admin.",
            ],
            "marital": [
                "single",
                "married",
                "single",
                "married",
                "single",
                "married",
                "married",
                "single",
            ],
            "education": [
                "university.degree",
                "professional.course",
                "high.school",
                "university.degree",
                "high.school",
                "basic.9y",
                "basic.6y",
                "university.degree",
            ],
            "campaign": [1, 2, 1, 3, 1, 2, 4, 1],
            "pdays": [5, 999, 999, 4, 999, 2, 999, 8],
            "previous": [1, 0, 0, 2, 0, 3, 0, 1],
            "euribor3m": [
                1.20,
                4.85,
                4.90,
                1.40,
                4.80,
                1.10,
                4.95,
                1.50,
            ],
        }
    )

    y = pd.Series(
        [1, 0, 0, 1, 0, 1, 0, 1],
        name="y",
    )

    return X, y


def test_pipeline_builds():
    """Pipeline should contain preprocessing and model stages."""

    X, _ = make_sample_data()

    pipeline = build_pipeline(X)

    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_pipeline_fits():
    """Pipeline should fit successfully on representative data."""

    X, y = make_sample_data()

    pipeline = build_pipeline(X)
    pipeline.fit(X, y)

    assert hasattr(
        pipeline.named_steps["model"],
        "classes_",
    )


def test_predictions_are_binary():
    """Predictions should contain only binary class labels."""

    X, y = make_sample_data()

    pipeline = build_pipeline(X)
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_probabilities_are_valid():
    """Positive-class probabilities must remain between 0 and 1."""

    X, y = make_sample_data()

    pipeline = build_pipeline(X)
    pipeline.fit(X, y)

    probabilities = pipeline.predict_proba(X)[:, 1]

    assert len(probabilities) == len(X)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_classifier_classes_are_binary():
    """Final classifier should represent a binary problem."""

    X, y = make_sample_data()

    pipeline = build_pipeline(X)
    pipeline.fit(X, y)

    classes = pipeline.named_steps["model"].classes_

    assert len(classes) == 2
    assert set(classes.tolist()) == {0, 1}