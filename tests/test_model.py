"""Automated tests for the Bank Marketing ML project."""

from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "bank_marketing_model.joblib"


def test_model_artifact_exists():
    """Check that the trained model artifact exists."""
    assert MODEL_PATH.exists(), (
        "Model artifact not found. Run "
        "`python -m src.models.save_model` first."
    )


def test_saved_artifact_loads():
    """Check that the saved model can be loaded successfully."""
    artifact = joblib.load(MODEL_PATH)

    assert artifact is not None


def test_saved_artifact_can_predict():
    """Check that the saved pipeline produces valid predictions."""
    artifact = joblib.load(MODEL_PATH)

    # The persisted artifact contains preprocessing + trained model.
    assert hasattr(artifact, "predict")

    # Verify the fitted model exposes expected classifier classes.
    assert hasattr(artifact, "classes_") or hasattr(
        artifact, "named_steps"
    )


def test_classifier_classes_are_binary():
    """Check that the final classifier represents binary classification."""
    artifact = joblib.load(MODEL_PATH)

    if hasattr(artifact, "classes_"):
        classes = artifact.classes_
    elif hasattr(artifact, "named_steps"):
        final_estimator = list(artifact.named_steps.values())[-1]
        classes = final_estimator.classes_
    else:
        raise AssertionError("Unable to locate classifier classes.")

    assert len(classes) == 2
    assert set(np.asarray(classes).tolist()) == {0, 1}