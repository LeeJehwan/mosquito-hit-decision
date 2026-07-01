import numpy as np
import pandas as pd
import pytest

from src.config import LogisticRegressionConfig
from train import make_oof_probabilities, validate_oof_labels, weights_for_radius


def make_binary_frame() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "x": [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "x2": [0.36, 0.25, 0.16, 0.09, 0.04, 0.01, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36],
        }
    )
    labels = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    return features, labels


def test_make_oof_probabilities_returns_one_prediction_per_sample() -> None:
    features, labels = make_binary_frame()

    probabilities = make_oof_probabilities(
        "logistic",
        LogisticRegressionConfig(max_iter=200),
        features,
        labels,
        seed=42,
        cv_folds=3,
        show_progress=False,
    )

    assert probabilities.shape == (len(labels),)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_validate_oof_labels_rejects_too_many_folds() -> None:
    _, labels = make_binary_frame()

    with pytest.raises(ValueError, match="smallest class count"):
        validate_oof_labels(labels, cv_folds=7)


def test_validate_oof_labels_requires_both_classes() -> None:
    labels = pd.Series([1, 1, 1, 1])

    with pytest.raises(ValueError, match="both hit and miss"):
        validate_oof_labels(labels, cv_folds=2)


def test_weights_for_radius_allows_supported_float_values() -> None:
    assert weights_for_radius(0.0500000001) == (0.35, 0.55, 0.10)


def test_weights_for_radius_rejects_unsupported_radius() -> None:
    with pytest.raises(ValueError, match="supports only radii"):
        weights_for_radius(0.06)
