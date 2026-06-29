from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import LightGBMConfig, LogisticRegressionConfig
from src.model import (
    create_model,
    load_model,
    predict_hit_probabilities,
    save_model,
    train_model,
)


def make_binary_dataset() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "small_scale": [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
            "large_scale": [-2000, -1000, 0, 1000, 2000, 3000],
        }
    )
    labels = pd.Series([0, 0, 0, 1, 1, 1])
    return features, labels


def test_model_factory_creates_supported_model_types() -> None:
    lightgbm = create_model("lightgbm", LightGBMConfig(n_estimators=2), seed=42)
    logistic = create_model("logistic", LogisticRegressionConfig(max_iter=200), seed=42)

    assert isinstance(lightgbm, LGBMClassifier)
    assert isinstance(logistic, Pipeline)
    assert isinstance(logistic.named_steps["scaler"], StandardScaler)
    assert logistic.named_steps["classifier"].class_weight is None


@pytest.mark.parametrize(
    ("model_type", "config"),
    [
        ("lightgbm", LightGBMConfig(n_estimators=2)),
        ("logistic", LogisticRegressionConfig(max_iter=200)),
    ],
)
def test_models_predict_probabilities_and_survive_round_trip(
    tmp_path: Path,
    model_type: str,
    config: LightGBMConfig | LogisticRegressionConfig,
) -> None:
    features, labels = make_binary_dataset()
    model = train_model(create_model(model_type, config, seed=42), features, labels)
    expected = predict_hit_probabilities(model, features)
    model_path = tmp_path / f"{model_type}.pkl"

    save_model(model, model_path)
    actual = predict_hit_probabilities(load_model(model_path), features)

    assert expected.shape == (len(features),)
    assert np.all((expected >= 0.0) & (expected <= 1.0))
    np.testing.assert_allclose(actual, expected)


def test_model_factory_rejects_mismatched_configuration() -> None:
    with pytest.raises(TypeError, match="requires LogisticRegressionConfig"):
        create_model("logistic", LightGBMConfig(), seed=42)
