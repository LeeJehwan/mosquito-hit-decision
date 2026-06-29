import pytest

from infer import resolve_threshold, validate_artifact_model_type


def test_threshold_resolution_accepts_cli_override() -> None:
    assert resolve_threshold(0.7, {"threshold": 0.2}) == 0.7


def test_artifact_model_type_must_match_selection() -> None:
    validate_artifact_model_type({"model_type": "logistic"}, "logistic")

    with pytest.raises(ValueError, match="does not match"):
        validate_artifact_model_type({"model_type": "logistic"}, "lightgbm")


def test_legacy_artifact_defaults_to_lightgbm() -> None:
    validate_artifact_model_type({"threshold": 0.8}, "lightgbm")

    with pytest.raises(ValueError, match="Artifact model type lightgbm"):
        validate_artifact_model_type({"threshold": 0.8}, "logistic")
