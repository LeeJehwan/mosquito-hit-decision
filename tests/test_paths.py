from pathlib import Path

from src.paths import resolve_inference_paths, resolve_train_output_paths


def test_train_output_directory_contains_every_artifact() -> None:
    output = Path("runs/experiment_01")

    paths = resolve_train_output_paths(output=output)

    assert paths.model == output / "hit_lgbm.pkl"
    assert paths.features == output / "feature_columns.json"
    assert paths.threshold == output / "decision_threshold.json"
    assert paths.metrics == output / "valid_metrics.json"


def test_inference_reuses_artifacts_and_writes_predictions() -> None:
    output = Path("runs/experiment_01")

    paths = resolve_inference_paths(output=output)

    assert paths.model == output / "hit_lgbm.pkl"
    assert paths.features == output / "feature_columns.json"
    assert paths.threshold == output / "decision_threshold.json"
    assert paths.predictions == output / "hit_predictions.csv"
    assert paths.metrics == output / "test_metrics.json"


def test_logistic_model_uses_distinct_default_filename() -> None:
    output = Path("runs/logistic_01")

    train_paths = resolve_train_output_paths(output=output, model_type="logistic")
    inference_paths = resolve_inference_paths(output=output, model_type="logistic")

    assert train_paths.model == output / "hit_logistic.pkl"
    assert inference_paths.model == output / "hit_logistic.pkl"


def test_ensemble_model_uses_distinct_default_filename() -> None:
    output = Path("runs/ensemble_01")

    train_paths = resolve_train_output_paths(output=output, model_type="ensemble")
    inference_paths = resolve_inference_paths(output=output, model_type="ensemble")

    assert train_paths.model == output / "hit_ensemble.pkl"
    assert inference_paths.model == output / "hit_ensemble.pkl"


def test_weighted_ensemble_model_uses_distinct_default_filename() -> None:
    output = Path("runs/weighted_ensemble_01")

    train_paths = resolve_train_output_paths(output=output, model_type="weighted_ensemble")
    inference_paths = resolve_inference_paths(output=output, model_type="weighted_ensemble")

    assert train_paths.model == output / "hit_weighted_ensemble.pkl"
    assert inference_paths.model == output / "hit_weighted_ensemble.pkl"


def test_explicit_path_overrides_output_directory() -> None:
    explicit_model = Path("shared/model.pkl")

    paths = resolve_train_output_paths(
        output=Path("runs/experiment_01"),
        model=explicit_model,
    )

    assert paths.model == explicit_model
