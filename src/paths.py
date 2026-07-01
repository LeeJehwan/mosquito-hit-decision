from dataclasses import dataclass
from pathlib import Path

from src.config import DEFAULT_MODEL_TYPE, MODEL_TYPES, PATH_DEFAULTS


@dataclass(frozen=True)
class TrainOutputPaths:
    model: Path
    features: Path
    threshold: Path
    metrics: Path


@dataclass(frozen=True)
class InferencePaths:
    model: Path
    features: Path
    threshold: Path
    predictions: Path
    metrics: Path


def _under_output(output: Path | None, default_path: Path) -> Path:
    return output / default_path.name if output is not None else default_path


def _model_output_for_type(model_type: str) -> Path:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")
    if model_type == "logistic":
        return PATH_DEFAULTS.logistic_model_output
    if model_type == "ensemble":
        return PATH_DEFAULTS.ensemble_model_output
    if model_type == "weighted_ensemble":
        return PATH_DEFAULTS.weighted_ensemble_model_output
    return PATH_DEFAULTS.model_output


def resolve_train_output_paths(
    output: Path | None,
    model: Path | None = None,
    features: Path | None = None,
    threshold: Path | None = None,
    metrics: Path | None = None,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> TrainOutputPaths:
    return TrainOutputPaths(
        model=model or _under_output(output, _model_output_for_type(model_type)),
        features=features or _under_output(output, PATH_DEFAULTS.feature_output),
        threshold=threshold or _under_output(output, PATH_DEFAULTS.threshold_output),
        metrics=metrics or _under_output(output, PATH_DEFAULTS.metrics_output),
    )


def resolve_inference_paths(
    output: Path | None,
    model: Path | None = None,
    features: Path | None = None,
    threshold: Path | None = None,
    predictions: Path | None = None,
    metrics: Path | None = None,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> InferencePaths:
    return InferencePaths(
        model=model or _under_output(output, _model_output_for_type(model_type)),
        features=features or _under_output(output, PATH_DEFAULTS.feature_output),
        threshold=threshold or _under_output(output, PATH_DEFAULTS.threshold_output),
        predictions=predictions or _under_output(output, PATH_DEFAULTS.prediction_output),
        metrics=metrics or _under_output(output, PATH_DEFAULTS.test_metrics_output),
    )
