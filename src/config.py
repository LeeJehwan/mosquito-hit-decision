from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    source_dir: Path = Path("open")
    dataset_dir: Path = Path("dataset")
    train_dir: Path = Path("dataset/train")
    test_dir: Path = Path("dataset/test")
    dataset_metadata: Path = Path("dataset/metadata.json")
    model_output: Path = Path("artifacts/hit_lgbm.pkl")
    logistic_model_output: Path = Path("artifacts/hit_logistic.pkl")
    feature_output: Path = Path("artifacts/feature_columns.json")
    threshold_output: Path = Path("artifacts/decision_threshold.json")
    metrics_output: Path = Path("outputs/valid_metrics.json")
    test_metrics_output: Path = Path("outputs/test_metrics.json")
    prediction_output: Path = Path("outputs/hit_predictions.csv")


@dataclass(frozen=True)
class TrainingConfig:
    radius: float = 0.05
    valid_size: float = 0.2
    seed: int = 42
    threshold_min: float = 0.0
    threshold_max: float = 1.0
    threshold_step: float = 0.01


@dataclass(frozen=True)
class LightGBMConfig:
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 300
    max_depth: int = -1
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    n_jobs: int = -1


@dataclass(frozen=True)
class LogisticRegressionConfig:
    C: float = 1.0
    max_iter: int = 2000
    solver: str = "lbfgs"


MODEL_TYPES = ("lightgbm", "logistic")
DEFAULT_MODEL_TYPE = "lightgbm"
PATH_DEFAULTS = PathConfig()
TRAINING_DEFAULTS = TrainingConfig()
MODEL_DEFAULTS = LightGBMConfig()
LOGISTIC_MODEL_DEFAULTS = LogisticRegressionConfig()
