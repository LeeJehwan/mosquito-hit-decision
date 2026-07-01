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
    ensemble_model_output: Path = Path("artifacts/hit_ensemble.pkl")
    weighted_ensemble_model_output: Path = Path("artifacts/hit_weighted_ensemble.pkl")
    feature_output: Path = Path("artifacts/feature_columns.json")
    threshold_output: Path = Path("artifacts/decision_threshold.json")
    metrics_output: Path = Path("outputs/valid_metrics.json")
    test_metrics_output: Path = Path("outputs/test_metrics.json")
    prediction_output: Path = Path("outputs/hit_predictions.csv")


@dataclass(frozen=True)
class TrainingConfig:
    radius: float = 0.05
    valid_size: float = 0.2
    cv_folds: int = 5
    seed: int = 42
    threshold_min: float = 0.0
    threshold_max: float = 1.0
    threshold_step: float = 0.005


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


@dataclass(frozen=True)
class EnsembleConfig:
    """1위 방법론: LightGBM + HistGradientBoosting + MLP 소프트보팅(확률 평균).

    각 멤버는 서로 다른 inductive bias(leaf-wise 트리 / level-wise 트리 / 신경망)를
    가져 오류가 비상관이므로, 확률을 평균하면 랭킹(AUROC)이 안정적으로 향상된다.
    """

    # LightGBM member
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 300
    # HistGradientBoosting member
    hist_max_iter: int = 400
    hist_max_leaf_nodes: int = 31
    hist_l2: float = 1.0
    # MLP member (StandardScaler 포함 pipeline)
    mlp_hidden: tuple[int, ...] = (128, 64)
    mlp_alpha: float = 1e-3
    mlp_batch_size: int = 256
    mlp_learning_rate_init: float = 1e-3
    mlp_max_iter: int = 300
    mlp_early_stopping: bool = True
    n_jobs: int = -1


@dataclass(frozen=True)
class WeightedEnsembleConfig(EnsembleConfig):
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)


WEIGHTED_ENSEMBLE_WEIGHTS_BY_RADIUS = {
    0.01: (0.50, 0.10, 0.40),
    0.02: (0.50, 0.20, 0.30),
    0.03: (0.25, 0.30, 0.45),
    0.04: (0.20, 0.45, 0.35),
    0.05: (0.35, 0.55, 0.10),
}


MODEL_TYPES = ("lightgbm", "logistic", "ensemble", "weighted_ensemble")
DEFAULT_MODEL_TYPE = "lightgbm"
PATH_DEFAULTS = PathConfig()
TRAINING_DEFAULTS = TrainingConfig()
MODEL_DEFAULTS = LightGBMConfig()
LOGISTIC_MODEL_DEFAULTS = LogisticRegressionConfig()
ENSEMBLE_MODEL_DEFAULTS = EnsembleConfig()
WEIGHTED_ENSEMBLE_MODEL_DEFAULTS = WeightedEnsembleConfig()
