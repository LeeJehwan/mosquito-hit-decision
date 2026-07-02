import argparse
from pathlib import Path

from src.config import (
    DEFAULT_MODEL_TYPE,
    LOGISTIC_MODEL_DEFAULTS,
    MODEL_DEFAULTS,
    MODEL_TYPES,
    PATH_DEFAULTS,
    TRAINING_DEFAULTS,
)
from src.dataset import SUPPORTED_RADII


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _supported_radius(value: str) -> float:
    parsed = float(value)
    if not any(abs(parsed - radius) <= 1e-9 for radius in SUPPORTED_RADII):
        supported = ", ".join(f"{radius:.2f}" for radius in SUPPORTED_RADII)
        raise argparse.ArgumentTypeError(f"must be one of: {supported}")
    return parsed


def build_prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a fixed mosquito train/test dataset.")
    parser.add_argument("--source-dir", type=Path, default=PATH_DEFAULTS.source_dir)
    parser.add_argument("--dataset-dir", type=Path, default=PATH_DEFAULTS.dataset_dir)
    parser.add_argument("--seed", type=int, default=TRAINING_DEFAULTS.seed)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars (use --no-progress to disable).",
    )
    return parser


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a mosquito hit-decision model.")
    parser.add_argument("--dataset-dir", type=Path, default=PATH_DEFAULTS.dataset_dir)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--model-type", choices=MODEL_TYPES, default=DEFAULT_MODEL_TYPE)
    parser.add_argument(
        "--output",
        type=Path,
        help="Directory for all model artifacts and training results.",
    )
    parser.add_argument("--radius", type=_supported_radius, default=TRAINING_DEFAULTS.radius)
    parser.add_argument(
        "--valid-size",
        type=_probability,
        default=TRAINING_DEFAULTS.valid_size,
        help="Deprecated; threshold selection now uses OOF folds.",
    )
    parser.add_argument("--cv-folds", type=_positive_int, default=TRAINING_DEFAULTS.cv_folds)
    parser.add_argument("--seed", type=int, default=TRAINING_DEFAULTS.seed)
    parser.add_argument("--threshold", type=_probability)
    parser.add_argument("--threshold-min", type=_probability, default=TRAINING_DEFAULTS.threshold_min)
    parser.add_argument("--threshold-max", type=_probability, default=TRAINING_DEFAULTS.threshold_max)
    parser.add_argument("--threshold-step", type=_positive_float, default=TRAINING_DEFAULTS.threshold_step)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--feature-output", type=Path)
    parser.add_argument("--threshold-output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--num-leaves", type=int, default=MODEL_DEFAULTS.num_leaves)
    parser.add_argument("--learning-rate", type=_positive_float, default=MODEL_DEFAULTS.learning_rate)
    parser.add_argument("--n-estimators", type=int, default=MODEL_DEFAULTS.n_estimators)
    parser.add_argument("--max-depth", type=int, default=MODEL_DEFAULTS.max_depth)
    parser.add_argument("--subsample", type=_probability, default=MODEL_DEFAULTS.subsample)
    parser.add_argument("--colsample-bytree", type=_probability, default=MODEL_DEFAULTS.colsample_bytree)
    parser.add_argument("--n-jobs", type=int, default=MODEL_DEFAULTS.n_jobs)
    parser.add_argument("--logistic-c", type=_positive_float, default=LOGISTIC_MODEL_DEFAULTS.C)
    parser.add_argument(
        "--logistic-max-iter",
        type=_positive_int,
        default=LOGISTIC_MODEL_DEFAULTS.max_iter,
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars (use --no-progress to disable).",
    )
    return parser


def build_infer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer mosquito hit probabilities and fire decisions.")
    parser.add_argument("--dataset-dir", type=Path, default=PATH_DEFAULTS.dataset_dir)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--model-type", choices=MODEL_TYPES, default=DEFAULT_MODEL_TYPE)
    parser.add_argument(
        "--output",
        type=Path,
        help="Directory containing training artifacts and receiving inference results.",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--feature-path", type=Path)
    parser.add_argument("--threshold-path", type=Path)
    parser.add_argument("--threshold", type=_probability)
    parser.add_argument("--radius", type=_supported_radius, default=TRAINING_DEFAULTS.radius)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars (use --no-progress to disable).",
    )
    return parser


def parse_train_args() -> argparse.Namespace:
    return build_train_parser().parse_args()


def parse_infer_args() -> argparse.Namespace:
    return build_infer_parser().parse_args()


def parse_prepare_args() -> argparse.Namespace:
    return build_prepare_parser().parse_args()
