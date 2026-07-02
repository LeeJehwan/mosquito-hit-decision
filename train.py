import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from src.args import parse_train_args
from src.config import (
    ENSEMBLE_MODEL_DEFAULTS,
    WEIGHTED_ENSEMBLE_WEIGHTS_BY_RADIUS,
    EnsembleConfig,
    LightGBMConfig,
    LogisticRegressionConfig,
    WeightedEnsembleConfig,
)
from src.data_io import save_json
from src.dataset import load_prepared_trajectories, radius_to_label
from src.features import build_feature_frame
from src.features_advanced import build_feature_frame_advanced
from src.metrics import evaluate_predictions, find_best_threshold, make_threshold_candidates
from src.model import create_model, predict_hit_probabilities, save_model, train_model
from src.paths import resolve_train_output_paths


def resolve_train_input_paths(args):
    train_dir = args.train_dir or args.dataset_dir / "train"
    metadata_path = args.metadata_path or args.dataset_dir / "metadata.json"
    return train_dir, metadata_path


def validate_train_args(args) -> None:
    if not 0.0 < args.valid_size < 1.0:
        raise ValueError("valid-size must be strictly between 0 and 1")
    if args.cv_folds < 2:
        raise ValueError("cv-folds must be at least 2")
    if args.threshold_min > args.threshold_max:
        raise ValueError("threshold-min must not exceed threshold-max")
    if args.model_type == "lightgbm" and (args.num_leaves <= 1 or args.n_estimators <= 0):
        raise ValueError("num-leaves must exceed 1 and n-estimators must be positive")


def build_features_for(model_type, trajectories, show_progress):
    # 앙상블(1위 방법론)은 외삽-백테스트 등 물리 기반 특징까지 사용한다.
    if model_type in {"ensemble", "weighted_ensemble"}:
        return build_feature_frame_advanced(trajectories, show_progress=show_progress)
    return build_feature_frame(trajectories, show_progress=show_progress)


def weights_for_radius(radius: float) -> tuple[float, float, float]:
    for supported_radius, weights in WEIGHTED_ENSEMBLE_WEIGHTS_BY_RADIUS.items():
        if np.isclose(radius, supported_radius, rtol=0.0, atol=1e-9):
            return weights
    supported = ", ".join(f"{radius:.2f}" for radius in WEIGHTED_ENSEMBLE_WEIGHTS_BY_RADIUS)
    raise ValueError(f"weighted_ensemble supports only radii: {supported}")


def make_model_config(args):
    if args.model_type == "weighted_ensemble":
        return WeightedEnsembleConfig(weights=weights_for_radius(args.radius))
    if args.model_type == "ensemble":
        return ENSEMBLE_MODEL_DEFAULTS
    if args.model_type == "logistic":
        return LogisticRegressionConfig(
            C=args.logistic_c,
            max_iter=args.logistic_max_iter,
        )
    return LightGBMConfig(
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        n_jobs=args.n_jobs,
    )


def select_threshold(args, labels, probabilities):
    if args.threshold is not None:
        return args.threshold, None
    candidates = make_threshold_candidates(
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )
    result = find_best_threshold(labels.to_numpy(), probabilities, candidates)
    return result.threshold, result


def validate_oof_labels(labels: pd.Series, cv_folds: int) -> None:
    counts = labels.value_counts()
    if len(counts) != 2:
        raise ValueError("OOF threshold selection requires both hit and miss labels")
    smallest_class = int(counts.min())
    if cv_folds > smallest_class:
        raise ValueError(
            f"cv-folds={cv_folds} exceeds the smallest class count ({smallest_class})"
        )


def make_oof_probabilities(
    model_type: str,
    model_config: (
        LightGBMConfig
        | LogisticRegressionConfig
        | EnsembleConfig
        | WeightedEnsembleConfig
    ),
    features: pd.DataFrame,
    labels: pd.Series,
    seed: int,
    cv_folds: int,
    show_progress: bool,
) -> np.ndarray:
    validate_oof_labels(labels, cv_folds)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    probabilities = np.empty(len(labels), dtype=float)
    fold_iterator = tqdm(
        splitter.split(features, labels),
        total=cv_folds,
        desc="OOF threshold folds",
        unit="fold",
        disable=not show_progress,
    )
    for fold_index, (fit_indices, valid_indices) in enumerate(fold_iterator, start=1):
        model = train_model(
            create_model(model_type, model_config, seed),
            features.iloc[fit_indices],
            labels.iloc[fit_indices],
            show_progress=show_progress,
            description=f"Training OOF fold {fold_index}/{cv_folds}",
        )
        probabilities[valid_indices] = predict_hit_probabilities(
            model,
            features.iloc[valid_indices],
        )
    return probabilities


def main() -> None:
    args = parse_train_args()
    validate_train_args(args)
    train_dir, metadata_path = resolve_train_input_paths(args)
    output_paths = resolve_train_output_paths(
        output=args.output,
        model_type=args.model_type,
        model=args.model_output,
        features=args.feature_output,
        threshold=args.threshold_output,
        metrics=args.metrics_output,
    )

    trajectories, labels, metadata = load_prepared_trajectories(
        train_dir,
        metadata_path,
        split="train",
        show_progress=args.progress,
    )
    feature_frame = build_features_for(args.model_type, trajectories, args.progress)
    dataset = feature_frame.merge(labels, on="id", validate="one_to_one")
    feature_columns = [column for column in feature_frame.columns if column != "id"]
    label_column = radius_to_label(args.radius)
    x_all = dataset.loc[:, feature_columns]
    y_all = dataset[label_column]

    model_config = make_model_config(args)
    oof_probabilities = make_oof_probabilities(
        args.model_type,
        model_config,
        x_all,
        y_all,
        seed=args.seed,
        cv_folds=args.cv_folds,
        show_progress=args.progress,
    )
    threshold, threshold_result = select_threshold(args, y_all, oof_probabilities)
    metrics = evaluate_predictions(y_all.to_numpy(), oof_probabilities, threshold)
    metrics.update(
        {
            "radius": args.radius,
            "seed": args.seed,
            "validation_method": "oof",
            "cv_folds": args.cv_folds,
            "oof_samples": len(dataset),
            "dataset_samples": len(dataset),
            "dataset_id_sha256": metadata["split_id_sha256"]["train"],
            "threshold_source": "argument" if args.threshold is not None else "oof_search",
            "model_type": args.model_type,
        }
    )

    final_model = train_model(
        create_model(args.model_type, model_config, args.seed),
        dataset.loc[:, feature_columns],
        dataset[label_column],
        show_progress=args.progress,
        description="Training final model",
    )
    threshold_artifact = {
        "threshold": threshold,
        "radius": args.radius,
        "validation_hit_score": metrics["hit_score"],
        "validation_mean_hit_score": metrics["mean_hit_score"],
        "dataset_source_id_sha256": metadata["source_id_sha256"],
        "source": metrics["threshold_source"],
        "model_type": args.model_type,
        "validation_method": metrics["validation_method"],
        "cv_folds": args.cv_folds,
    }
    if threshold_result is not None:
        threshold_artifact["candidate_hit_score"] = threshold_result.hit_score

    save_model(final_model, output_paths.model)
    save_json(feature_columns, output_paths.features)
    save_json(threshold_artifact, output_paths.threshold)
    save_json(metrics, output_paths.metrics)

    print(f"OOF validation metrics: {metrics}")
    print(f"Saved training artifacts to {output_paths.model.parent}")


if __name__ == "__main__":
    main()
