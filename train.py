from src.args import parse_train_args
from src.config import ENSEMBLE_MODEL_DEFAULTS, LightGBMConfig, LogisticRegressionConfig
from src.data_io import save_json
from src.dataset import load_prepared_trajectories, radius_to_label, split_train_validation
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
    if args.threshold_min > args.threshold_max:
        raise ValueError("threshold-min must not exceed threshold-max")
    if args.model_type == "lightgbm" and (args.num_leaves <= 1 or args.n_estimators <= 0):
        raise ValueError("num-leaves must exceed 1 and n-estimators must be positive")


def build_features_for(model_type, trajectories, show_progress):
    # 앙상블(1위 방법론)은 외삽-백테스트 등 물리 기반 특징까지 사용한다.
    if model_type == "ensemble":
        return build_feature_frame_advanced(trajectories, show_progress=show_progress)
    return build_feature_frame(trajectories, show_progress=show_progress)


def make_model_config(args):
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


def select_threshold(args, y_valid, probabilities):
    if args.threshold is not None:
        return args.threshold, None
    candidates = make_threshold_candidates(
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )
    result = find_best_threshold(y_valid.to_numpy(), probabilities, candidates)
    return result.threshold, result


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

    fit_indices, valid_indices = split_train_validation(
        dataset,
        valid_size=args.valid_size,
        seed=args.seed,
    )
    fit_dataset = dataset.iloc[fit_indices]
    valid_dataset = dataset.iloc[valid_indices]
    x_train = fit_dataset.loc[:, feature_columns]
    y_train = fit_dataset[label_column]
    x_valid = valid_dataset.loc[:, feature_columns]
    y_valid = valid_dataset[label_column]

    model_config = make_model_config(args)
    validation_model = train_model(
        create_model(args.model_type, model_config, args.seed),
        x_train,
        y_train,
        show_progress=args.progress,
        description="Training validation model",
    )
    valid_probabilities = predict_hit_probabilities(validation_model, x_valid)
    threshold, threshold_result = select_threshold(args, y_valid, valid_probabilities)
    metrics = evaluate_predictions(y_valid.to_numpy(), valid_probabilities, threshold)
    metrics.update(
        {
            "radius": args.radius,
            "seed": args.seed,
            "valid_size": args.valid_size,
            "train_samples": len(fit_dataset),
            "validation_samples": len(valid_dataset),
            "dataset_samples": len(dataset),
            "dataset_id_sha256": metadata["split_id_sha256"]["train"],
            "threshold_source": "argument" if args.threshold is not None else "validation_search",
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
    }
    if threshold_result is not None:
        threshold_artifact["candidate_hit_score"] = threshold_result.hit_score

    save_model(final_model, output_paths.model)
    save_json(feature_columns, output_paths.features)
    save_json(threshold_artifact, output_paths.threshold)
    save_json(metrics, output_paths.metrics)

    print(f"Validation metrics: {metrics}")
    print(f"Saved training artifacts to {output_paths.model.parent}")


if __name__ == "__main__":
    main()
