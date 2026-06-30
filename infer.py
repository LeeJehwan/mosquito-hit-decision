import numpy as np

from src.args import parse_infer_args
from src.data_io import load_json, save_dataframe, save_json
from src.dataset import load_prepared_trajectories, radius_to_label
from src.features import align_feature_columns, build_feature_frame
from src.features_advanced import build_feature_frame_advanced
from src.labels import build_aim_frame
from src.metrics import evaluate_predictions, probabilities_to_decisions
from src.model import load_model, predict_hit_probabilities
from src.paths import resolve_inference_paths


def resolve_threshold(cli_threshold: float | None, artifact: dict) -> float:
    if cli_threshold is not None:
        return cli_threshold
    threshold = float(artifact["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Stored threshold must be between 0 and 1, got {threshold}")
    return threshold


def validate_artifact_model_type(artifact: dict, selected_model_type: str) -> None:
    artifact_model_type = artifact.get("model_type", "lightgbm")
    if artifact_model_type != selected_model_type:
        raise ValueError(
            f"Artifact model type {artifact_model_type} does not match "
            f"selected model type {selected_model_type}"
        )


def validate_probabilities(probabilities: np.ndarray) -> None:
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities contain non-finite values")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Predicted probabilities must be between 0 and 1")


def main() -> None:
    args = parse_infer_args()
    test_dir = args.test_dir or args.dataset_dir / "test"
    metadata_path = args.metadata_path or args.dataset_dir / "metadata.json"
    paths = resolve_inference_paths(
        output=args.output,
        model_type=args.model_type,
        model=args.model_path,
        features=args.feature_path,
        threshold=args.threshold_path,
        predictions=args.output_path,
        metrics=args.metrics_output,
    )
    trajectories, labels, metadata = load_prepared_trajectories(
        test_dir,
        metadata_path,
        split="test",
        show_progress=args.progress,
    )
    if args.model_type == "ensemble":
        feature_frame = build_feature_frame_advanced(trajectories, show_progress=args.progress)
    else:
        feature_frame = build_feature_frame(trajectories, show_progress=args.progress)
    feature_columns = load_json(paths.features)
    features = align_feature_columns(feature_frame, feature_columns)
    threshold_artifact = load_json(paths.threshold)
    validate_artifact_model_type(threshold_artifact, args.model_type)
    model = load_model(paths.model)
    probabilities = predict_hit_probabilities(model, features)
    validate_probabilities(probabilities)

    threshold = resolve_threshold(args.threshold, threshold_artifact)
    trained_radius = float(threshold_artifact["radius"])
    if not np.isclose(trained_radius, args.radius, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Model radius {trained_radius:.2f} does not match inference radius {args.radius:.2f}"
        )
    trained_source_hash = threshold_artifact.get("dataset_source_id_sha256")
    if trained_source_hash != metadata["source_id_sha256"]:
        raise ValueError("Model and test dataset source hashes do not match")
    label_column = radius_to_label(args.radius)
    decisions = probabilities_to_decisions(probabilities, threshold)
    result = build_aim_frame(trajectories, show_progress=args.progress)
    result["hit_probability"] = probabilities
    result["fire_decision"] = decisions
    result = result.merge(labels.loc[:, ["id", label_column]], on="id", validate="one_to_one")
    metrics = evaluate_predictions(labels[label_column].to_numpy(), probabilities, threshold)
    metrics.update(
        {
            "radius": args.radius,
            "dataset_id_sha256": metadata["split_id_sha256"]["test"],
            "model_type": args.model_type,
        }
    )
    save_dataframe(result, paths.predictions)
    save_json(metrics, paths.metrics)
    print(f"Saved {len(result)} predictions to {paths.predictions} with threshold={threshold:.4f}")
    print(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()
