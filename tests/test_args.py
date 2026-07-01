import pytest

from src.args import build_infer_parser, build_prepare_parser, build_train_parser


def test_train_arguments_override_major_configuration() -> None:
    args = build_train_parser().parse_args(
        [
            "--radius",
            "0.03",
            "--threshold",
            "0.8",
            "--learning-rate",
            "0.1",
            "--n-estimators",
            "100",
            "--cv-folds",
            "4",
            "--model-type",
            "logistic",
            "--logistic-c",
            "0.5",
            "--logistic-max-iter",
            "500",
        ]
    )

    assert args.radius == 0.03
    assert args.threshold == 0.8
    assert args.learning_rate == 0.1
    assert args.n_estimators == 100
    assert args.cv_folds == 4
    assert args.model_type == "logistic"
    assert args.logistic_c == 0.5
    assert args.logistic_max_iter == 500


def test_lightgbm_is_the_default_model_type() -> None:
    assert build_train_parser().parse_args([]).model_type == "lightgbm"
    assert build_infer_parser().parse_args([]).model_type == "lightgbm"


def test_weighted_ensemble_is_available_in_train_and_infer() -> None:
    assert (
        build_train_parser().parse_args(["--model-type", "weighted_ensemble"]).model_type
        == "weighted_ensemble"
    )
    assert (
        build_infer_parser().parse_args(["--model-type", "weighted_ensemble"]).model_type
        == "weighted_ensemble"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--model-type", "unsupported"],
        ["--logistic-c", "0"],
        ["--logistic-max-iter", "0"],
    ],
)
def test_train_rejects_invalid_model_arguments(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_train_parser().parse_args(arguments)


def test_infer_threshold_override_is_optional() -> None:
    parser = build_infer_parser()

    assert parser.parse_args([]).threshold is None
    assert parser.parse_args(["--threshold", "0.7"]).threshold == 0.7


def test_output_directory_is_available_for_train_and_infer() -> None:
    train_args = build_train_parser().parse_args(["--output", "runs/experiment_01"])
    infer_args = build_infer_parser().parse_args(["--output", "runs/experiment_01"])

    assert train_args.output.parts == ("runs", "experiment_01")
    assert infer_args.output.parts == ("runs", "experiment_01")


def test_progress_is_enabled_by_default_and_can_be_disabled() -> None:
    assert build_train_parser().parse_args([]).progress is True
    assert build_train_parser().parse_args(["--no-progress"]).progress is False
    assert build_infer_parser().parse_args([]).progress is True
    assert build_infer_parser().parse_args(["--no-progress"]).progress is False
    assert build_prepare_parser().parse_args(["--no-progress"]).progress is False


def test_dataset_paths_are_the_defaults() -> None:
    train_args = build_train_parser().parse_args([])
    infer_args = build_infer_parser().parse_args([])

    assert str(train_args.dataset_dir) == "dataset"
    assert str(infer_args.dataset_dir) == "dataset"


def test_radius_must_have_a_precomputed_label() -> None:
    with pytest.raises(SystemExit):
        build_train_parser().parse_args(["--radius", "0.06"])
