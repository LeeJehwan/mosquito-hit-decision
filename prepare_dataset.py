from src.args import parse_prepare_args
from src.data_io import load_future_labels, load_trajectories, save_dataframe, save_json
from src.dataset import (
    LABEL_COLUMNS,
    SUPPORTED_RADII,
    build_metadata,
    split_by_error,
)
from src.labels import build_hit_labels
from tqdm.auto import tqdm


EXPECTED_SOURCE_SAMPLES = 10_000
TEST_SIZE = 0.2


def write_split(trajectories, labels, output_dir, show_progress: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_ids = set(labels.index)
    for stale_path in output_dir.glob("*.csv"):
        if stale_path.stem not in expected_ids:
            stale_path.unlink()
    iterator = tqdm(
        labels.iterrows(),
        total=len(labels),
        desc=f"Writing {output_dir.name} dataset",
        unit="file",
        disable=not show_progress,
    )
    for sample_id, label_row in iterator:
        frame = trajectories[sample_id].copy()
        for label in LABEL_COLUMNS:
            frame[label] = int(label_row[label])
        save_dataframe(frame, output_dir / f"{sample_id}.csv")


def main() -> None:
    args = parse_prepare_args()
    train_dir = args.source_dir / "train"
    label_path = args.source_dir / "train_labels.csv"

    trajectories = load_trajectories(
        train_dir,
        show_progress=args.progress,
        description="Loading source trajectories",
    )
    future_labels = load_future_labels(label_path)
    if len(trajectories) != EXPECTED_SOURCE_SAMPLES or len(future_labels) != EXPECTED_SOURCE_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_SAMPLES} source samples, got "
            f"{len(trajectories)} trajectories and {len(future_labels)} labels"
        )

    hit_frame = build_hit_labels(
        trajectories,
        future_labels,
        max(SUPPORTED_RADII),
        show_progress=args.progress,
    )
    derived = hit_frame.loc[:, ["id", "error"]].copy()
    for radius, label in zip(SUPPORTED_RADII, LABEL_COLUMNS, strict=True):
        derived[label] = (derived["error"] <= radius).astype(int)

    train, test = split_by_error(derived, test_size=TEST_SIZE, seed=args.seed)
    metadata = build_metadata(
        source_ids=derived["id"].tolist(),
        train_ids=train["id"].tolist(),
        test_ids=test["id"].tolist(),
        seed=args.seed,
        test_size=TEST_SIZE,
    )

    write_split(
        trajectories,
        train.set_index("id").loc[:, LABEL_COLUMNS],
        args.dataset_dir / "train",
        args.progress,
    )
    write_split(
        trajectories,
        test.set_index("id").loc[:, LABEL_COLUMNS],
        args.dataset_dir / "test",
        args.progress,
    )
    save_json(metadata, args.dataset_dir / "metadata.json")
    print(f"Saved {len(train)} train and {len(test)} test samples to {args.dataset_dir}")


if __name__ == "__main__":
    main()
