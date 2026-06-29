from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_io import save_dataframe, save_json
from src.dataset import (
    DATASET_COLUMNS,
    LABEL_COLUMNS,
    SUPPORTED_RADII,
    build_metadata,
    hash_ids,
    load_prepared_trajectories,
    radius_to_label,
    split_by_error,
    split_train_validation,
)


def make_split_frame(samples_per_band: int = 10) -> pd.DataFrame:
    errors = np.repeat([0.005, 0.015, 0.025, 0.035, 0.045, 0.055], samples_per_band)
    frame = pd.DataFrame(
        {
            "id": [f"TRAIN_{index:05d}" for index in range(len(errors))],
            "error": errors,
        }
    )
    for radius, label in zip(SUPPORTED_RADII, LABEL_COLUMNS, strict=True):
        frame[label] = frame["error"].le(radius).astype(int)
    return frame


def make_sample(labels: dict[str, int]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestep_ms": np.arange(-400, 40, 40),
            "x": np.linspace(0.0, 1.0, 11),
            "y": np.linspace(1.0, 2.0, 11),
            "z": np.linspace(2.0, 3.0, 11),
        }
    )
    for label in LABEL_COLUMNS:
        frame[label] = labels[label]
    return frame


def test_fixed_split_is_reproducible_disjoint_and_stratified() -> None:
    samples = make_split_frame()
    train_a, test_a = split_by_error(samples, test_size=0.2, seed=42)
    train_b, test_b = split_by_error(samples, test_size=0.2, seed=42)
    _, test_c = split_by_error(samples, test_size=0.2, seed=7)

    assert len(train_a) == 48
    assert len(test_a) == 12
    assert train_a["id"].tolist() == train_b["id"].tolist()
    assert test_a["id"].tolist() == test_b["id"].tolist()
    assert test_a["id"].tolist() != test_c["id"].tolist()
    assert set(train_a["id"]).isdisjoint(test_a["id"])
    assert set(train_a["id"]) | set(test_a["id"]) == set(samples["id"])
    assert test_a["error"].value_counts().eq(2).all()


def test_internal_split_preserves_all_radius_label_bands() -> None:
    labels = make_split_frame().drop(columns="error")
    fit_indices, valid_indices = split_train_validation(labels, valid_size=0.2, seed=42)

    assert len(fit_indices) == 48
    assert len(valid_indices) == 12
    assert set(fit_indices).isdisjoint(valid_indices)


def test_prepared_loader_reads_trajectory_and_constant_labels(tmp_path: Path) -> None:
    samples = make_split_frame(samples_per_band=1)
    train_ids = samples["id"].iloc[:3].tolist()
    test_ids = samples["id"].iloc[3:].tolist()
    metadata = build_metadata(samples["id"].tolist(), train_ids, test_ids, seed=42, test_size=0.5)
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    for sample_id in train_ids:
        row = samples.set_index("id").loc[sample_id]
        save_dataframe(make_sample(row.loc[list(LABEL_COLUMNS)].astype(int).to_dict()), train_dir / f"{sample_id}.csv")
    metadata_path = tmp_path / "metadata.json"
    save_json(metadata, metadata_path)

    trajectories, labels, loaded_metadata = load_prepared_trajectories(
        train_dir, metadata_path, split="train"
    )

    assert len(trajectories) == 3
    assert trajectories[train_ids[0]].columns.tolist() == ["timestep_ms", "x", "y", "z"]
    assert labels.columns.tolist() == ["id", *LABEL_COLUMNS]
    assert loaded_metadata["split_id_sha256"]["train"] == hash_ids(train_ids)


def test_prepared_loader_rejects_nonconstant_label(tmp_path: Path) -> None:
    samples = make_split_frame(samples_per_band=1)
    sample_id = samples["id"].iloc[0]
    row = samples.set_index("id").loc[sample_id]
    frame = make_sample(row.loc[list(LABEL_COLUMNS)].astype(int).to_dict())
    frame.loc[1, "hit_r001"] = 1 - frame.loc[0, "hit_r001"]
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    save_dataframe(frame, train_dir / f"{sample_id}.csv")
    metadata = build_metadata([sample_id], [sample_id], [], seed=42, test_size=0.0)
    metadata_path = tmp_path / "metadata.json"
    save_json(metadata, metadata_path)

    with pytest.raises(ValueError, match="constant within a sample"):
        load_prepared_trajectories(train_dir, metadata_path, split="train")


def test_sample_schema_contains_only_trajectory_and_hit_columns() -> None:
    assert DATASET_COLUMNS == ("timestep_ms", "x", "y", "z", *LABEL_COLUMNS)
    assert radius_to_label(0.03) == "hit_r003"
    with pytest.raises(ValueError, match="radius must be one of"):
        radius_to_label(0.06)
