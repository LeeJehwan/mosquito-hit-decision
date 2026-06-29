import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_io import (
    REQUIRED_TRAJECTORY_COLUMNS,
    list_csv_files,
    load_json,
    validate_trajectory,
)


SCHEMA_VERSION = 2
SUPPORTED_RADII = (0.01, 0.02, 0.03, 0.04, 0.05)
LABEL_COLUMNS = tuple(f"hit_r{int(round(radius * 100)):03d}" for radius in SUPPORTED_RADII)
DATASET_COLUMNS = (*REQUIRED_TRAJECTORY_COLUMNS, *LABEL_COLUMNS)


def radius_to_label(radius: float) -> str:
    for supported in SUPPORTED_RADII:
        if np.isclose(radius, supported, rtol=0.0, atol=1e-9):
            return f"hit_r{int(round(supported * 100)):03d}"
    supported = ", ".join(f"{value:.2f}" for value in SUPPORTED_RADII)
    raise ValueError(f"radius must be one of: {supported}")


def make_error_strata(errors: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(errors, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("error values must be finite")
    return np.searchsorted(np.asarray(SUPPORTED_RADII), values, side="left")


def make_label_strata(labels: pd.DataFrame) -> np.ndarray:
    if labels.columns.tolist() != list(LABEL_COLUMNS):
        raise ValueError("label columns do not match the prepared dataset schema")
    return labels.to_numpy(dtype=int).sum(axis=1)


def hash_ids(ids: pd.Series | list[str] | np.ndarray) -> str:
    normalized = "\n".join(sorted(str(value) for value in ids))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_by_error(
    samples: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1")
    train_indices, test_indices = train_test_split(
        np.arange(len(samples)),
        test_size=test_size,
        random_state=seed,
        stratify=make_error_strata(samples["error"]),
    )
    train = samples.iloc[train_indices].sort_values("id").reset_index(drop=True)
    test = samples.iloc[test_indices].sort_values("id").reset_index(drop=True)
    return train, test


def split_train_validation(
    labels: pd.DataFrame,
    valid_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    return train_test_split(
        indices,
        test_size=valid_size,
        random_state=seed,
        stratify=make_label_strata(labels.loc[:, LABEL_COLUMNS]),
    )


def build_metadata(
    source_ids: list[str],
    train_ids: list[str],
    test_ids: list[str],
    seed: int,
    test_size: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "test_size": test_size,
        "counts": {"source": len(source_ids), "train": len(train_ids), "test": len(test_ids)},
        "source_id_sha256": hash_ids(source_ids),
        "split_id_sha256": {
            "train": hash_ids(train_ids),
            "test": hash_ids(test_ids),
        },
        "trajectory_columns": list(REQUIRED_TRAJECTORY_COLUMNS),
        "label_columns": list(LABEL_COLUMNS),
        "dataset_columns": list(DATASET_COLUMNS),
    }


def load_dataset_metadata(path: Path) -> dict[str, Any]:
    metadata = load_json(path)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported dataset schema version: {metadata.get('schema_version')}; "
            f"expected {SCHEMA_VERSION}"
        )
    if metadata.get("dataset_columns") != list(DATASET_COLUMNS):
        raise ValueError("metadata dataset columns do not match the expected schema")
    return metadata


def validate_prepared_sample(frame: pd.DataFrame, source: Path) -> None:
    if frame.columns.tolist() != list(DATASET_COLUMNS):
        raise ValueError(f"{source}: expected columns {list(DATASET_COLUMNS)}")
    validate_trajectory(frame, source)
    label_values = frame.loc[:, LABEL_COLUMNS].to_numpy()
    if not np.isin(label_values, [0, 1]).all():
        raise ValueError(f"{source}: hit labels must contain only 0 or 1")
    if any(frame[column].nunique(dropna=False) != 1 for column in LABEL_COLUMNS):
        raise ValueError(f"{source}: each hit label must be constant within a sample")
    if np.any(np.diff(label_values, axis=1) < 0):
        raise ValueError(f"{source}: radius hit labels must be monotonic")


def load_prepared_trajectories(
    directory: Path,
    metadata_path: Path,
    split: str,
    show_progress: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    metadata = load_dataset_metadata(metadata_path)
    paths = list_csv_files(directory)
    expected_count = metadata["counts"][split]
    if len(paths) != expected_count:
        raise ValueError(f"{directory}: expected {expected_count} CSV files, got {len(paths)}")

    trajectories: dict[str, pd.DataFrame] = {}
    label_rows: list[dict[str, int | str]] = []
    from tqdm.auto import tqdm

    for path in tqdm(paths, desc=f"Loading {split} dataset", unit="file", disable=not show_progress):
        sample_id = path.stem
        if sample_id in trajectories:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        frame = pd.read_csv(path)
        validate_prepared_sample(frame, path)
        trajectories[sample_id] = (
            frame.loc[:, REQUIRED_TRAJECTORY_COLUMNS]
            .sort_values("timestep_ms")
            .reset_index(drop=True)
        )
        label_rows.append({"id": sample_id, **frame.loc[0, LABEL_COLUMNS].astype(int).to_dict()})

    if hash_ids(list(trajectories)) != metadata["split_id_sha256"][split]:
        raise ValueError(f"{directory}: {split} id hash does not match metadata")
    labels = pd.DataFrame(label_rows).sort_values("id").reset_index(drop=True)
    return trajectories, labels, metadata
