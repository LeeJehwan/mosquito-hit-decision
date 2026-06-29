import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


REQUIRED_TRAJECTORY_COLUMNS = ("timestep_ms", "x", "y", "z")
EXPECTED_TIMESTEP_COUNT = 11


def list_csv_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"CSV directory does not exist: {directory}")
    paths = sorted(directory.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in: {directory}")
    return paths


def extract_sample_id(path: Path) -> str:
    return path.stem


def validate_trajectory(frame: pd.DataFrame, source: Path) -> None:
    missing = [column for column in REQUIRED_TRAJECTORY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")
    if len(frame) != EXPECTED_TIMESTEP_COUNT:
        raise ValueError(
            f"{source}: expected {EXPECTED_TIMESTEP_COUNT} timesteps, got {len(frame)}"
        )
    values = frame.loc[:, REQUIRED_TRAJECTORY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: trajectory contains non-finite values")
    if frame["timestep_ms"].duplicated().any():
        raise ValueError(f"{source}: timestep_ms values must be unique")


def load_trajectory(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_trajectory(frame, path)
    return frame.loc[:, REQUIRED_TRAJECTORY_COLUMNS].sort_values("timestep_ms").reset_index(drop=True)


def load_trajectories(
    directory: Path,
    show_progress: bool = False,
    description: str = "Loading trajectories",
) -> dict[str, pd.DataFrame]:
    trajectories: dict[str, pd.DataFrame] = {}
    paths = list_csv_files(directory)
    for path in tqdm(paths, desc=description, unit="file", disable=not show_progress):
        sample_id = extract_sample_id(path)
        if sample_id in trajectories:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        trajectories[sample_id] = load_trajectory(path)
    return trajectories


def load_future_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ("id", "x", "y", "z")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate ids are not allowed")
    coordinates = frame.loc[:, ("x", "y", "z")].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{path}: labels contain non-finite coordinates")
    return frame.loc[:, required].copy()


def ensure_matching_ids(trajectory_ids: set[str], label_ids: set[str]) -> None:
    missing_labels = sorted(trajectory_ids - label_ids)
    missing_trajectories = sorted(label_ids - trajectory_ids)
    if missing_labels or missing_trajectories:
        raise ValueError(
            "Trajectory/label id mismatch. "
            f"missing labels={missing_labels[:10]}, "
            f"missing trajectories={missing_trajectories[:10]}"
        )


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    ensure_parent_directory(path)
    frame.to_csv(path, index=False)


def save_json(data: Any, path: Path) -> None:
    ensure_parent_directory(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
