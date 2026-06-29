from collections.abc import Mapping

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


AXES = ("x", "y", "z")
EPSILON = 1e-12


def calculate_velocities(trajectory: pd.DataFrame) -> np.ndarray:
    positions = trajectory.loc[:, AXES].to_numpy(dtype=float)
    elapsed_seconds = np.diff(trajectory["timestep_ms"].to_numpy(dtype=float)) / 1000.0
    if np.any(elapsed_seconds <= 0.0):
        raise ValueError("timestep_ms values must be strictly increasing")
    return np.diff(positions, axis=0) / elapsed_seconds[:, None]


def calculate_accelerations(trajectory: pd.DataFrame, velocities: np.ndarray) -> np.ndarray:
    elapsed_seconds = np.diff(trajectory["timestep_ms"].to_numpy(dtype=float)) / 1000.0
    velocity_elapsed = (elapsed_seconds[:-1] + elapsed_seconds[1:]) / 2.0
    return np.diff(velocities, axis=0) / velocity_elapsed[:, None]


def calculate_turn_angles(velocities: np.ndarray) -> np.ndarray:
    first = velocities[:-1]
    second = velocities[1:]
    denominators = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosines = np.divide(
        np.sum(first * second, axis=1),
        denominators,
        out=np.ones_like(denominators),
        where=denominators > EPSILON,
    )
    return np.arccos(np.clip(cosines, -1.0, 1.0))


def _add_matrix_features(features: dict[str, float], prefix: str, values: np.ndarray) -> None:
    for index, row in enumerate(values):
        for axis, value in zip(AXES, row, strict=True):
            features[f"{prefix}_{index:02d}_{axis}"] = float(value)


def _add_vector_features(features: dict[str, float], prefix: str, values: np.ndarray) -> None:
    for axis, value in zip(AXES, values, strict=True):
        features[f"{prefix}_{axis}"] = float(value)


def _add_summary_features(features: dict[str, float], prefix: str, values: np.ndarray) -> None:
    features[f"{prefix}_min"] = float(np.min(values))
    features[f"{prefix}_max"] = float(np.max(values))
    features[f"{prefix}_mean"] = float(np.mean(values))
    features[f"{prefix}_std"] = float(np.std(values))


def build_trajectory_features(trajectory: pd.DataFrame) -> dict[str, float]:
    positions = trajectory.loc[:, AXES].to_numpy(dtype=float)
    velocities = calculate_velocities(trajectory)
    accelerations = calculate_accelerations(trajectory, velocities)
    speeds = np.linalg.norm(velocities, axis=1)
    acceleration_norms = np.linalg.norm(accelerations, axis=1)
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    path_length = float(np.sum(segment_lengths))
    straight_distance = float(np.linalg.norm(positions[-1] - positions[0]))
    turn_angles = calculate_turn_angles(velocities)

    features: dict[str, float] = {}
    _add_matrix_features(features, "position", positions)
    _add_matrix_features(features, "velocity", velocities)
    _add_matrix_features(features, "acceleration", accelerations)
    _add_vector_features(features, "recent_velocity", velocities[-1])
    _add_vector_features(features, "mean_velocity", np.mean(velocities, axis=0))
    _add_vector_features(features, "recent_acceleration", accelerations[-1])
    _add_summary_features(features, "speed", speeds)
    _add_summary_features(features, "acceleration_norm", acceleration_norms)
    _add_summary_features(features, "turn_angle", turn_angles)
    features["path_length"] = path_length
    features["straight_distance"] = straight_distance
    features["tortuosity"] = path_length / max(straight_distance, EPSILON)
    return features


def build_feature_frame(
    trajectories: Mapping[str, pd.DataFrame],
    show_progress: bool = False,
) -> pd.DataFrame:
    rows = [
        {"id": sample_id, **build_trajectory_features(trajectory)}
        for sample_id, trajectory in tqdm(
            trajectories.items(),
            total=len(trajectories),
            desc="Extracting features",
            unit="sample",
            disable=not show_progress,
        )
    ]
    frame = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    if frame.drop(columns="id").isna().any().any():
        raise ValueError("Generated features contain NaN values")
    return frame


def align_feature_columns(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = sorted(set(feature_columns) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(feature_columns) - {"id"})
    if missing or extra:
        raise ValueError(f"Feature schema mismatch. missing={missing}, extra={extra}")
    return frame.loc[:, feature_columns]
