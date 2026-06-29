import numpy as np
import pandas as pd


AXES = ("x", "y", "z")


def get_position_at(trajectory: pd.DataFrame, timestep_ms: int) -> np.ndarray:
    rows = trajectory.loc[trajectory["timestep_ms"] == timestep_ms, AXES]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one position at {timestep_ms}ms, got {len(rows)}")
    return rows.iloc[0].to_numpy(dtype=float)


def calculate_recent_velocity(trajectory: pd.DataFrame) -> np.ndarray:
    previous = get_position_at(trajectory, -40)
    current = get_position_at(trajectory, 0)
    return (current - previous) / 0.04


def calculate_aim_position(trajectory: pd.DataFrame, horizon_seconds: float = 0.08) -> np.ndarray:
    current = get_position_at(trajectory, 0)
    return current + calculate_recent_velocity(trajectory) * horizon_seconds


def euclidean_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)))

