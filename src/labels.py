from collections.abc import Mapping

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data_io import ensure_matching_ids
from src.physics import calculate_aim_position, euclidean_distance


def create_hit_label(error: float, radius: float) -> int:
    return int(error <= radius)


def build_hit_labels(
    trajectories: Mapping[str, pd.DataFrame],
    future_labels: pd.DataFrame,
    radius: float,
    show_progress: bool = False,
) -> pd.DataFrame:
    if radius <= 0.0:
        raise ValueError("radius must be greater than 0")
    labels_by_id = future_labels.set_index("id")
    ensure_matching_ids(set(trajectories), set(labels_by_id.index))

    rows: list[dict[str, float | int | str]] = []
    iterator = tqdm(
        trajectories.items(),
        total=len(trajectories),
        desc="Generating hit labels",
        unit="sample",
        disable=not show_progress,
    )
    for sample_id, trajectory in iterator:
        aim = calculate_aim_position(trajectory)
        actual = labels_by_id.loc[sample_id, ["x", "y", "z"]].to_numpy(dtype=float)
        error = euclidean_distance(aim, actual)
        rows.append(
            {
                "id": sample_id,
                "aim_x": float(aim[0]),
                "aim_y": float(aim[1]),
                "aim_z": float(aim[2]),
                "error": error,
                "hit": create_hit_label(error, radius),
            }
        )
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def build_aim_frame(
    trajectories: Mapping[str, pd.DataFrame],
    show_progress: bool = False,
) -> pd.DataFrame:
    rows = []
    iterator = tqdm(
        trajectories.items(),
        total=len(trajectories),
        desc="Calculating aim positions",
        unit="sample",
        disable=not show_progress,
    )
    for sample_id, trajectory in iterator:
        aim = calculate_aim_position(trajectory)
        rows.append({"id": sample_id, "aim_x": aim[0], "aim_y": aim[1], "aim_z": aim[2]})
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
