from pathlib import Path

import pandas as pd
import pytest

from src.data_io import ensure_matching_ids, load_trajectory


def test_load_trajectory_rejects_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    pd.DataFrame({"timestep_ms": range(11), "x": 0, "y": 0}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_trajectory(path)


def test_load_trajectory_rejects_wrong_timestep_count(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    pd.DataFrame(
        {"timestep_ms": range(10), "x": 0.0, "y": 0.0, "z": 0.0}
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="expected 11 timesteps"):
        load_trajectory(path)


def test_matching_ids_reports_mismatch() -> None:
    with pytest.raises(ValueError, match="missing labels=.*TRAIN_2"):
        ensure_matching_ids({"TRAIN_1", "TRAIN_2"}, {"TRAIN_1"})

