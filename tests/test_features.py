import numpy as np
import pandas as pd

from src.features import align_feature_columns, build_feature_frame, build_trajectory_features


def make_trajectory() -> pd.DataFrame:
    timesteps = np.arange(-400, 1, 40)
    return pd.DataFrame(
        {
            "timestep_ms": timesteps,
            "x": timesteps * 0.001,
            "y": timesteps * 0.002,
            "z": timesteps * -0.001,
        }
    )


def test_feature_generation_is_finite() -> None:
    features = build_trajectory_features(make_trajectory())

    assert len([key for key in features if key.startswith("position_")]) == 33
    assert np.isfinite(list(features.values())).all()


def test_feature_alignment_preserves_training_order() -> None:
    frame = build_feature_frame({"TRAIN_1": make_trajectory()})
    columns = list(reversed([column for column in frame.columns if column != "id"]))

    aligned = align_feature_columns(frame, columns)

    assert aligned.columns.tolist() == columns

