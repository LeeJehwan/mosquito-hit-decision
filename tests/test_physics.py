import numpy as np
import pandas as pd

from src.physics import calculate_aim_position, calculate_recent_velocity, euclidean_distance


def make_linear_trajectory() -> pd.DataFrame:
    timesteps = np.arange(-400, 1, 40)
    seconds = timesteps / 1000.0
    return pd.DataFrame(
        {
            "timestep_ms": timesteps,
            "x": 1.0 + 2.0 * seconds,
            "y": 2.0 - seconds,
            "z": 3.0 + 0.5 * seconds,
        }
    )


def test_recent_velocity_uses_last_40ms() -> None:
    velocity = calculate_recent_velocity(make_linear_trajectory())

    np.testing.assert_allclose(velocity, [2.0, -1.0, 0.5])


def test_aim_position_projects_80ms() -> None:
    aim = calculate_aim_position(make_linear_trajectory())

    np.testing.assert_allclose(aim, [1.16, 1.92, 3.04])
    assert euclidean_distance(aim, aim) == 0.0

