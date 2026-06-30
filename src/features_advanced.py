"""Physics-informed features that estimate the reliability of the fixed linear aim.

문제 재정의: 조준점은 `aim = p(0) + v_recent * 0.08`로 고정돼 있고, 모델은 이 조준이
80ms 뒤 반경 안에 들어갈지(=쏠지 말지)를 판단한다. 따라서 가장 직접적인 신호는
"현재 궤적에서 등속 외삽이 80ms 뒤를 얼마나 잘 맞히는가"이다.

핵심 아이디어(extrapolation backtest): 라벨이 만들어진 방식과 동일한 외삽을 과거
구간에서 재현한다. 시점 i에서 v=(p[i]-p[i-1])/0.04로 80ms(=2 step) 뒤를 예측하고,
실제 p[i+2]와의 거리를 잰다. 이 과거 오차들은 미래 외삽 오차의 강력한 대리지표다.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.features import build_trajectory_features


AXES = ("x", "y", "z")
EPSILON = 1e-12
DT = 0.04  # 40ms 간격(초)
HORIZON = 0.08  # 80ms 예측 지평(초)
HORIZON_STEPS = 2  # 80ms = 2 step


def _aim_from(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    return position + velocity * HORIZON


def extrapolation_backtest_errors(positions: np.ndarray) -> np.ndarray:
    """라벨 생성과 동일한 등속 외삽을 과거 구간에서 재현한 오차 배열.

    anchor i in [1, n-1-HORIZON_STEPS]: v=(p[i]-p[i-1])/dt, aim=p[i]+v*0.08,
    error=||aim - p[i+2]||. 마지막 anchor가 현재(0ms)에 가장 가깝다.
    """
    n = len(positions)
    errors = []
    for i in range(1, n - HORIZON_STEPS):
        velocity = (positions[i] - positions[i - 1]) / DT
        aim = _aim_from(positions[i], velocity)
        errors.append(float(np.linalg.norm(aim - positions[i + HORIZON_STEPS])))
    return np.asarray(errors, dtype=float)


def build_advanced_features(trajectory: pd.DataFrame) -> dict[str, float]:
    positions = trajectory.loc[:, AXES].to_numpy(dtype=float)
    velocities = np.diff(positions, axis=0) / DT
    accelerations = np.diff(velocities, axis=0) / DT
    jerks = np.diff(accelerations, axis=0) / DT

    v_recent = velocities[-1]
    a_recent = accelerations[-1]

    features: dict[str, float] = {}

    # 1) Extrapolation backtest: 미래 외삽 오차의 직접 대리지표 (가장 강력한 신호)
    backtest = extrapolation_backtest_errors(positions)
    features["bt_err_last"] = float(backtest[-1])  # 가장 최근(가장 관련 높음)
    features["bt_err_last3_mean"] = float(np.mean(backtest[-3:]))
    features["bt_err_mean"] = float(np.mean(backtest))
    features["bt_err_max"] = float(np.max(backtest))
    features["bt_err_min"] = float(np.min(backtest))
    features["bt_err_std"] = float(np.std(backtest))
    # 추세: 최근 외삽 오차가 커지는 중인지
    features["bt_err_trend"] = float(backtest[-1] - backtest[0])

    # 2) 등속 vs 등가속 조준점 간극: 0.5*|a|*h^2 만큼 선형 조준이 휜다
    curvature_gap = 0.5 * a_recent * HORIZON * HORIZON
    features["aim_curvature_gap"] = float(np.linalg.norm(curvature_gap))
    features["recent_accel_norm"] = float(np.linalg.norm(a_recent))

    # 3) Jerk(3차 미분): 가속도 자체가 변하면 등가속 보정도 신뢰 어렵다
    jerk_norms = np.linalg.norm(jerks, axis=1)
    features["recent_jerk_norm"] = float(jerk_norms[-1])
    features["jerk_norm_mean"] = float(np.mean(jerk_norms))
    features["jerk_norm_max"] = float(np.max(jerk_norms))

    # 4) 속도 일관성: 단구간 속도와 장구간 평균속도가 어긋나면 등속 가정이 깨진다
    v_long = (positions[-1] - positions[-3]) / (2 * DT)
    features["recent_vel_mismatch"] = float(np.linalg.norm(v_recent - v_long))
    speeds = np.linalg.norm(velocities, axis=1)
    features["speed_recent"] = float(speeds[-1])
    features["speed_recent_ratio"] = float(speeds[-1] / max(np.mean(speeds), EPSILON))
    features["speed_recent_change"] = float(speeds[-1] - speeds[-2])

    # 5) 최근 방향 안정성: 마지막 속도와 직전 3개 평균 속도 사이 각도
    recent_dirs = velocities[-3:]
    mean_dir = np.mean(recent_dirs, axis=0)
    denom = np.linalg.norm(v_recent) * np.linalg.norm(mean_dir)
    cos = np.dot(v_recent, mean_dir) / denom if denom > EPSILON else 1.0
    features["recent_turn_angle"] = float(np.arccos(np.clip(cos, -1.0, 1.0)))

    return features


def build_feature_frame_advanced(
    trajectories: Mapping[str, pd.DataFrame],
    include_baseline: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    """baseline 114 특징(옵션) + 물리 기반 외삽 신뢰도 특징을 합친 프레임."""
    rows = []
    for sample_id, trajectory in tqdm(
        trajectories.items(),
        total=len(trajectories),
        desc="Extracting advanced features",
        unit="sample",
        disable=not show_progress,
    ):
        row: dict[str, float | str] = {"id": sample_id}
        if include_baseline:
            row.update(build_trajectory_features(trajectory))
        row.update(build_advanced_features(trajectory))
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    if frame.drop(columns="id").isna().any().any():
        raise ValueError("Generated features contain NaN values")
    return frame
