from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    hit_score: int
    mean_hit_score: float


def probabilities_to_decisions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(probabilities, dtype=float) >= threshold).astype(int)


def calculate_hit_rewards(y_true: np.ndarray, fire_decisions: np.ndarray) -> np.ndarray:
    actual = np.asarray(y_true, dtype=int)
    decisions = np.asarray(fire_decisions, dtype=int)
    if actual.shape != decisions.shape:
        raise ValueError("y_true and fire_decisions must have the same shape")
    if not np.isin(actual, [0, 1]).all() or not np.isin(decisions, [0, 1]).all():
        raise ValueError("y_true and fire_decisions must contain only 0 or 1")
    return np.where(decisions == 0, 0, np.where(actual == 1, 1, -2))


def calculate_hit_score(y_true: np.ndarray, fire_decisions: np.ndarray) -> int:
    return int(calculate_hit_rewards(y_true, fire_decisions).sum())


def calculate_mean_hit_score(y_true: np.ndarray, fire_decisions: np.ndarray) -> float:
    return float(calculate_hit_rewards(y_true, fire_decisions).mean())


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    decisions = probabilities_to_decisions(probabilities, threshold)
    actual = np.asarray(y_true, dtype=int)
    auroc = float(roc_auc_score(actual, probabilities)) if np.unique(actual).size == 2 else None
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(actual, decisions)),
        "precision": float(precision_score(actual, decisions, zero_division=0)),
        "recall": float(recall_score(actual, decisions, zero_division=0)),
        "f1": float(f1_score(actual, decisions, zero_division=0)),
        "auroc": auroc,
        "hit_score": calculate_hit_score(actual, decisions),
        "mean_hit_score": calculate_mean_hit_score(actual, decisions),
        "shots_fired": int(decisions.sum()),
        "samples": int(actual.size),
    }


def make_threshold_candidates(minimum: float, maximum: float, step: float) -> np.ndarray:
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError("threshold bounds must satisfy 0 <= min <= max <= 1")
    if step <= 0.0:
        raise ValueError("threshold step must be greater than 0")
    count = int(np.floor((maximum - minimum) / step))
    candidates = minimum + np.arange(count + 1) * step
    if candidates[-1] < maximum and not np.isclose(candidates[-1], maximum):
        candidates = np.append(candidates, maximum)
    return np.clip(candidates, minimum, maximum)


def find_best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    candidates: np.ndarray,
) -> ThresholdResult:
    if len(candidates) == 0:
        raise ValueError("at least one threshold candidate is required")
    results = []
    for threshold in candidates:
        decisions = probabilities_to_decisions(probabilities, float(threshold))
        score = calculate_hit_score(y_true, decisions)
        results.append((score, float(threshold)))
    best_score, best_threshold = max(results, key=lambda item: (item[0], item[1]))
    return ThresholdResult(
        threshold=best_threshold,
        hit_score=best_score,
        mean_hit_score=best_score / len(y_true),
    )

