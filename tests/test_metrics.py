import numpy as np

from src.metrics import (
    calculate_hit_rewards,
    calculate_hit_score,
    find_best_threshold,
    make_threshold_candidates,
)


def test_hit_score_applies_reward_rules() -> None:
    actual = np.array([1, 0, 1, 0])
    decisions = np.array([1, 1, 0, 0])

    assert calculate_hit_rewards(actual, decisions).tolist() == [1, -2, 0, 0]
    assert calculate_hit_score(actual, decisions) == -1


def test_best_threshold_maximizes_hit_score() -> None:
    actual = np.array([1, 0, 1])
    probabilities = np.array([0.9, 0.8, 0.7])
    candidates = np.array([0.5, 0.75, 0.95])

    result = find_best_threshold(actual, probabilities, candidates)

    assert result.threshold == 0.95
    assert result.hit_score == 0


def test_threshold_candidates_include_maximum() -> None:
    candidates = make_threshold_candidates(0.0, 1.0, 0.3)

    assert candidates[-1] == 1.0

