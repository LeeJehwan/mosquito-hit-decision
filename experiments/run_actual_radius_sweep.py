"""Actual classifier retraining sweep for radius cost-efficiency analysis.

This script intentionally retrains the same classifier family for every
candidate radius. It is slower than an error-regression approximation, but it
matches the real train/test protocol:

1. Recompute hit labels for each arbitrary radius from continuous miss error.
2. Build OOF probabilities on train with LGBM/HGB/MLP classifiers.
3. Select blend weights and firing threshold by train OOF mean_hit_score only.
4. Train final base models on all train and evaluate on held-out test.
"""

import argparse
import json
import math
import sys
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_methodology_comparison import (  # noqa: E402
    feature_columns,
    load_features,
    make_histgbm,
    make_lgbm,
    make_mlp,
)
from src.data_io import load_future_labels, save_dataframe, save_json  # noqa: E402
from src.dataset import load_prepared_trajectories  # noqa: E402
from src.labels import build_hit_labels  # noqa: E402
from src.metrics import evaluate_predictions, find_best_threshold, make_threshold_candidates  # noqa: E402


METHODS = {
    "lgbm": ("LightGBM + adv", make_lgbm),
    "hist": ("HistGBM + adv", make_histgbm),
    "mlp": ("MLP + adv", make_mlp),
}


def radius_grid(minimum: float, maximum: float, step: float) -> np.ndarray:
    if minimum <= 0.0 or maximum <= 0.0:
        raise ValueError("radius bounds must be positive")
    if minimum > maximum:
        raise ValueError("radius-min must not exceed radius-max")
    if step <= 0.0:
        raise ValueError("radius-step must be positive")
    count = int(np.floor((maximum - minimum) / step))
    values = minimum + np.arange(count + 1) * step
    if values[-1] < maximum and not np.isclose(values[-1], maximum):
        values = np.append(values, maximum)
    return np.round(values, 10)


def weight_grid(n_models: int, step: float) -> list[np.ndarray]:
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("weight-step must divide 1.0")
    weights = []
    for values in product(range(units + 1), repeat=n_models):
        if sum(values) == units:
            weights.append(np.asarray(values, dtype=float) / units)
    return weights


def radius_key(radius: float) -> str:
    return f"r{radius:.5f}".replace(".", "p")


def dataframe_to_markdown(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else format(value, floatfmt))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_errors_for_split(dataset_dir: Path, source_dir: Path, split: str) -> pd.DataFrame:
    trajectories, _, _ = load_prepared_trajectories(
        dataset_dir / split,
        dataset_dir / "metadata.json",
        split=split,
        show_progress=False,
    )
    future_labels = load_future_labels(source_dir / "train_labels.csv")
    future_labels = future_labels[future_labels["id"].isin(trajectories)].reset_index(drop=True)
    error_frame = build_hit_labels(trajectories, future_labels, radius=1.0, show_progress=False)
    return error_frame.loc[:, ["id", "error"]].sort_values("id").reset_index(drop=True)


def load_feature_and_error_frames(
    dataset_dir: Path,
    source_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_frame, train_merged = load_features(dataset_dir, "train", advanced=True)
    test_frame, test_merged = load_features(dataset_dir, "test", advanced=True)
    columns = feature_columns(train_frame)

    train_errors = load_errors_for_split(dataset_dir, source_dir, "train")
    test_errors = load_errors_for_split(dataset_dir, source_dir, "test")
    train_merged = train_merged.merge(train_errors, on="id", validate="one_to_one")
    test_merged = test_merged.merge(test_errors, on="id", validate="one_to_one")
    return train_merged, test_merged, columns


def positive_proba(model, features: np.ndarray) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


def cached_predictions(
    cache_dir: Path,
    method_key: str,
    radius: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    cv_folds: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{radius_key(radius)}_{method_key}_seed{seed}_fold{cv_folds}.pkl"
    if cache_path.exists():
        payload = joblib.load(cache_path)
        return payload["oof"], payload["test"], 0.0

    if np.unique(y_train).size != 2:
        raise ValueError(f"radius={radius:.5f} does not contain both classes")
    smallest_class = int(np.bincount(y_train.astype(int)).min())
    if smallest_class < cv_folds:
        raise ValueError(
            f"radius={radius:.5f}: smallest class count {smallest_class} < cv_folds {cv_folds}"
        )

    started = time.time()
    _, builder = METHODS[method_key]
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof = np.empty(len(y_train), dtype=float)
    for fold_index, (fit_indices, valid_indices) in enumerate(
        splitter.split(x_train, y_train),
        start=1,
    ):
        print(
            f"    radius={radius:.5f} {method_key}: fold {fold_index}/{cv_folds}",
            flush=True,
        )
        model = builder(seed)
        model.fit(x_train[fit_indices], y_train[fit_indices])
        oof[valid_indices] = positive_proba(model, x_train[valid_indices])

    final_model = builder(seed)
    final_model.fit(x_train, y_train)
    test = positive_proba(final_model, x_test)
    elapsed = time.time() - started
    joblib.dump({"oof": oof, "test": test}, cache_path)
    return oof, test, elapsed


def evaluate_blends(
    method_keys: list[str],
    oof_by_method: dict[str, np.ndarray],
    test_by_method: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    threshold_candidates: np.ndarray,
    weights: list[np.ndarray],
) -> tuple[dict, dict, list[dict]]:
    oof_matrix = np.column_stack([oof_by_method[key] for key in method_keys])
    test_matrix = np.column_stack([test_by_method[key] for key in method_keys])
    rows = []

    for weight in weights:
        oof = oof_matrix @ weight
        threshold_result = find_best_threshold(y_train, oof, threshold_candidates)
        oof_metrics = evaluate_predictions(y_train, oof, threshold_result.threshold)
        test = test_matrix @ weight
        test_metrics = evaluate_predictions(y_test, test, threshold_result.threshold)
        row = {
            "weights": {
                key: round(float(value), 6)
                for key, value in zip(method_keys, weight, strict=True)
            },
            "threshold": round(float(threshold_result.threshold), 6),
            "oof_mean_hit": threshold_result.mean_hit_score,
            "oof_hit_score": threshold_result.hit_score,
            "oof_shots_fired": oof_metrics["shots_fired"],
            "test_mean_hit": test_metrics["mean_hit_score"],
            "test_hit_score": test_metrics["hit_score"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_auroc": test_metrics["auroc"],
            "test_shots_fired": test_metrics["shots_fired"],
        }
        rows.append(row)

    selected = max(rows, key=lambda row: (row["oof_mean_hit"], -row["oof_shots_fired"]))
    test_oracle = max(rows, key=lambda row: row["test_mean_hit"])
    rows.sort(key=lambda row: (row["oof_mean_hit"], row["test_mean_hit"]), reverse=True)
    return selected, test_oracle, rows


def add_marginal_columns(rows: list[dict]) -> None:
    previous = None
    for row in rows:
        if previous is None:
            row["delta_mean_hit"] = None
            row["delta_oracle_mean_hit"] = None
            row["roi_per_radius"] = None
            row["roi_per_area"] = None
        else:
            delta_score = row["selected_test_mean_hit"] - previous["selected_test_mean_hit"]
            delta_oracle = row["oracle_mean_hit"] - previous["oracle_mean_hit"]
            delta_radius = row["radius"] - previous["radius"]
            delta_area = row["area"] - previous["area"]
            row["delta_mean_hit"] = delta_score
            row["delta_oracle_mean_hit"] = delta_oracle
            row["roi_per_radius"] = delta_score / delta_radius
            row["roi_per_area"] = delta_score / delta_area
        previous = row


def summarize(rows: list[dict]) -> dict:
    scores = np.asarray([row["selected_test_mean_hit"] for row in rows], dtype=float)
    radii = np.asarray([row["radius"] for row in rows], dtype=float)
    score_range = np.ptp(scores)
    if score_range == 0.0 or np.ptp(radii) == 0.0:
        knee_index = int(np.argmax(scores))
    else:
        normalized_x = (radii - radii.min()) / np.ptp(radii)
        normalized_y = (scores - scores.min()) / score_range
        knee_index = int(np.argmax(normalized_y - normalized_x))

    max_gain = float(scores.max() - scores[0])
    if max_gain <= 0.0:
        gain95_index = int(np.argmax(scores))
    else:
        target = scores[0] + 0.95 * max_gain
        gain95_index = int(np.argmax(scores >= target))

    marginal_rows = [row for row in rows if row["roi_per_radius"] is not None]
    positive_marginal_rows = [row for row in marginal_rows if row["delta_mean_hit"] > 0]
    best_marginal = max(
        positive_marginal_rows or marginal_rows,
        key=lambda row: row["roi_per_radius"],
    )
    return {
        "best_raw_score": rows[int(np.argmax(scores))],
        "knee_point": rows[knee_index],
        "first_95pct_of_max_gain": rows[gain95_index],
        "best_positive_marginal_interval_end": best_marginal,
    }


def run_radius(args, train_frame, test_frame, feature_cols, radius, weights):
    x_train = train_frame.loc[:, feature_cols].to_numpy(dtype=float)
    y_train = (train_frame["error"].to_numpy(dtype=float) <= radius).astype(int)
    x_test = test_frame.loc[:, feature_cols].to_numpy(dtype=float)
    y_test = (test_frame["error"].to_numpy(dtype=float) <= radius).astype(int)

    print(
        f"\n=== radius={radius:.5f} "
        f"train_pos={y_train.mean():.4f} test_pos={y_test.mean():.4f} ===",
        flush=True,
    )
    oof_by_method = {}
    test_by_method = {}
    method_rows = []
    for method_key, (method_name, _) in METHODS.items():
        oof, test, elapsed = cached_predictions(
            args.cache_dir,
            method_key,
            radius,
            x_train,
            y_train,
            x_test,
            args.seed,
            args.cv_folds,
        )
        threshold_result = find_best_threshold(y_train, oof, args.threshold_candidates)
        metrics = evaluate_predictions(y_test, test, threshold_result.threshold)
        method_rows.append(
            {
                "method": method_name,
                "key": method_key,
                "threshold": threshold_result.threshold,
                "oof_mean_hit": threshold_result.mean_hit_score,
                "test_mean_hit": metrics["mean_hit_score"],
                "test_hit_score": metrics["hit_score"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "auroc": metrics["auroc"],
                "shots": metrics["shots_fired"],
                "seconds": round(elapsed, 1),
            }
        )
        oof_by_method[method_key] = oof
        test_by_method[method_key] = test
        print(
            f"  {method_key:5s} OOF={threshold_result.mean_hit_score:.4f} "
            f"TEST={metrics['mean_hit_score']:.4f} sec={elapsed:.1f}",
            flush=True,
        )

    selected, test_oracle, top_blends = evaluate_blends(
        list(METHODS),
        oof_by_method,
        test_by_method,
        y_train,
        y_test,
        args.threshold_candidates,
        weights,
    )
    print(
        f"  >>> selected OOF={selected['oof_mean_hit']:.4f} "
        f"TEST={selected['test_mean_hit']:.4f} thr={selected['threshold']:.3f} "
        f"weights={selected['weights']}",
        flush=True,
    )
    return {
        "radius": radius,
        "area": float(math.pi * radius * radius),
        "train_hit_rate": float(y_train.mean()),
        "test_hit_rate": float(y_test.mean()),
        "fire_all_mean_hit": float(3.0 * y_test.mean() - 2.0),
        "oracle_mean_hit": float(y_test.mean()),
        "selected_threshold": selected["threshold"],
        "selected_weights": selected["weights"],
        "selected_oof_mean_hit": selected["oof_mean_hit"],
        "selected_oof_hit_score": selected["oof_hit_score"],
        "selected_test_mean_hit": selected["test_mean_hit"],
        "selected_test_hit_score": selected["test_hit_score"],
        "selected_test_precision": selected["test_precision"],
        "selected_test_recall": selected["test_recall"],
        "selected_test_auroc": selected["test_auroc"],
        "selected_test_shots_fired": selected["test_shots_fired"],
        "test_oracle_blend_mean_hit": test_oracle["test_mean_hit"],
        "test_oracle_blend_weights": test_oracle["weights"],
        "methods": method_rows,
        "top_oof_blends": top_blends[: args.top_k],
    }


def write_report(path: Path, payload: dict) -> None:
    rows = payload["results"]
    summary = payload["summary"]
    table_columns = [
        "radius",
        "selected_test_mean_hit",
        "delta_mean_hit",
        "roi_per_radius",
        "test_hit_rate",
        "selected_test_precision",
        "selected_test_recall",
        "selected_test_shots_fired",
        "selected_threshold",
    ]
    table = pd.DataFrame(rows).loc[:, table_columns]
    table_md = dataframe_to_markdown(table)

    best = summary["best_raw_score"]
    knee = summary["knee_point"]
    gain95 = summary["first_95pct_of_max_gain"]
    marginal = summary["best_positive_marginal_interval_end"]
    config = payload["config"]
    text = f"""# Actual Radius Sweep Report

## 목적

반경 r이 커지면 명중 가능 구간이 커지므로 raw `mean_hit_score`는 대체로 올라간다. 하지만 반경 확대에 비용이 있다면 중요한 값은 절대 점수와 함께 **r을 한 단계 늘렸을 때 추가로 얻는 mean_hit**이다.

## 실험 프로토콜

- 데이터: `dataset/train` 8,000개로 OOF 선택, `dataset/test` 2,000개로 최종 평가.
- 라벨: `open/train_labels.csv`의 실제 80ms 후 좌표와 고정 조준점 사이 거리(`error`)를 재계산하고, 각 후보는 `hit = error <= radius`로 정의.
- 모델: 각 반경마다 LightGBM + HistGradientBoosting + MLP 분류기를 실제로 다시 학습.
- 선택: train 5-fold OOF 확률만 사용해 blend weight와 발사 threshold를 선택. test 점수는 선택에 사용하지 않음.
- 탐색 범위: radius `{config['radius_min']:.4f}` ~ `{config['radius_max']:.4f}`, step `{config['radius_step']:.4f}`.

## 핵심 결과

- 비용 무시 최고점: radius `{best['radius']:.5f}` / test mean_hit `{best['selected_test_mean_hit']:.4f}`.
- knee point: radius `{knee['radius']:.5f}` / test mean_hit `{knee['selected_test_mean_hit']:.4f}`.
- 최대 점수 상승분의 95%에 처음 도달: radius `{gain95['radius']:.5f}` / test mean_hit `{gain95['selected_test_mean_hit']:.4f}`.
- 가장 큰 양의 marginal ROI 구간 끝점: radius `{marginal['radius']:.5f}` / ROI per radius `{marginal['roi_per_radius']:.4f}`.

비용 계수가 정해지면 `selected_test_mean_hit - cost_per_radius * radius` 또는 `selected_test_mean_hit - cost_per_area * area`를 최대화하는 지점을 고르면 된다. 비용 계수가 아직 없다면 knee point와 95% gain 지점을 함께 제시하는 것이 가장 방어적이다.

## 결과 테이블

{table_md}

## 주의

이 결과는 실제 재학습 기반이지만, 반경별 weight 선택은 OOF 성능으로만 했다. `test_oracle_blend_*` 값은 JSON에 진단용으로만 저장되어 있으며 발표 결론에는 사용하지 않는 것이 맞다.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "open")
    parser.add_argument("--radius-min", type=float, default=0.01)
    parser.add_argument("--radius-max", type=float, default=0.10)
    parser.add_argument("--radius-step", type=float, default=0.005)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".actual_radius_sweep_cache")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "actual_radius_sweep_results.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "actual_radius_sweep_results.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "ACTUAL_RADIUS_SWEEP_REPORT.md",
    )
    args = parser.parse_args()
    args.threshold_candidates = make_threshold_candidates(0.0, 1.0, args.threshold_step)
    return args


def main() -> None:
    args = parse_args()
    radii = radius_grid(args.radius_min, args.radius_max, args.radius_step)
    print(
        f"Actual retraining sweep: {len(radii)} radii "
        f"from {radii[0]:.5f} to {radii[-1]:.5f}",
        flush=True,
    )
    train_frame, test_frame, feature_cols = load_feature_and_error_frames(
        args.dataset_dir,
        args.source_dir,
    )
    weights = weight_grid(len(METHODS), args.weight_step)
    rows = [
        run_radius(args, train_frame, test_frame, feature_cols, radius, weights)
        for radius in radii
    ]
    add_marginal_columns(rows)
    payload = {
        "config": {
            "radius_min": args.radius_min,
            "radius_max": args.radius_max,
            "radius_step": args.radius_step,
            "threshold_step": args.threshold_step,
            "weight_step": args.weight_step,
            "cv_folds": args.cv_folds,
            "seed": args.seed,
            "feature_count": len(feature_cols),
            "methods": list(METHODS),
            "selection_protocol": "blend weight and threshold selected by train OOF only",
        },
        "summary": summarize(rows),
        "results": rows,
    }
    save_json(payload, args.output_json)
    save_dataframe(pd.DataFrame(rows).drop(columns=["methods", "top_oof_blends"]), args.output_csv)
    write_report(args.report_output, payload)
    print(f"\nSaved JSON to {args.output_json}")
    print(f"Saved CSV to {args.output_csv}")
    print(f"Saved report to {args.report_output}")
    print(
        "Best raw score: "
        f"radius={payload['summary']['best_raw_score']['radius']:.5f}, "
        f"mean_hit={payload['summary']['best_raw_score']['selected_test_mean_hit']:.4f}"
    )
    print(
        "Knee point: "
        f"radius={payload['summary']['knee_point']['radius']:.5f}, "
        f"mean_hit={payload['summary']['knee_point']['selected_test_mean_hit']:.4f}"
    )


if __name__ == "__main__":
    main()
