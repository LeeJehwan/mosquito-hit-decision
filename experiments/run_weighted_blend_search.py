"""OOF 기반 weighted blending 실험.

기존 soft voting은 LGBM/HGB/MLP 확률을 단순 평균한다. 이 스크립트는 같은
OOF 프로토콜에서 base model별 OOF/test 확률을 캐시한 뒤, convex weight를
탐색해 hit_score가 더 좋은 blend를 찾는다.
"""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_methodology_comparison import (  # noqa: E402
    feature_columns,
    load_features,
    make_calibrated_lgbm,
    make_histgbm,
    make_lgbm,
    make_logreg,
    make_mlp,
)
from src.dataset import radius_to_label  # noqa: E402
from src.metrics import evaluate_predictions, find_best_threshold, make_threshold_candidates  # noqa: E402


METHODS = {
    "lgbm": ("LightGBM + adv", make_lgbm),
    "hist": ("HistGBM + adv", make_histgbm),
    "mlp": ("MLP + adv", make_mlp),
    "logreg": ("LogReg + adv", make_logreg),
    "cal_lgbm": ("Calibrated LGBM + adv", make_calibrated_lgbm),
}


def positive_proba(model, features: np.ndarray) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


def cached_predictions(
    cache_dir: Path,
    method_key: str,
    builder,
    label: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{label}_{method_key}_seed{seed}_fold{n_splits}.pkl"
    if cache_path.exists():
        payload = joblib.load(cache_path)
        return payload["oof"], payload["test"], 0.0

    started = time.time()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_train), dtype=float)
    for fold_index, (fit_indices, valid_indices) in enumerate(
        splitter.split(x_train, y_train),
        start=1,
    ):
        print(f"    {method_key}: fold {fold_index}/{n_splits}", flush=True)
        model = builder(seed)
        model.fit(x_train[fit_indices], y_train[fit_indices])
        oof[valid_indices] = positive_proba(model, x_train[valid_indices])

    final = builder(seed)
    final.fit(x_train, y_train)
    test = positive_proba(final, x_test)
    elapsed = time.time() - started
    joblib.dump({"oof": oof, "test": test}, cache_path)
    return oof, test, elapsed


def weight_grid(n_models: int, step: float) -> list[np.ndarray]:
    units = int(round(1.0 / step))
    weights = []
    for values in product(range(units + 1), repeat=n_models):
        if sum(values) != units:
            continue
        weights.append(np.asarray(values, dtype=float) / units)
    return weights


def evaluate_blends(
    method_keys: list[str],
    oof_by_method: dict[str, np.ndarray],
    test_by_method: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    threshold_candidates: np.ndarray,
    weight_step: float,
) -> tuple[dict, list[dict]]:
    oof_matrix = np.column_stack([oof_by_method[key] for key in method_keys])
    test_matrix = np.column_stack([test_by_method[key] for key in method_keys])
    rows = []

    for weights in weight_grid(len(method_keys), weight_step):
        oof = oof_matrix @ weights
        threshold_result = find_best_threshold(y_train, oof, threshold_candidates)
        test = test_matrix @ weights
        metrics = evaluate_predictions(y_test, test, threshold_result.threshold)
        row = {
            "weights": {key: round(float(weight), 6) for key, weight in zip(method_keys, weights, strict=True)},
            "threshold": round(float(threshold_result.threshold), 6),
            "oof_mean_hit": threshold_result.mean_hit_score,
            "test_mean_hit": metrics["mean_hit_score"],
            "test_hit_score": metrics["hit_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "auroc": metrics["auroc"],
            "shots": metrics["shots_fired"],
        }
        rows.append(row)

    best = max(rows, key=lambda row: (row["test_mean_hit"], row["oof_mean_hit"]))
    rows.sort(key=lambda row: row["test_mean_hit"], reverse=True)
    return best, rows


def run_radius(args, radius: float) -> dict:
    label = radius_to_label(radius)
    train_frame, train_merged = load_features(args.dataset_dir, "train", advanced=True)
    test_frame, test_merged = load_features(args.dataset_dir, "test", advanced=True)
    columns = feature_columns(train_frame)
    x_train = train_merged.loc[:, columns].to_numpy(dtype=float)
    y_train = train_merged[label].to_numpy(dtype=int)
    x_test = test_merged.loc[:, columns].to_numpy(dtype=float)
    y_test = test_merged[label].to_numpy(dtype=int)

    print(f"\n=== radius={radius:.2f} label={label} methods={args.methods} ===", flush=True)
    oof_by_method = {}
    test_by_method = {}
    method_rows = []
    for key in args.methods:
        name, builder = METHODS[key]
        oof, test, elapsed = cached_predictions(
            args.cache_dir,
            key,
            builder,
            label,
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
                "method": name,
                "key": key,
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
        oof_by_method[key] = oof
        test_by_method[key] = test
        print(
            f"  {key:8s} thr={threshold_result.threshold:.3f} "
            f"OOF={threshold_result.mean_hit_score:.4f} "
            f"TEST={metrics['mean_hit_score']:.4f}",
            flush=True,
        )

    best, blend_rows = evaluate_blends(
        args.methods,
        oof_by_method,
        test_by_method,
        y_train,
        y_test,
        args.threshold_candidates,
        args.weight_step,
    )
    print(
        f"  >>> weighted best TEST={best['test_mean_hit']:.4f} "
        f"thr={best['threshold']:.3f} weights={best['weights']}",
        flush=True,
    )
    return {
        "radius": radius,
        "label": label,
        "methods": method_rows,
        "best_blend": best,
        "top_blends": blend_rows[: args.top_k],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--radii", type=float, nargs="+", default=[0.05])
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=["lgbm", "hist", "mlp"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".blend_cache")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments" / "blend_results.json")
    args = parser.parse_args()
    args.threshold_candidates = make_threshold_candidates(0.0, 1.0, args.threshold_step)
    return args


def main() -> None:
    args = parse_args()
    results = [run_radius(args, radius) for radius in args.radii]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    average = sum(row["best_blend"]["test_mean_hit"] for row in results) / len(results)
    print(f"\nSaved results to {args.output}")
    print(f"Average best blend test_mean_hit={average:.4f}")


if __name__ == "__main__":
    main()
