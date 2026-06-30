"""명중률(hit_score) 향상 방법론 비교 시뮬레이션.

평가 프로토콜
- train 8,000개에 대해 StratifiedKFold(5) OOF 확률을 만든다.
- OOF에서 hit_score를 최대화하는 threshold를 고른다(단일 valid split보다 저분산).
- 전체 train으로 재학습 후 test 2,000개에서 최종 지표를 계산한다.
- 보정(calibrated) 모델은 비용 기반 이론 threshold(=2/3)도 함께 평가한다.

비용 구조: 발사·명중 +1, 발사·빗나감 -2, 미발사 0.
기대보상 = 3·P(hit) - 2 > 0  ⇔  P(hit) > 2/3.  → 잘 보정된 확률이면 최적 threshold ≈ 0.667.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_prepared_trajectories, radius_to_label  # noqa: E402
from src.features import build_feature_frame  # noqa: E402
from src.features_advanced import build_feature_frame_advanced  # noqa: E402
from src.metrics import (  # noqa: E402
    calculate_hit_score,
    evaluate_predictions,
    find_best_threshold,
    make_threshold_candidates,
    probabilities_to_decisions,
)

COST_THRESHOLD = 2.0 / 3.0  # 비용 기반 이론 최적 threshold
CACHE = PROJECT_ROOT / ".feature_cache"


# --------------------------------------------------------------------------- #
# 데이터 / 특징
# --------------------------------------------------------------------------- #
def load_features(dataset_dir: Path, split: str, advanced: bool):
    key = f"{split}_{'adv' if advanced else 'base'}.pkl"
    cache_path = CACHE / key
    if cache_path.exists():
        return joblib.load(cache_path)
    trajectories, labels, _ = load_prepared_trajectories(
        dataset_dir / split, dataset_dir / "metadata.json", split=split, show_progress=False
    )
    if advanced:
        frame = build_feature_frame_advanced(trajectories, include_baseline=True)
    else:
        frame = build_feature_frame(trajectories)
    merged = frame.merge(labels, on="id", validate="one_to_one")
    CACHE.mkdir(exist_ok=True)
    joblib.dump((frame, merged), cache_path)
    return frame, merged


def feature_columns(frame):
    return [c for c in frame.columns if c != "id"]


# --------------------------------------------------------------------------- #
# 모델 팩토리
# --------------------------------------------------------------------------- #
def make_lgbm(seed, **overrides):
    params = dict(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    params.update(overrides)
    return LGBMClassifier(**params)


def make_logreg(seed):
    return Pipeline(
        [("scaler", StandardScaler()),
         ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed))]
    )


def make_histgbm(seed):
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=False, random_state=seed
    )


def make_mlp(seed):
    return Pipeline(
        [("scaler", StandardScaler()),
         ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                               alpha=1e-3, batch_size=256, learning_rate_init=1e-3,
                               max_iter=300, early_stopping=True, n_iter_no_change=15,
                               random_state=seed))]
    )


def make_calibrated_lgbm(seed):
    # isotonic 보정 → 확률 신뢰도를 높여 비용 기반 threshold(2/3)가 의미를 갖게 함
    return CalibratedClassifierCV(make_lgbm(seed), method="isotonic", cv=5)


class SoftVotingEnsemble:
    """확률 평균 소프트 보팅(직접 구현해 동일 fit/predict_proba 인터페이스 제공)."""

    def __init__(self, builders, seed):
        self.models = [b(seed) for b in builders]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        probs = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1 - probs, probs])


def make_ensemble(seed):
    return SoftVotingEnsemble([make_lgbm, make_histgbm, make_mlp], seed)


# (이름, 빌더, advanced특징사용여부, 비용threshold도평가)
METHODS = [
    ("LightGBM (baseline)",            make_lgbm,            False, False),
    ("LogReg+Scaler (baseline)",       make_logreg,          False, False),
    ("LogReg+Scaler + adv feat",       make_logreg,          True,  False),
    ("HistGradientBoosting + adv feat", make_histgbm,        True,  False),
    ("MLP (neural net) + adv feat",    make_mlp,             True,  False),
    ("LightGBM + adv feat",            make_lgbm,            True,  False),
    ("LightGBM + adv feat (calibrated)", make_calibrated_lgbm, True, True),
    ("Ensemble(LGBM+HGB+MLP) + adv",   make_ensemble,        True,  False),
]


# --------------------------------------------------------------------------- #
# OOF 확률
# --------------------------------------------------------------------------- #
def oof_probabilities(builder, X, y, seed, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for tr, va in skf.split(X, y):
        model = builder(seed)
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
    return oof


def positive_proba(model, X):
    return model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------- #
# 실행
# --------------------------------------------------------------------------- #
def run_radius(dataset_dir, radius, seed):
    label = radius_to_label(radius)
    candidates = make_threshold_candidates(0.0, 1.0, 0.005)

    cache = {}
    for adv in (False, True):
        tr_frame, tr_merged = load_features(dataset_dir, "train", adv)
        te_frame, te_merged = load_features(dataset_dir, "test", adv)
        cols = feature_columns(tr_frame)
        cache[adv] = dict(
            cols=cols,
            X_train=tr_merged.loc[:, cols].to_numpy(dtype=float),
            y_train=tr_merged[label].to_numpy(dtype=int),
            X_test=te_merged.loc[:, cols].to_numpy(dtype=float),
            y_test=te_merged[label].to_numpy(dtype=int),
        )

    y_test_ref = cache[False]["y_test"]
    pos_rate = float(cache[False]["y_train"].mean())
    print(f"\n=== radius={radius} ({label}) | train pos-rate={pos_rate:.3f} "
          f"| n_feat base={len(cache[False]['cols'])} adv={len(cache[True]['cols'])} ===")

    rows = []
    for name, builder, adv, eval_cost in METHODS:
        d = cache[adv]
        t0 = time.time()
        oof = oof_probabilities(builder, d["X_train"], d["y_train"], seed)
        thr_result = find_best_threshold(d["y_train"], oof, candidates)
        thr = thr_result.threshold
        oof_mean = thr_result.mean_hit_score

        final = builder(seed)
        final.fit(d["X_train"], d["y_train"])
        test_proba = positive_proba(final, d["X_test"])
        metrics = evaluate_predictions(d["y_test"], test_proba, thr)
        elapsed = time.time() - t0

        row = dict(
            method=name, n_feat=len(d["cols"]), threshold=round(thr, 3),
            oof_mean_hit=round(oof_mean, 4),
            test_mean_hit=round(metrics["mean_hit_score"], 4),
            test_hit_score=metrics["hit_score"],
            precision=round(metrics["precision"], 4),
            recall=round(metrics["recall"], 4),
            auroc=round(metrics["auroc"], 4) if metrics["auroc"] is not None else None,
            shots=metrics["shots_fired"], seconds=round(elapsed, 1),
        )
        if eval_cost:
            cost_metrics = evaluate_predictions(d["y_test"], test_proba, COST_THRESHOLD)
            row["test_mean_hit_cost23"] = round(cost_metrics["mean_hit_score"], 4)
        rows.append(row)
        print(f"  {name:42s} thr={thr:0.3f} "
              f"OOF={oof_mean:0.4f} TEST={metrics['mean_hit_score']:0.4f} "
              f"P={metrics['precision']:0.3f} R={metrics['recall']:0.3f} "
              f"AUROC={row['auroc']} ({elapsed:0.0f}s)")

    # 참고용 상·하한
    fire_all = 3 * float(y_test_ref.mean()) - 2
    oracle = float(y_test_ref.mean())
    print(f"  {'[reference] fire-all':42s}             TEST={fire_all:0.4f}")
    print(f"  {'[reference] oracle (fire only hits)':42s}             TEST={oracle:0.4f}")

    best = max(rows, key=lambda r: r["test_mean_hit"])
    print(f"  >>> BEST: {best['method']}  test_mean_hit={best['test_mean_hit']}")
    return dict(radius=radius, label=label, pos_rate=pos_rate,
                fire_all=round(fire_all, 4), oracle=round(oracle, 4),
                results=rows, best=best["method"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--radii", type=float, nargs="+", default=[0.05])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments" / "results.json")
    args = parser.parse_args()

    summary = [run_radius(args.dataset_dir, r, args.seed) for r in args.radii]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
