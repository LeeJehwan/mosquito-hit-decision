"""방법론별 '판정 지연(latency)' 측정.

판정 시간 = 특징 추출(per sample) + 모델 추론(predict_proba).
- 배치(test 2,000개 한 번에) 처리량과
- 단일 샘플(실시간 발사 1건) 지연을 모두 측정한다.
학습 시간은 제외(서버에서 1회). 측정은 중앙값으로 보고한다.
"""

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_prepared_trajectories, radius_to_label  # noqa: E402
from src.features import build_trajectory_features  # noqa: E402
from src.features_advanced import build_advanced_features  # noqa: E402
from experiments.run_methodology_comparison import (  # noqa: E402
    METHODS, load_features, feature_columns,
)

RADIUS = 0.05
SEED = 42


def median_time(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def bench_feature_extraction(dataset_dir):
    """샘플 1개당 특징 추출 시간(ms): baseline vs advanced(=baseline+추가)."""
    trajectories, _, _ = load_prepared_trajectories(
        dataset_dir / "test", dataset_dir / "metadata.json", split="test", show_progress=False
    )
    sample = list(trajectories.values())[:200]

    def base_once():
        for tr in sample:
            build_trajectory_features(tr)

    def adv_once():
        for tr in sample:
            build_trajectory_features(tr)   # advanced는 baseline 포함
            build_advanced_features(tr)

    base_ms = median_time(base_once, 5) / len(sample) * 1000
    adv_ms = median_time(adv_once, 5) / len(sample) * 1000
    return base_ms, adv_ms


def main():
    dataset_dir = PROJECT_ROOT / "dataset"
    label = radius_to_label(RADIUS)

    feat_base_ms, feat_adv_ms = bench_feature_extraction(dataset_dir)
    print(f"특징 추출 per sample:  baseline={feat_base_ms:.3f} ms,  advanced={feat_adv_ms:.3f} ms\n")

    cache = {}
    for adv in (False, True):
        tr_frame, tr_merged = load_features(dataset_dir, "train", adv)
        te_frame, te_merged = load_features(dataset_dir, "test", adv)
        cols = feature_columns(tr_frame)
        cache[adv] = dict(
            X_train=tr_merged.loc[:, cols].to_numpy(dtype=float),
            y_train=tr_merged[label].to_numpy(dtype=int),
            X_test=te_merged.loc[:, cols].to_numpy(dtype=float),
        )

    header = (f"{'방법론':40s} {'배치2000(ms)':>13s} {'처리량(/s)':>12s} "
              f"{'모델1건(ms)':>12s} {'특징(ms)':>9s} {'판정총(ms)':>11s}")
    print(header)
    print("-" * len(header))

    rows = []
    for name, builder, adv, _ in METHODS:
        d = cache[adv]
        model = builder(SEED)
        model.fit(d["X_train"], d["y_train"])
        X = d["X_test"]
        x1 = X[:1]

        model.predict_proba(x1)  # warm-up

        batch_s = median_time(lambda: model.predict_proba(X), 7)
        single_s = median_time(lambda: model.predict_proba(x1), 200)

        batch_ms = batch_s * 1000
        throughput = len(X) / batch_s
        single_ms = single_s * 1000
        feat_ms = feat_adv_ms if adv else feat_base_ms
        total_ms = feat_ms + single_ms

        rows.append((name, batch_ms, throughput, single_ms, feat_ms, total_ms))
        print(f"{name:40s} {batch_ms:13.1f} {throughput:12.0f} "
              f"{single_ms:12.3f} {feat_ms:9.3f} {total_ms:11.3f}")

    print("\n* 배치2000 = test 2,000건 한 번에 predict_proba (ms)")
    print("* 모델1건 = 단일 샘플 predict_proba 중앙값 (실시간 1발 판정)")
    print("* 판정총 = 특징추출(1건) + 모델추론(1건)  → 실시간 1발 결정 지연")


if __name__ == "__main__":
    main()
