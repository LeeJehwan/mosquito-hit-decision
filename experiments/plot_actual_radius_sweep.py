"""Plot the actual radius retraining sweep results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT_PATH = ROOT / "actual_radius_sweep_results.json"
OUTPUT_PATH = ROOT / "actual_radius_sweep.png"


def load_rows() -> list[dict]:
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    return payload["results"]


def annotate_point(ax, x: float, y: float, label: str, color: str) -> None:
    ax.scatter([x], [y], s=72, color=color, zorder=5)
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=9,
        color=color,
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
    )


def main() -> None:
    rows = load_rows()
    radius = np.asarray([row["radius"] for row in rows], dtype=float)
    mean_hit = np.asarray([row["selected_test_mean_hit"] for row in rows], dtype=float)
    hit_rate = np.asarray([row["test_hit_rate"] for row in rows], dtype=float)
    fire_all = np.asarray([row["fire_all_mean_hit"] for row in rows], dtype=float)
    precision = np.asarray([row["selected_test_precision"] for row in rows], dtype=float)
    recall = np.asarray([row["selected_test_recall"] for row in rows], dtype=float)
    shots = np.asarray([row["selected_test_shots_fired"] for row in rows], dtype=float)
    delta = np.asarray(
        [np.nan if row["delta_mean_hit"] is None else row["delta_mean_hit"] for row in rows],
        dtype=float,
    )
    roi = np.asarray(
        [np.nan if row["roi_per_radius"] is None else row["roi_per_radius"] for row in rows],
        dtype=float,
    )

    knee = rows[int(np.argmax((mean_hit - mean_hit.min()) / np.ptp(mean_hit) - (radius - radius.min()) / np.ptp(radius)))]
    best = rows[int(np.argmax(mean_hit))]
    gain95_target = mean_hit[0] + 0.95 * (mean_hit.max() - mean_hit[0])
    gain95 = rows[int(np.argmax(mean_hit >= gain95_target))]

    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.0])
    ax_score = fig.add_subplot(grid[0, :])
    ax_delta = fig.add_subplot(grid[1, 0])
    ax_roi = fig.add_subplot(grid[1, 1])
    ax_pr = fig.add_subplot(grid[2, 0])
    ax_shots = fig.add_subplot(grid[2, 1])

    ax_score.plot(radius, mean_hit, marker="o", lw=2.2, color="#2563eb", label="Selected test mean_hit")
    ax_score.plot(radius, fire_all, marker="x", lw=1.4, ls="--", color="#b45309", label="Fire-all baseline")
    ax_score.plot(radius, hit_rate, marker=".", lw=1.4, color="#64748b", label="Oracle upper bound: hit rate")
    annotate_point(ax_score, knee["radius"], knee["selected_test_mean_hit"], "knee r=0.030", "#dc2626")
    annotate_point(ax_score, gain95["radius"], gain95["selected_test_mean_hit"], "95% gain r=0.075", "#16a34a")
    annotate_point(ax_score, best["radius"], best["selected_test_mean_hit"], "best r=0.100", "#7c3aed")
    ax_score.set_title("Actual Radius Sweep: retrain ensemble + OOF blend/threshold")
    ax_score.set_xlabel("radius")
    ax_score.set_ylabel("test mean_hit")
    ax_score.set_xticks(radius)
    ax_score.grid(alpha=0.25)
    ax_score.legend(loc="lower right")

    bar_width = 0.0036
    ax_delta.bar(radius[1:], delta[1:], width=bar_width, color="#0f766e")
    ax_delta.axhline(0.0, color="#334155", lw=0.8)
    ax_delta.set_title("Marginal mean_hit gain per +0.005 radius")
    ax_delta.set_xlabel("radius")
    ax_delta.set_ylabel("delta mean_hit")
    ax_delta.grid(axis="y", alpha=0.25)

    ax_roi.plot(radius[1:], roi[1:], marker="o", lw=1.8, color="#ea580c")
    ax_roi.axhline(0.0, color="#334155", lw=0.8)
    ax_roi.set_title("ROI per radius")
    ax_roi.set_xlabel("radius")
    ax_roi.set_ylabel("delta mean_hit / delta radius")
    ax_roi.grid(alpha=0.25)

    ax_pr.plot(radius, precision, marker="o", lw=1.8, color="#9333ea", label="precision")
    ax_pr.plot(radius, recall, marker="o", lw=1.8, color="#0284c7", label="recall")
    ax_pr.set_title("Precision / Recall")
    ax_pr.set_xlabel("radius")
    ax_pr.set_ylabel("score")
    ax_pr.set_ylim(0.55, 1.02)
    ax_pr.grid(alpha=0.25)
    ax_pr.legend(loc="lower right")

    ax_shots.plot(radius, shots, marker="o", lw=1.8, color="#475569")
    ax_shots.set_title("Shots fired")
    ax_shots.set_xlabel("radius")
    ax_shots.set_ylabel("count out of 2000")
    ax_shots.set_ylim(700, 2020)
    ax_shots.grid(alpha=0.25)

    for ax in (ax_score, ax_delta, ax_roi, ax_pr, ax_shots):
        ax.tick_params(axis="x", labelrotation=45)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
