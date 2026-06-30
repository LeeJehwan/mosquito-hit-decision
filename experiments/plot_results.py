"""통합 results.json을 읽어 5개 반경 전체 비교 차트(PNG)를 만든다."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 (Windows): Malgun Gothic
for _font in ("Malgun Gothic", "맑은 고딕", "NanumGothic", "AppleGothic"):
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
blocks = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
data = {b["radius"]: b for b in blocks}
radii = sorted(data, reverse=True)  # 0.05 .. 0.01
order = [r["method"] for r in data[radii[0]]["results"]]

short = {
    "LightGBM (baseline)": "LGBM\n(baseline)",
    "LogReg+Scaler (baseline)": "LogReg\n(baseline)",
    "LogReg+Scaler + adv feat": "LogReg\n+adv",
    "HistGradientBoosting + adv feat": "HistGBM\n+adv",
    "MLP (neural net) + adv feat": "MLP\n+adv",
    "LightGBM + adv feat": "LGBM\n+adv",
    "LightGBM + adv feat (calibrated)": "LGBM+adv\n(calib)",
    "Ensemble(LGBM+HGB+MLP) + adv": "Ensemble\n+adv",
}
colors = ["#9aa0a6", "#c0606e", "#7aa8d8", "#69a36b", "#caa45a", "#5b8fb0", "#d98445", "#7a5fb0"]


def get(r, m, k):
    return next(x for x in data[r]["results"] if x["method"] == m)[k]


# --- Figure 1: 반경별 점수 막대 (5 subplots) ---
fig, axes = plt.subplots(1, len(radii), figsize=(4.2 * len(radii), 5.4))
for ax, r in zip(axes, radii):
    block = data[r]
    vals = [get(r, m, "test_mean_hit") for m in order]
    bars = ax.bar(range(len(order)), vals, color=colors)
    ax.axhline(block["fire_all"], ls="--", c="#b00020", lw=1.1, label=f"fire-all {block['fire_all']:.3f}")
    ax.axhline(block["oracle"], ls=":", c="#1b7a1b", lw=1.1, label=f"oracle {block['oracle']:.3f}")
    best = max(vals)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.004 if v >= 0 else -0.012), f"{v:.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.5,
                fontweight="bold" if v == best else "normal")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short[m] for m in order], fontsize=7)
    lo = min(vals + [block["fire_all"]]) - 0.02
    hi = block["oracle"] + 0.03
    ax.set_ylim(lo, hi)
    ax.set_title(f"radius={r}  (pos {block['pos_rate']:.3f})", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, c="#444", lw=0.6)
axes[0].set_ylabel("test mean hit score  (명중 +1, 빗나감 -2, 미발사 0)")
fig.suptitle("Mosquito Hit-Decision: 방법론별 점수 비교 — 5개 반경 (test 2,000)", fontsize=13, y=1.02)
fig.tight_layout()
out = ROOT / "methodology_comparison.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)

# --- Figure 2: 평균순위 + AUROC 추세 ---
fig2, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.2))

# (좌) 평균순위 (낮을수록 좋음)
ranks = {m: [] for m in order}
for r in radii:
    ordered = sorted(order, key=lambda m: -get(r, m, "test_mean_hit"))
    for pos, m in enumerate(ordered, 1):
        ranks[m].append(pos)
avgrank = {m: float(np.mean(ranks[m])) for m in order}
ms_sorted = sorted(order, key=lambda m: avgrank[m])
ypos = np.arange(len(ms_sorted))[::-1]
axL.barh(ypos, [avgrank[m] for m in ms_sorted],
         color=[colors[order.index(m)] for m in ms_sorted])
for y, m in zip(ypos, ms_sorted):
    axL.text(avgrank[m] + 0.05, y, f"{avgrank[m]:.1f}", va="center", fontsize=9)
axL.set_yticks(ypos)
axL.set_yticklabels([short[m].replace("\n", " ") for m in ms_sorted], fontsize=9)
axL.set_xlabel("평균 순위 (5개 반경, 낮을수록 좋음)")
axL.set_title("점수 기준 평균 순위")
axL.grid(axis="x", alpha=0.3)

# (우) AUROC 추세 (반경별)
x = list(range(len(radii)))
for m in order:
    aur = [get(r, m, "auroc") for r in radii]
    axR.plot(x, aur, marker="o", lw=1.6, label=short[m].replace("\n", " "),
             color=colors[order.index(m)])
axR.set_xticks(x)
axR.set_xticklabels([f"r={r}" for r in radii])
axR.set_xlabel("반경 (오른쪽일수록 어려움)")
axR.set_ylabel("AUROC (확률 판별력)")
axR.set_title("방법론별 AUROC — 반경별 추세")
axR.legend(fontsize=7.5, ncol=2, loc="lower left")
axR.grid(alpha=0.3)
fig2.tight_layout()
out2 = ROOT / "ranking_auroc.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print("saved", out2)
