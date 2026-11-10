"""Experiment-1.5 figures: spread vs regret across the four control axes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "outputs/experiments/2026-11-05_supplement/run015_stats.json"
OUT = ROOT / "outputs/experiments/2026-11-05_supplement/figures"

SPREAD = "#2b7bba"
REGRET = "#c0392b"
GRAY = "#b8c2cc"

PANELS = [
    ("seed", "划分 seed", ["seed0", "seed1", "seed2"], ["s0", "s1", "s2"]),
    ("alpha", "Dirichlet α", ["alpha0.1", "seed0", "alpha0.5"], ["0.1", "0.3", "0.5"]),
    ("clients", "客户端数", ["clients10", "seed0", "clients50"], ["10", "20", "50"]),
    ("dataset", "数据集", ["seed0", "cifar100"], ["CIFAR-10", "CIFAR-100"]),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(STATS.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.4))
    for ax, (_, title_text, keys, labels) in zip(axes, PANELS):
        xs = range(len(keys))
        spread = [data[k]["mean_range"] for k in keys]
        regret = [data[k]["mean_regret"] for k in keys]
        width = 0.36
        ax.bar([x - width / 2 for x in xs], spread, width, color=SPREAD, label="目标内极差")
        ax.bar([x + width / 2 for x in xs], regret, width, color=REGRET, label="KL top-1 regret")
        for x, value in zip(xs, spread):
            ax.text(x - width / 2, value + 0.15, f"{value:.1f}", ha="center", fontsize=7.5)
        for x, value in zip(xs, regret):
            ax.text(x + width / 2, value + 0.15, f"{value:.1f}", ha="center", fontsize=7.5)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title_text, fontsize=10)
        ax.set_ylim(0, 15)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", color=GRAY, alpha=0.45, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("百分点 (pp)", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    first = OUT / "fig1_axes_overview.png"
    fig.savefig(first, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    groups = [
        ("CIFAR-10\nseed 0", data["seed0"]["per_target_range"], SPREAD),
        ("CIFAR-100", data["cifar100"]["per_target_range"], "#6b9e3f"),
    ]
    for idx, (label, values, color) in enumerate(groups):
        xs = [idx + (i - len(values) / 2) * 0.022 for i in range(len(values))]
        ax.scatter(xs, values, s=26, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
        mean = sum(values) / len(values)
        ax.hlines(mean, idx - 0.28, idx + 0.28, color=color, linewidth=2)
        ax.text(idx + 0.30, mean, f"均值 {mean:.2f}", va="center", fontsize=8, color=color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("同一目标内的 Δacc 极差 (pp)", fontsize=9)
    ax.grid(axis="y", color=GRAY, alpha=0.45, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    second = OUT / "fig2_spread_by_dataset.png"
    fig.savefig(second, dpi=200)
    plt.close(fig)

    print("written:", first)
    print("written:", second)


if __name__ == "__main__":
    main()
