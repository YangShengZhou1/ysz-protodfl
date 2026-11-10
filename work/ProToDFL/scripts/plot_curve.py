"""按 analyze_curve.py 的 JSON 画 seed 曲线图。

用法：
  python scripts/plot_curve.py <analyze_curve.json> <输出 png 目录>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

STYLE = {
    "dir": {"color": "#1f77b4", "label": "Dirichlet"},
    "path": {"color": "#d62728", "label": "Pathological"},
}
SEED_MARK = {0: "o", 1: "s", 2: "^"}


def parse_key(key: str):
    grid, rest = key.split("-s")
    seed_s, rnd_s = rest.split("-r")
    return grid, int(seed_s), int(rnd_s)


def main():
    table = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)

    series = defaultdict(dict)
    for key, v in table.items():
        grid, seed, rnd = parse_key(key)
        series[(grid, seed)][rnd] = v

    # 图 1：平均即时收益随轮次
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for (grid, seed), rounds in sorted(series.items()):
        xs = sorted(rounds)
        ys = [rounds[r]["mean_pp"] for r in xs]
        ax.plot(xs, ys, marker=SEED_MARK.get(seed, "x"), ms=4, lw=1.4,
                color=STYLE[grid]["color"], alpha=0.45 + 0.25 * seed,
                label=f"{STYLE[grid]['label']} seed{seed}")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("round")
    ax.set_ylabel("mean Δacc (pp)")
    ax.set_title("KL 首选候选的平均即时收益")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "curve_mean_pp.png", dpi=200)
    plt.close(fig)

    # 图 2：并列安全命中率随轮次
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for (grid, seed), rounds in sorted(series.items()):
        xs = sorted(rounds)
        ys = [rounds[r]["hit_tie"] for r in xs]
        ax.plot(xs, ys, marker=SEED_MARK.get(seed, "x"), ms=4, lw=1.4,
                color=STYLE[grid]["color"], alpha=0.45 + 0.25 * seed,
                label=f"{STYLE[grid]['label']} seed{seed}")
    ax.set_xlabel("round")
    ax.set_ylabel("hits / 100")
    ax.set_title("KL 首选落在事后最优集合内的目标数")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "curve_hit_tie.png", dpi=200)
    plt.close(fig)

    print(f"wrote {outdir}/curve_mean_pp.png, {outdir}/curve_hit_tie.png")


if __name__ == "__main__":
    main()
