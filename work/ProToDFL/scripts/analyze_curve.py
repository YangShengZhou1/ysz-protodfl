"""汇总 seed/轮次曲线的逐配对 KL 评估结果。

用法：
  python scripts/analyze_curve.py <csv 目录> [--json 输出]

识别两种文件名：
  kl_paper100-next-<grid>_r<round>.csv      视为 seed 0
  kl_cur-<grid>-s<seed>_r<round>.csv        视为对应 seed
两种口径都给出：唯一最优（并列算不中）与并列安全（落在最优集合内）。
"""

from __future__ import annotations

import csv
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=Warning)

NAME_RE = re.compile(
    r"kl_(?:paper100-next-(?P<g0>dir|path)_r(?P<r0>\d+)|cur-(?P<g1>dir|path)-s(?P<s1>\d)_r(?P<r1>\d+))\.csv$"
)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x, y).statistic)
    except ImportError:
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])


def summarize(path: Path) -> dict:
    groups: dict[int, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            groups[int(row["target"])].append(row)

    deltas: list[float] = []
    uniq = tie = 0
    zero_pairs = 0
    degenerate_pairs = 0
    within: list[float] = []
    kl_top_delta: list[float] = []
    regrets: list[float] = []
    hops = defaultdict(list)

    for rows in groups.values():
        kl = np.array([float(r["kl_sym"]) for r in rows])
        dl = np.array([float(r["delta_acc"]) for r in rows]) * 100.0
        hp = [int(r["hop"]) for r in rows]
        deltas.extend(dl.tolist())
        zero_pairs += int(np.isclose(dl, 0.0, atol=1e-9).sum())
        degenerate_pairs += sum(
            1 for r in rows if str(r.get("degenerate_head", "")).lower() in {"1", "true"}
        )
        for d, h in zip(dl, hp):
            hops[h].append(d)

        idx = int(np.argmin(kl))
        mx = dl.max()
        n_best = int(np.isclose(dl, mx, atol=1e-9).sum())
        if np.isclose(dl[idx], mx, atol=1e-9):
            tie += 1
            uniq += int(n_best == 1)
        kl_top_delta.append(dl[idx])
        regrets.append(mx - dl[idx])
        within.append(spearman(kl, dl))

    all_kl = []
    all_dl = []
    for rows in groups.values():
        all_kl.extend(float(r["kl_sym"]) for r in rows)
        all_dl.extend(float(r["delta_acc"]) * 100.0 for r in rows)

    n_targets = len(groups)
    n_pairs = sum(len(v) for v in groups.values())
    return {
        "pairs": n_pairs,
        "targets": n_targets,
        "mean_pp": float(np.mean(deltas)),
        "median_pp": float(np.median(deltas)),
        "std_pp": float(np.std(deltas, ddof=1)),
        "zero_share": zero_pairs / n_pairs,
        "degenerate_share": degenerate_pairs / n_pairs,
        "hit_unique": uniq,
        "hit_tie": tie,
        "kl_top_mean_pp": float(np.mean(kl_top_delta)),
        "regret_mean_pp": float(np.mean(regrets)),
        "regret_median_pp": float(np.median(regrets)),
        "rho_mixed": spearman(np.array(all_kl), np.array(all_dl)),
        "rho_within_mean": float(np.nanmean(within)),
        "hop_mean_pp": {int(k): float(np.mean(v)) for k, v in sorted(hops.items())},
        "hop_pairs": {int(k): len(v) for k, v in sorted(hops.items())},
    }


def main() -> None:
    base = Path(sys.argv[1])
    out = None
    if "--json" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--json") + 1])

    table = {}
    for path in sorted(base.glob("*.csv")):
        m = NAME_RE.search(path.name)
        if not m:
            continue
        if m.group("g0"):
            grid, seed, rnd = m.group("g0"), 0, int(m.group("r0"))
        else:
            grid, seed, rnd = m.group("g1"), int(m.group("s1")), int(m.group("r1"))
        table[f"{grid}-s{seed}-r{rnd}"] = summarize(path)

    print(f"{'配置':<18}{'平均pp':>8}{'中位':>8}{'零收益':>8}{'退化':>7}"
          f"{'唯一最优':>9}{'并列安全':>9}{'KL首选':>8}{'regret':>8}{'内rho':>8}")
    for key in sorted(table, key=lambda k: (k.split("-s")[0], int(k.split("-s")[1].split("-")[0]),
                                             int(k.split("-r")[1]))):
        v = table[key]
        print(f"{key:<18}{v['mean_pp']:>8.2f}{v['median_pp']:>8.2f}"
              f"{v['zero_share'] * 100:>7.1f}%{v['degenerate_share'] * 100:>6.1f}%"
              f"{v['hit_unique']:>6}/100{v['hit_tie']:>6}/100"
              f"{v['kl_top_mean_pp']:>8.2f}{v['regret_mean_pp']:>8.2f}{v['rho_within_mean']:>8.3f}")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written: {out}")


if __name__ == "__main__":
    main()
