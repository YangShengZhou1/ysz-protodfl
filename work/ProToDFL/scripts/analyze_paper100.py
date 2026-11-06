"""汇总 paper100-a03-r500 三个快照的 KL 排序能力。

读取 kl_benefit_distribution.py 产出的 r{1,250,500}.json，基于全部 rows 重算指标
（脚本自带的 clean 过滤会把 my_part 下的退化头全部剔掉，这里不过滤，另报退化比例）。
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SRC = Path(__file__).resolve().parents[1] / "outputs" / "paper100"
ROUNDS = [1, 250, 500]


def pct(x):
    return f"{x * 100:.2f}%"


def pp(x):
    return f"{x * 100:+.2f} pp"


def analyze(rows):
    kls = np.array([r["kl_sym"] for r in rows], dtype=float)
    deltas = np.array([r["delta_acc"] for r in rows], dtype=float)
    deg = np.array([r["degenerate_head"] for r in rows], dtype=int)
    finite = np.isfinite(kls) & np.isfinite(deltas)
    rho = spearmanr(kls[finite], deltas[finite]).statistic if finite.sum() > 2 else float("nan")

    by_target = {}
    for r in rows:
        by_target.setdefault(r["target"], []).append(r)

    target_rhos, regrets, hits = [], [], []
    for target, subset in by_target.items():
        if len(subset) < 3:
            continue
        k = np.array([r["kl_sym"] for r in subset], dtype=float)
        d = np.array([r["delta_acc"] for r in subset], dtype=float)
        ok = np.isfinite(k) & np.isfinite(d)
        if ok.sum() > 2:
            r_ = spearmanr(k[ok], d[ok]).statistic
            if np.isfinite(r_):
                target_rhos.append(r_)
        best = max(subset, key=lambda r: r["delta_acc"])
        top1 = min(subset, key=lambda r: r["kl_sym"])
        regrets.append(best["delta_acc"] - top1["delta_acc"])
        hits.append(int(best["candidate"] == top1["candidate"]))

    return {
        "n_pairs": len(rows),
        "degenerate_frac": deg.mean(),
        "zero_delta_frac": float(np.mean(np.abs(deltas) < 1e-12)),
        "negative_frac": float(np.mean(deltas < 0)),
        "mean_delta": float(deltas.mean()),
        "delta_std": float(deltas.std(ddof=0)),
        "delta_range": float(deltas.max() - deltas.min()),
        "overall_rho": float(rho),
        "mean_target_rho": float(np.mean(target_rhos)) if target_rhos else float("nan"),
        "n_targets": len(by_target),
        "top1_hit": f"{sum(hits)}/{len(hits)}",
        "mean_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "unique_pred_counts": dict(zip(*np.unique(
            [r["pre_unique_predictions"] for r in rows], return_counts=True
        ))),
    }


def main():
    results = {}
    for r in ROUNDS:
        path = SRC / f"r{r}.json"
        if not path.exists():
            print(f"[skip] {path} 不存在")
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        results[r] = analyze(rows)

    if not results:
        return

    cols = [
        ("n_pairs", "配对数", lambda v: str(v)),
        ("n_targets", "目标数", lambda v: str(v)),
        ("degenerate_frac", "退化头比例", pct),
        ("zero_delta_frac", "零收益比例", pct),
        ("negative_frac", "负迁移比例", pct),
        ("mean_delta", "平均收益", pp),
        ("delta_std", "收益标准差", pp),
        ("delta_range", "收益极差", pp),
        ("overall_rho", "混算 Spearman", lambda v: f"{v:+.3f}"),
        ("mean_target_rho", "目标内 Spearman", lambda v: f"{v:+.3f}"),
        ("top1_hit", "KL 首选命中", str),
        ("mean_regret", "首选 regret", pp),
    ]
    header = ["指标"] + [f"r{r}" for r in results]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for key, label, fmt in cols:
        print(" | ".join([label] + [fmt(results[r][key]) for r in results]))

    print("\n各快照聚合前预测类别数分布：")
    for r, res in results.items():
        print(f"  r{r}: {res['unique_pred_counts']}")


if __name__ == "__main__":
    main()
