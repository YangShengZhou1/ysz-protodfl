"""论文级第一趟实验（paper100-a03-r500）的完整统计。

输入：outputs/paper100/r{1,250,500}.json（kl_benefit_distribution.py 的 rows）
输出：outputs/paper100/detail.json + 控制台表格
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SRC = Path(__file__).resolve().parents[1] / "outputs" / "paper100"
ROUNDS = [1, 250, 500]
KL_MAX = 1e6  # kl_between 把 NaN/inf 夹到 1e12，这些不是有效 KL


def pp(x):
    return x * 100


def load(round_idx):
    return json.loads((SRC / f"r{round_idx}.json").read_text(encoding="utf-8"))["rows"]


def arr(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def rho(k, d):
    ok = np.isfinite(k) & np.isfinite(d)
    if ok.sum() < 3:
        return float("nan")
    v = spearmanr(k[ok], d[ok]).statistic
    return float(v) if np.isfinite(v) else float("nan")


def analyze(rows):
    kls = arr(rows, "kl_sym")
    deltas = arr(rows, "delta_acc")
    pre_acc = arr(rows, "pre_acc")
    valid = np.isfinite(kls) & (np.abs(kls) < KL_MAX) & np.isfinite(deltas)
    vrows = [r for r, ok in zip(rows, valid) if ok]
    vk, vd = kls[valid], deltas[valid]

    by_target = {}
    for r in vrows:
        by_target.setdefault(r["target"], []).append(r)
    target_rhos, regrets, hits = [], [], []
    for subset in by_target.values():
        if len(subset) < 3:
            continue
        k = arr(subset, "kl_sym")
        d = arr(subset, "delta_acc")
        r_ = rho(k, d)
        if np.isfinite(r_):
            target_rhos.append(r_)
        best = max(subset, key=lambda x: x["delta_acc"])
        top1 = min(subset, key=lambda x: x["kl_sym"])
        regrets.append(best["delta_acc"] - top1["delta_acc"])
        hits.append(int(best["candidate"] == top1["candidate"]))

    # KL 五分箱（按有效配对）
    order = np.argsort(vk)
    bins = []
    for chunk in np.array_split(order, 5):
        if len(chunk) == 0:
            continue
        bins.append({
            "kl_min": float(vk[chunk].min()),
            "kl_max": float(vk[chunk].max()),
            "n": int(len(chunk)),
            "mean_delta_pp": float(pp(vd[chunk].mean())),
            "zero_frac": float(np.mean(np.abs(vd[chunk]) < 1e-12)),
        })

    # my_part：100 客户端 / 10 组，每组 10 个，组内同质
    same = [r for r in vrows if r["target"] // 10 == r["candidate"] // 10]
    cross = [r for r in vrows if r["target"] // 10 != r["candidate"] // 10]

    def group_stat(subset):
        if not subset:
            return {"n": 0}
        k, d = arr(subset, "kl_sym"), arr(subset, "delta_acc")
        return {
            "n": len(subset),
            "mean_kl": float(k.mean()),
            "mean_delta_pp": float(pp(d.mean())),
            "zero_frac": float(np.mean(np.abs(d) < 1e-12)),
            "negative_frac": float(np.mean(d < 0)),
        }

    hops = Counter(r["hop"] for r in rows)
    hop_stats = []
    for hop in sorted(hops):
        subset = [r for r in vrows if r["hop"] == hop]
        if subset:
            hop_stats.append({
                "hop": hop,
                "n": len(subset),
                "mean_delta_pp": float(pp(arr(subset, "delta_acc").mean())),
            })

    return {
        "n_pairs": len(rows),
        "n_valid_kl": int(valid.sum()),
        "n_targets": len(by_target),
        "degenerate_frac": float(arr(rows, "degenerate_head").mean()),
        "pre_acc_mean": float(pre_acc.mean()),
        "pre_acc_min": float(pre_acc.min()),
        "pre_acc_max": float(pre_acc.max()),
        "unique_pred_dist": {str(k): int(v) for k, v in sorted(
            Counter(int(x) for x in arr(rows, "pre_unique_predictions")).items()
        )},
        "delta_mean_pp": float(pp(deltas.mean())),
        "delta_median_pp": float(pp(np.median(deltas))),
        "delta_std_pp": float(pp(deltas.std(ddof=0))),
        "delta_min_pp": float(pp(deltas.min())),
        "delta_max_pp": float(pp(deltas.max())),
        "zero_frac": float(np.mean(np.abs(deltas) < 1e-12)),
        "positive_frac": float(np.mean(deltas > 1e-12)),
        "negative_frac": float(np.mean(deltas < -1e-12)),
        "overall_rho": rho(vk, vd),
        "mean_target_rho": float(np.mean(target_rhos)) if target_rhos else None,
        "median_target_rho": float(np.median(target_rhos)) if target_rhos else None,
        "n_targets_rho_positive": int(sum(1 for x in target_rhos if x > 0)),
        "n_targets_rho_negative": int(sum(1 for x in target_rhos if x < 0)),
        "n_targets_evaluated": len(target_rhos),
        "top1_hit": int(sum(hits)),
        "top1_regret_mean_pp": float(pp(np.mean(regrets))) if regrets else None,
        "top1_regret_median_pp": float(pp(np.median(regrets))) if regrets else None,
        "kl_bins": bins,
        "same_group": group_stat(same),
        "cross_group": group_stat(cross),
        "hop_stats": hop_stats,
    }


def main():
    result = {f"r{r}": analyze(load(r)) for r in ROUNDS}
    (SRC / "detail.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for r in ROUNDS:
        d = result[f"r{r}"]
        print(f"\n===== round_{r} =====")
        print(f"配对 {d['n_pairs']}（有效 KL {d['n_valid_kl']}），目标 {d['n_targets']}，"
              f"退化头 {d['degenerate_frac']:.2%}")
        print(f"聚合前精度 均值 {d['pre_acc_mean']:.3f}，范围 {d['pre_acc_min']:.3f}–{d['pre_acc_max']:.3f}")
        print(f"预测类别数分布 {d['unique_pred_dist']}")
        print(f"收益 均值 {d['delta_mean_pp']:+.2f} pp，中位 {d['delta_median_pp']:+.2f}，"
              f"标准差 {d['delta_std_pp']:.2f}，范围 {d['delta_min_pp']:+.2f} ~ {d['delta_max_pp']:+.2f}")
        print(f"零/正/负 {d['zero_frac']:.2%} / {d['positive_frac']:.2%} / {d['negative_frac']:.2%}")
        print(f"混算 rho {d['overall_rho']:+.3f}；目标内 rho 均值 {d['mean_target_rho']:+.3f} "
              f"（正 {d['n_targets_rho_positive']} / 负 {d['n_targets_rho_negative']}，共 {d['n_targets_evaluated']}）")
        print(f"KL 首选命中 {d['top1_hit']}/100，regret 均值 {d['top1_regret_mean_pp']:.2f} pp "
              f"中位 {d['top1_regret_median_pp']:.2f} pp")
        print("KL 五分箱：")
        for b in d["kl_bins"]:
            print(f"  [{b['kl_min']:.2f}, {b['kl_max']:.2f}] n={b['n']:4d} "
                  f"mean_delta={b['mean_delta_pp']:+.2f} pp zero={b['zero_frac']:.2%}")
        print(f"同组 {d['same_group']}")
        print(f"跨组 {d['cross_group']}")
        print("按跳数：", d["hop_stats"])


if __name__ == "__main__":
    main()
