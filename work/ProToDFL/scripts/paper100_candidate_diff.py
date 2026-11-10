"""从论文级第一趟的逐配对结果整理「候选差异」与「伙伴距离」两张表。

不需要模型快照：输入是 kl_benefit_distribution.py 产出的 r{1,250,500}.json 的 rows。
输出 outputs/paper100/candidate_diff/{per_target.csv, summary.json}
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SRC = Path(__file__).resolve().parents[1] / "outputs" / "paper100"
OUT = SRC / "candidate_diff"
ROUNDS = [1, 250, 500]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_target = []
    summary = {}

    for r in ROUNDS:
        rows = json.loads((SRC / f"r{r}.json").read_text(encoding="utf-8"))["rows"]
        by_target = {}
        for x in rows:
            by_target.setdefault(x["target"], []).append(x)

        round_rows = []
        for target, sub in sorted(by_target.items()):
            kls = np.array([x["kl_sym"] for x in sub], dtype=float)
            deltas = np.array([x["delta_acc"] for x in sub], dtype=float)
            ok = np.isfinite(kls) & np.isfinite(deltas)
            rho = spearmanr(kls[ok], deltas[ok]).statistic if ok.sum() > 2 else float("nan")
            top1 = min(sub, key=lambda x: x["kl_sym"])
            best = max(sub, key=lambda x: x["delta_acc"])
            rec = {
                "round": r,
                "target": target,
                "n_candidates": len(sub),
                "kl_top1": top1["candidate"],
                "kl_top1_delta_pp": top1["delta_acc"] * 100,
                "best": best["candidate"],
                "best_delta_pp": best["delta_acc"] * 100,
                "regret_pp": (best["delta_acc"] - top1["delta_acc"]) * 100,
                "range_pp": (deltas.max() - deltas.min()) * 100,
                "mean_delta_pp": deltas.mean() * 100,
                "target_rho": float(rho) if np.isfinite(rho) else None,
                "top1_hop": top1["hop"],
                "best_hop": best["hop"],
                "best_two_hop": int(best["hop"] <= 2),
                "kl_top1_is_best": int(best["candidate"] == top1["candidate"]),
            }
            round_rows.append(rec)
            per_target.append(rec)

        col = lambda k: np.array([x[k] for x in round_rows], dtype=float)
        hop_stats = []
        for hop in sorted(set(x["hop"] for x in rows)):
            sub = [x for x in rows if x["hop"] == hop]
            hop_stats.append({
                "hop": hop,
                "n_pairs": len(sub),
                "mean_delta_pp": float(np.mean([x["delta_acc"] for x in sub]) * 100),
            })
        summary[f"r{r}"] = {
            "n_targets": len(round_rows),
            "n_pairs": len(rows),
            "range_pp_mean": float(col("range_pp").mean()),
            "range_pp_median": float(np.median(col("range_pp"))),
            "regret_pp_mean": float(col("regret_pp").mean()),
            "regret_pp_median": float(np.median(col("regret_pp"))),
            "mean_delta_pp": float(col("mean_delta_pp").mean()),
            "target_rho_mean": float(np.nanmean([x["target_rho"] if x["target_rho"] is not None else np.nan for x in round_rows])),
            "kl_top1_is_best": f"{int(col('kl_top1_is_best').sum())}/{len(round_rows)}",
            "best_two_hop": f"{int(col('best_two_hop').sum())}/{len(round_rows)}",
            "hop_stats": hop_stats,
        }

    with open(OUT / "per_target.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_target[0].keys()))
        w.writeheader()
        w.writerows(per_target)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in ROUNDS:
        s = summary[f"r{r}"]
        print(f"r{r}: 目标内极差 {s['range_pp_mean']:.2f} pp, regret {s['regret_pp_mean']:.2f} pp, "
              f"KL首选命中 {s['kl_top1_is_best']}, 最优落两跳内 {s['best_two_hop']}")
        print("   跳数:", [(h["hop"], h["n_pairs"], round(h["mean_delta_pp"], 2)) for h in s["hop_stats"]])
    print("written:", OUT)


if __name__ == "__main__":
    main()
