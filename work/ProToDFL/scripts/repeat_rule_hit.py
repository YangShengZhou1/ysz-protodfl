"""固定 10 个客户端上, 规则能否提前挑中事后最优伙伴。"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs"
SRC = ROOT / "experiments/2026-11-07_next"
OUT = ROOT / "experiments/2026-11-07_repeat"

rows = []
for tag in ("dir", "path"):
    per_round = {}
    for r in (250, 500):
        with (SRC / f"allcand_paper100-next-{tag}_r{r}.csv").open(encoding="utf-8", newline="") as f:
            per_round[r] = [x for x in csv.DictReader(f) if int(x["target"]) <= 4]

    by_target = {}
    for r in (250, 500):
        for x in per_round[r]:
            by_target.setdefault(int(x["target"]), {}).setdefault(r, []).append(x)

    for target, rounds in sorted(by_target.items()):
        r500 = rounds[500]
        r250 = {x["candidate"]: float(x["gain_after_pp"]) for x in rounds[250]}
        gain500 = {x["candidate"]: float(x["gain_after_pp"]) for x in r500}
        best_after = max(gain500, key=gain500.get)
        kl_top1 = min(r500, key=lambda x: float(x["kl_sym"]))["candidate"]
        select_best = min(r500, key=lambda x: float(x["select_loss_before"]))["candidate"]
        both_positive = [c for c in gain500 if gain500[c] > 1e-12 and r250.get(c, 0) > 1e-12]
        rows.append({
            "partition": tag,
            "target": target,
            "n_candidates": len(r500),
            "best_after": best_after,
            "best_gain_r500_pp": round(gain500[best_after], 2),
            "best_gain_r250_pp": round(r250.get(best_after, 0), 2),
            "kl_top1": kl_top1,
            "kl_hit": int(kl_top1 == best_after),
            "select_best": select_best,
            "select_hit": int(select_best == best_after),
            "candidates_both_rounds_positive": len(both_positive),
            "candidate_gains_r500_pp": "|".join(f"{c}:{gain500[c]:.1f}" for c in sorted(gain500, key=lambda z: int(z))),
        })

with (OUT / "rule_hit_small_table.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("目标数", len(rows))
for tag in ("dir", "path"):
    sub = [r for r in rows if r["partition"] == tag]
    print(tag, "| KL 命中", sum(r["kl_hit"] for r in sub), "/5",
          "| 选择集命中", sum(r["select_hit"] for r in sub), "/5",
          "| 有跨两轮正增益候选的目标", sum(1 for r in sub if r["candidates_both_rounds_positive"] > 0), "/5")
print("--- 逐目标小表 ---")
for r in rows:
    print(r["partition"], "t" + str(r["target"]),
          "| 事后最优", r["best_after"], r["best_gain_r500_pp"], "pp",
          "| KL", r["kl_top1"], "中" if r["kl_hit"] else "否",
          "| 选择集", r["select_best"], "中" if r["select_hit"] else "否",
          "| 两轮都正", r["candidates_both_rounds_positive"])
