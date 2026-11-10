"""统计重放后优于自己练的候选数量，以及跨 r250/r500 两轮次是否稳定。"""
import csv
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "outputs/experiments/2026-11-07_next"

for tag in ("dir", "path"):
    data = {}
    for r in (250, 500):
        f = P / f"allcand_paper100-next-{tag}_r{r}.csv"
        with f.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        data[r] = {(x["target"], x["candidate"]): float(x["gain_after_pp"]) for x in rows}

    for r in (250, 500):
        pos = sum(1 for v in data[r].values() if v > 1e-12)
        print(f"{tag} r{r}: 重放后优于自己练 {pos}/{len(data[r])} 对, "
              f"涉及 {len({t for (t, _), v in data[r].items() if v > 1e-12})} 个目标")

    common = set(data[250]) & set(data[500])
    both = {k for k in common if data[250][k] > 1e-12 and data[500][k] > 1e-12}
    print(f"{tag} 两轮都优于自己练: {len(both)}/{len(common)} 对, "
          f"涉及 {len({t for t, _ in both})} 个目标")
    if both:
        vals = sorted(data[500][k] for k in both)
        print(f"  r500 增益范围 {vals[0]:.2f} ~ {vals[-1]:.2f} pp, 中位 {vals[len(vals)//2]:.2f} pp")
