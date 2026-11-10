"""把全候选重放与分支起点核对结果并入 2026-11-07 的总汇总。"""
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "outputs/experiments/2026-11-07_next"
d = json.loads((P / "summary.json").read_text(encoding="utf-8"))

d["allcand"] = {}
for tag in ("dir", "path"):
    for r in (250, 500):
        f = P / f"allcand_paper100-next-{tag}_r{r}.json"
        d["allcand"][f"{tag}_r{r}"] = json.loads(f.read_text(encoding="utf-8"))
d["bn_check"] = json.loads((P / "bn_check.json").read_text(encoding="utf-8"))

(P / "summary.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("merged:", len(d["allcand"]), "all-candidate summaries")
