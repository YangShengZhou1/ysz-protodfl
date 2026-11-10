"""把多份 analyze_curve.py 的输出合并成一份，供 plot_curve.py 与报告脚本使用。

用法：
  python scripts/merge_curve_json.py <输出.json> <输入1.json> [输入2.json ...]

同名键以后出现的为准；seed0 只有 1/250/500 三个点，来自 2026-11-07_next，
seed1/2 有完整 11 个点，来自 2026-11-10_curve。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("用法：merge_curve_json.py <输出.json> <输入1.json> [输入2.json ...]")

    out = Path(sys.argv[1])
    merged: dict[str, dict] = {}
    for name in sys.argv[2:]:
        path = Path(name)
        table = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(table, dict):
            raise SystemExit(f"{path} 不是 analyze_curve.py 的 JSON 字典")
        overlap = sorted(set(merged) & set(table))
        if overlap:
            print(f"[warn] {path.name} 覆盖 {len(overlap)} 个同名键：{overlap[:5]}")
        merged.update(table)
        print(f"[read] {path.name} {len(table)} 项")

    keys = sorted(
        merged,
        key=lambda k: (k.split("-s")[0], int(k.split("-s")[1].split("-r")[0]), int(k.split("-r")[1])),
    )
    out.write_text(
        json.dumps({k: merged[k] for k in keys}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    grids = sorted({k.split("-s")[0] for k in keys})
    seeds = sorted({int(k.split("-s")[1].split("-r")[0]) for k in keys})
    print(f"written: {out} | {len(keys)} 项 | 格 {grids} | seed {seeds}")


if __name__ == "__main__":
    main()
