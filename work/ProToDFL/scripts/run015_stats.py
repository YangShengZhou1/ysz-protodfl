"""Recompute the experiment-1.5 (supplement) numbers from per-candidate CSVs.

Reads the eight same-target outputs produced for RUN-020/022/023 and emits a
single json plus a markdown table. No plotting, no docx.
"""

from __future__ import annotations

import json
from pathlib import Path

from run020_stats import load, stats

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs/experiments/2026-11-05_same-target-candidates/checks/same_target"
NEW = ROOT / "outputs/experiments/2026-11-05_supplement/checks/same_target"

CONFIGS = [
    ("seed0", "划分 seed 0（主口径）", OLD / "e5-round1-base-s0"),
    ("seed1", "划分 seed 1", NEW / "e5-round1-base-s1"),
    ("seed2", "划分 seed 2", NEW / "e5-round1-base-s2"),
    ("alpha0.1", "α=0.1", NEW / "e5-round1-base-a01-s0"),
    ("alpha0.5", "α=0.5", NEW / "e5-round1-base-a05-s0"),
    ("clients10", "10 客户端", NEW / "e5-round1-base-srv10-s0"),
    ("clients50", "50 客户端", NEW / "e5-round1-base-srv50-s0"),
    ("cifar100", "CIFAR-100", NEW / "e5-round1-base-c100-s0"),
]


def summarize(path: Path) -> dict:
    rows = load(path / "per_candidate.csv")
    detail = stats(rows)
    return {
        "n_pairs": detail["n_pairs"],
        "n_targets": detail["n_targets"],
        "mean_delta": detail["mean_delta"],
        "negative_fraction": detail["negative_fraction"],
        "mean_range": detail["mean_range"],
        "mean_regret": detail["mean_regret"],
        "top1_is_best": detail["top1_is_best"],
        "mean_rho": detail["mean_rho"],
        "overall_spearman": detail["overall_spearman"],
        "two_hop": {
            "n_targets": detail["n_two"],
            "mean_range": detail["mean_range_two"],
            "mean_regret": detail["mean_regret_two"],
            "top1_is_best": detail["top1_is_best_two"],
        },
        "per_target_range": [t["range"] for t in detail["per_target"]],
        "per_target_regret": [t["regret"] for t in detail["per_target"]],
    }


def main() -> None:
    out = {}
    for key, label, path in CONFIGS:
        out[key] = {"label": label, "dir": str(path.relative_to(ROOT)), **summarize(path)}
        print(
            f"{key:10s} pairs={out[key]['n_pairs']:4d} targets={out[key]['n_targets']:3d} "
            f"range={out[key]['mean_range']:6.2f} regret={out[key]['mean_regret']:6.2f} "
            f"top1best={out[key]['top1_is_best']} rho={out[key]['mean_rho']:+.3f}"
        )

    target = ROOT / "outputs/experiments/2026-11-05_supplement/run015_stats.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written:", target)


if __name__ == "__main__":
    main()
