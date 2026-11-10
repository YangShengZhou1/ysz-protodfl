"""Recompute RUN-020 statistics from per-candidate CSVs (no plotting, no docx)."""

from __future__ import annotations

import collections
import csv
import statistics as st
from pathlib import Path

from scipy.stats import spearmanr


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw["degenerate_head"] == "True":
                continue
            rows.append(
                {
                    "target": int(raw["target"]),
                    "cand": int(raw["candidate"]),
                    "hop": int(float(raw["hop"])),
                    "kl": float(raw["kl_sym"]),
                    "da": float(raw["delta_acc"]) * 100.0,
                    "dl": float(raw["delta_loss"]),
                }
            )
    return rows


def stats(rows: list[dict]) -> dict:
    by_target: dict[int, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_target[row["target"]].append(row)

    ranges, regrets, rho = [], [], []
    top1_is_best = top1_beats_second = 0
    ranges2, regrets2, top1_is_best2, n_two = [], [], 0, 0
    per_target = []

    for target in sorted(by_target):
        cands = by_target[target]
        if len(cands) < 2:
            continue
        values = [c["da"] for c in cands]
        ordered = sorted(cands, key=lambda c: c["kl"])
        top1, second = ordered[0], (ordered[1] if len(ordered) > 1 else None)
        best = max(cands, key=lambda c: c["da"])

        ranges.append(max(values) - min(values))
        regrets.append(best["da"] - top1["da"])
        top1_is_best += int(abs(top1["da"] - best["da"]) < 1e-9)
        top1_beats_second += int(second is not None and top1["da"] > second["da"] + 1e-9)
        if len(cands) >= 3:
            value = spearmanr([c["kl"] for c in cands], values).statistic
            if value == value:
                rho.append(value)

        inside = [c for c in cands if c["hop"] <= 2]
        if len(inside) >= 2:
            n_two += 1
            inside_values = [c["da"] for c in inside]
            ranges2.append(max(inside_values) - min(inside_values))
            inside_top1 = min(inside, key=lambda c: c["kl"])
            regrets2.append(max(inside_values) - inside_top1["da"])
            top1_is_best2 += int(abs(inside_top1["da"] - max(inside_values)) < 1e-9)

        per_target.append(
            {
                "target": target,
                "n": len(cands),
                "range": max(values) - min(values),
                "top1": top1,
                "best": best,
                "regret": best["da"] - top1["da"],
                "best_inside_two": best["hop"] <= 2,
            }
        )

    overall = spearmanr([r["kl"] for r in rows], [r["da"] for r in rows]).statistic
    buckets: dict[int, list[float]] = collections.defaultdict(list)
    for row in rows:
        buckets[row["hop"]].append(row["da"])

    return {
        "n_pairs": len(rows),
        "n_targets": len(per_target),
        "mean_range": st.mean(ranges),
        "median_range": st.median(ranges),
        "max_range": max(ranges),
        "mean_regret": st.mean(regrets),
        "median_regret": st.median(regrets),
        "max_regret": max(regrets),
        "top1_is_best": top1_is_best,
        "top1_beats_second": top1_beats_second,
        "mean_rho": st.mean(rho) if rho else float("nan"),
        "n_rho": len(rho),
        "n_rho_negative": sum(1 for v in rho if v < 0),
        "overall_spearman": overall,
        "n_two": n_two,
        "mean_range_two": st.mean(ranges2) if ranges2 else float("nan"),
        "max_range_two": max(ranges2) if ranges2 else float("nan"),
        "mean_regret_two": st.mean(regrets2) if regrets2 else float("nan"),
        "top1_is_best_two": top1_is_best2,
        "per_target": per_target,
        "distance": {hop: (len(v), st.mean(v)) for hop, v in sorted(buckets.items())},
        "mean_delta": st.mean([r["da"] for r in rows]),
        "negative_fraction": sum(1 for r in rows if r["da"] < 0) / len(rows),
    }
