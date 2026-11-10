"""RUN-020 figures: same-target candidate spread, KL regret, graph distance, KL vs benefit.

Reads the clean per-candidate CSV produced by ``same_target_candidate_analysis.py``
and writes PNGs for the delivery document. No model loading, no training.
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

NEG = "#c0392b"
POS = "#1f77b4"
GRAY = "#b8c2cc"
BLUE = "#2b7bba"


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw["degenerate_head"] == "True":
                continue
            rows.append(
                {
                    "target": int(raw["target"]),
                    "candidate": int(raw["candidate"]),
                    "hop": int(float(raw["hop"])),
                    "kl": float(raw["kl_sym"]),
                    "delta": float(raw["delta_acc"]) * 100.0,
                }
            )
    return rows


def fig_spread(rows: list[dict], out: Path) -> dict:
    by_target = collections.defaultdict(list)
    for row in rows:
        by_target[row["target"]].append(row)
    targets = sorted(by_target)

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for idx, target in enumerate(targets):
        cands = by_target[target]
        values = [c["delta"] for c in cands]
        ax.plot([min(values), max(values)], [idx, idx], color=GRAY, lw=7, solid_capstyle="butt", zorder=1)
        inside = [c["delta"] for c in cands if c["hop"] <= 2]
        if len(inside) >= 2:
            ax.plot([min(inside), max(inside)], [idx, idx], color=BLUE, lw=3, zorder=2)
        best = max(cands, key=lambda c: c["delta"])
        top1 = min(cands, key=lambda c: c["kl"])
        ax.plot([best["delta"]], [idx], "o", color=NEG, ms=5, zorder=3)
        ax.plot([top1["delta"]], [idx], "s", color="black", ms=4, zorder=4)

    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([str(t) for t in targets])
    ax.set_ylabel("目标客户端")
    ax.set_xlabel("聚合后相对未聚合的准确率变化（百分点）")
    ax.grid(axis="x", alpha=0.3)
    handles = [
        plt.Line2D([0], [0], color=GRAY, lw=7, label="全部候选范围"),
        plt.Line2D([0], [0], color=BLUE, lw=3, label="图距离 ≤2 范围"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=NEG, ms=6, label="事后最优"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="black", ms=5, label="KL 所选"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    ranges = [max(c["delta"] for c in v) - min(c["delta"] for c in v) for v in by_target.values()]
    return {"n_targets": len(targets), "mean_range_pp": st.mean(ranges), "max_range_pp": max(ranges)}


def fig_regret(rows: list[dict], out: Path) -> dict:
    by_target = collections.defaultdict(list)
    for row in rows:
        by_target[row["target"]].append(row)

    items = []
    for target, cands in by_target.items():
        top1 = min(cands, key=lambda c: c["kl"])
        best = max(c["delta"] for c in cands)
        inside = [c for c in cands if c["hop"] <= 2]
        best_inside = max((c["delta"] for c in inside), default=float("nan"))
        top1_inside = min(inside, key=lambda c: c["kl"]) if inside else None
        items.append(
            {
                "target": target,
                "all": best - top1["delta"],
                "two_hop": (best_inside - top1_inside["delta"]) if top1_inside else float("nan"),
            }
        )
    items.sort(key=lambda x: -x["all"])

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    idx = range(len(items))
    ax.bar([i - 0.19 for i in idx], [x["all"] for x in items], width=0.36, color=NEG, label="全部候选")
    two = [x["two_hop"] for x in items]
    ax.bar([i + 0.19 for i in idx], [0 if v != v else v for v in two], width=0.36, color=BLUE, label="图距离 ≤2")
    ax.set_xticks(list(idx))
    ax.set_xticklabels([str(x["target"]) for x in items])
    ax.set_xlabel("目标客户端")
    ax.set_ylabel("KL 所选与事后最优的差距（百分点）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "mean_regret_all_pp": st.mean([x["all"] for x in items]),
        "mean_regret_two_hop_pp": st.mean([x["two_hop"] for x in items if x["two_hop"] == x["two_hop"]]),
    }


def fig_distance(rows: list[dict], out: Path) -> dict:
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[row["hop"]].append(row["delta"])
    hops = sorted(buckets)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    means = [st.mean(buckets[h]) for h in hops]
    ax.plot(hops, means, "o-", color=BLUE, lw=1.8, ms=6)
    for hop, mean in zip(hops, means):
        ax.annotate(f"n={len(buckets[hop])}", (hop, mean), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(hops)
    ax.set_xlabel("候选到目标的图距离（快照邻接图最短路径）")
    ax.set_ylabel("平均 Δacc（百分点）")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return {str(h): st.mean(buckets[h]) for h in hops}


def fig_scatter(rows: list[dict], out: Path) -> None:
    targets = sorted({row["target"] for row in rows})
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for idx, target in enumerate(targets):
        xs = [r["kl"] for r in rows if r["target"] == target]
        ys = [r["delta"] for r in rows if r["target"] == target]
        ax.scatter(xs, ys, s=18, color=cmap(idx % 20), label=str(target), alpha=0.85, linewidths=0)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("对称 KL")
    ax.set_ylabel("Δacc（百分点）")
    ax.grid(alpha=0.3)
    ax.legend(title="目标", fontsize=7, ncol=2, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="outputs/experiments/2026-11-05_same-target-candidates/checks/same_target/e5-round1-base-s0/per_candidate.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/experiments/2026-11-05_same-target-candidates/figures",
    )
    args = parser.parse_args()

    path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load(path)

    stats = {"n_clean": len(rows)}
    stats.update(fig_spread(rows, out_dir / "fig1_spread.png"))
    stats.update(fig_regret(rows, out_dir / "fig2_regret.png"))
    stats.update({"distance": fig_distance(rows, out_dir / "fig3_distance.png")})
    fig_scatter(rows, out_dir / "fig4_scatter.png")

    print(f"rows {len(rows)}")
    print(f"mean range {stats['mean_range_pp']:.2f} pp, max {stats['max_range_pp']:.2f} pp")
    print(f"regret all {stats['mean_regret_all_pp']:.2f} pp, two-hop {stats['mean_regret_two_hop_pp']:.2f} pp")
    print({k: round(v, 2) for k, v in stats["distance"].items()})
    print("written", out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
