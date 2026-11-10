"""离线诊断：KL 与即时收益的关系形状，以及保守采纳的收益模拟。

只读 2026-11-07_next 下已有的逐配对 csv，不加载模型、不占显存。
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "work/ProToDFL/outputs/experiments/2026-11-07_next"
OUT = ROOT / "docs/2026-11-09"
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("dir_r250", EXP / "kl_paper100-next-dir_r250.csv"),
    ("dir_r500", EXP / "kl_paper100-next-dir_r500.csv"),
    ("path_r250", EXP / "kl_paper100-next-path_r250.csv"),
    ("path_r500", EXP / "kl_paper100-next-path_r500.csv"),
]


def load(path: Path):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append(
                    {
                        "target": int(r["target"]),
                        "candidate": int(r["candidate"]),
                        "hop": int(r["hop"]),
                        "kl": float(r["kl_sym"]),
                        "delta": float(r["delta_acc"]) * 100.0,
                        "degen": int(float(r.get("degenerate_head") or 0)),
                    }
                )
            except (TypeError, ValueError):
                continue
    return rows


def group_by_target(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["target"]].append(r)
    return {t: sorted(v, key=lambda x: x["kl"]) for t, v in g.items()}


def quantile_bins(rows, n=10):
    ordered = sorted(rows, key=lambda x: x["kl"])
    size = len(ordered) / n
    bins = []
    for i in range(n):
        chunk = ordered[int(i * size) : int((i + 1) * size)]
        if not chunk:
            continue
        kl_lo, kl_hi = chunk[0]["kl"], chunk[-1]["kl"]
        deltas = [c["delta"] for c in chunk]
        bins.append(
            {
                "idx": i + 1,
                "n": len(chunk),
                "kl_lo": kl_lo,
                "kl_hi": kl_hi,
                "mean": sum(deltas) / len(deltas),
                "pos_rate": sum(1 for d in deltas if d > 0) / len(deltas),
                "degen": sum(c["degen"] for c in chunk) / len(chunk),
            }
        )
    return bins


def simulate(rows):
    g = group_by_target(rows)
    hits = 0
    ties = 0
    multi_best = []
    rule_sum = 0.0
    oracle_sum = 0.0
    uniform_sum = 0.0
    neg_rate = 0.0
    margins = []
    total = 0
    for target, cands in g.items():
        if len(cands) < 2:
            continue
        total += 1
        mx = max(c["delta"] for c in cands)
        mn = min(c["kl"] for c in cands)
        best_set = {c["candidate"] for c in cands if c["delta"] == mx}
        top_set = {c["candidate"] for c in cands if c["kl"] == mn}
        multi_best.append(len(best_set))
        ties += int(len(best_set) > 1)
        hits += int(bool(best_set & top_set))
        top1 = cands[0]
        rule_sum += top1["delta"]
        oracle_sum += mx
        uniform_sum += sum(c["delta"] for c in cands) / len(cands)
        neg_rate += sum(1 for c in cands if c["delta"] < 0) / len(cands)
        margin = cands[1]["kl"] - cands[0]["kl"] if len(cands) > 1 else 0.0
        margins.append((margin, top1["delta"], mx))
    if not total:
        return None
    margins.sort(key=lambda x: -x[0])
    coverage = {}
    for pct in (10, 20, 30, 50, 100):
        k = max(1, int(total * pct / 100))
        chosen = margins[:k]
        gain = sum(m[1] for m in chosen) / total
        oracle_gain = sum(m[2] for m in chosen) / total
        coverage[pct] = {
            "k": k,
            "gain": gain,
            "gain_if_oracle": oracle_gain,
            "mean_on_chosen": sum(m[1] for m in chosen) / k,
        }
    return {
        "targets": total,
        "hit": hits,
        "hit_rate": hits / total,
        "tie_targets": ties,
        "mean_best_set": sum(multi_best) / len(multi_best) if multi_best else 0.0,
        "max_best_set": max(multi_best) if multi_best else 0,
        "rule_mean": rule_sum / total,
        "oracle_mean": oracle_sum / total,
        "uniform_mean": uniform_sum / total,
        "neg_pair_rate": neg_rate / total,
        "coverage": coverage,
    }


def main():
    summary = {}
    lines = ["# KL 信号形状与保守采纳模拟", "", "数据：2026-11-07_next 的逐配对即时评估，每目标 20 候选，100 目标。", ""]
    for name, path in CONFIGS:
        if not path.exists():
            lines.append(f"- {name}: 文件缺失 {path}")
            continue
        rows = load(path)
        if not rows:
            lines.append(f"- {name}: 解析为空")
            continue
        bins = quantile_bins(rows)
        sim = simulate(rows)
        summary[name] = {"pairs": len(rows), "bins": bins, "sim": sim}

        lines.append(f"## {name}（{len(rows)} 对）")
        lines.append("")
        lines.append("| 分位 | KL 区间 | 均值 Δacc | 正收益比例 | 退化头比例 |")
        lines.append("|---|---|---|---|---|")
        for b in bins:
            lines.append(
                f"| {b['idx']} | {b['kl_lo']:.2f} ~ {b['kl_hi']:.2f} | {b['mean']:+.2f} pp | {b['pos_rate']*100:.0f}% | {b['degen']*100:.0f}% |"
            )
        lines.append("")
        if sim:
            lines.append(
                f"规则选择（KL 最小）命中 {sim['hit']}/{sim['targets']}（与最优集合求交），"
                f"平均收益 {sim['rule_mean']:+.2f} pp；oracle 平均 {sim['oracle_mean']:+.2f} pp；"
                f"全候选平均 {sim['uniform_mean']:+.2f} pp；负迁移配对占 {sim['neg_pair_rate']*100:.0f}%；"
                f"目标内最优并列 {sim['tie_targets']}/{sim['targets']}，平均并列集合 {sim['mean_best_set']:.1f}，最大 {sim['max_best_set']}。"
            )
            lines.append("")
            lines.append("| 覆盖率 | 采纳目标数 | 全量平均收益 | 被采纳目标上的平均收益 | 该子集 oracle |")
            lines.append("|---|---|---|---|---|")
            for pct, c in sim["coverage"].items():
                lines.append(
                    f"| {pct}% | {c['k']} | {c['gain']:+.2f} pp | {c['mean_on_chosen']:+.2f} pp | {c['gain_if_oracle']:+.2f} pp |"
                )
            lines.append("")

    (OUT / "kl_signal_bins.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "kl_signal_bins.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
