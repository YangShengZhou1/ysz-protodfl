"""Same-target candidate comparison on an existing KL-benefit CSV.

Reads one `kl_benefit*.csv` produced by `kl_benefit_distribution.py`, groups the
single-partner pairs by target client and asks:

1. do candidates with smaller prototype KL still differ clearly in benefit;
2. is the KL top-1 candidate the best one for that target;
3. at which graph distance the useful partners sit, and whether the best
   candidate lies inside the original two-hop neighbourhood.

No training and no model loading: the script only re-reads recorded columns.

Two health filters are available. `degenerate_head == 1` is dropped always.
`--max-majority-fraction` additionally drops pairs whose predicted class is too
concentrated even when more than one class is emitted; the default 1.0 keeps the
main 187-pair sample intact so that it stays comparable with earlier numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import spearmanr


REQUIRED = ("target", "candidate", "kl_sym", "pre_acc", "agg_acc", "delta_acc", "delta_loss")


def load_rows(path: Path, max_majority: float) -> tuple[list[dict], int, int]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise SystemExit(f"missing columns: {missing}")

    clean, dropped_degenerate, dropped_majority = [], 0, 0
    for row in rows:
        flag = row.get("degenerate_head")
        if flag not in (None, "") and int(float(flag)) == 1:
            dropped_degenerate += 1
            continue
        if max_majority < 1.0:
            pre_maj = row.get("pre_majority_prediction_fraction") or 0
            agg_maj = row.get("agg_majority_prediction_fraction") or 0
            if max(float(pre_maj), float(agg_maj)) >= max_majority:
                dropped_majority += 1
                continue
        parsed = dict(row)
        for key in ("target", "candidate"):
            parsed[key] = int(float(parsed[key]))
        for key, value in row.items():
            if key not in ("target", "candidate") and value not in ("", None):
                try:
                    parsed[key] = float(value)
                except ValueError:
                    pass
        clean.append(parsed)
    return clean, dropped_degenerate, dropped_majority


def analyse(rows: list[dict], min_pairs: int = 2) -> tuple[list[dict], list[dict]]:
    by_target: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_target[row["target"]].append(row)

    per_target, per_candidate = [], []
    for target in sorted(by_target):
        cands = sorted(by_target[target], key=lambda r: r["kl_sym"])
        for rank_kl, row in enumerate(cands, 1):
            per_candidate.append({**row, "kl_rank_in_target": rank_kl, "n_candidates_in_target": len(cands)})
        top1 = cands[0]
        best = max(cands, key=lambda r: r["delta_acc"])
        deltas = [r["delta_acc"] for r in cands]
        entry = {
            "target": target,
            "n_candidates": len(cands),
            "comparable": int(len(cands) >= min_pairs),
            "kl_top1_candidate": top1["candidate"],
            "kl_top1_delta_acc": top1["delta_acc"],
            "kl_top1_graph_distance": int(top1["hop"]) if top1.get("hop") is not None else None,
            "kl_second_candidate": cands[1]["candidate"] if len(cands) > 1 else None,
            "kl_second_delta_acc": cands[1]["delta_acc"] if len(cands) > 1 else None,
            "top1_minus_second_delta_acc": (top1["delta_acc"] - cands[1]["delta_acc"]) if len(cands) > 1 else None,
            "best_candidate": best["candidate"],
            "best_delta_acc": best["delta_acc"],
            "best_graph_distance": int(best["hop"]) if best.get("hop") is not None else None,
            "top1_regret": best["delta_acc"] - top1["delta_acc"] if len(cands) >= min_pairs else None,
            "delta_range": max(deltas) - min(deltas) if len(cands) >= min_pairs else None,
            "within_target_spearman": None,
        }
        if len(cands) >= 3:
            rho = spearmanr([r["kl_sym"] for r in cands], deltas)
            if rho.statistic == rho.statistic:
                entry["within_target_spearman"] = float(rho.statistic)
        per_target.append(entry)
    return per_target, per_candidate


def graph_distance_analysis(rows: list[dict], per_target: list[dict]) -> dict | None:
    if not rows or rows[0].get("hop") in ("", None):
        return None
    buckets: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        buckets[int(row["hop"])].append(row["delta_acc"])
    means = {d: statistics.mean(v) for d, v in sorted(buckets.items())}
    counts = {d: len(v) for d, v in sorted(buckets.items())}
    best_dists = Counter(r["best_graph_distance"] for r in per_target if r["best_graph_distance"] is not None)
    top1_dists = Counter(r["kl_top1_graph_distance"] for r in per_target if r["kl_top1_graph_distance"] is not None)
    inside2_best = sum(v for d, v in best_dists.items() if d <= 2)
    return {
        "mean_delta_by_distance": {str(d): means[d] for d in means},
        "count_by_distance": {str(d): counts[d] for d in counts},
        "best_distance_histogram": {str(d): v for d, v in sorted(best_dists.items())},
        "kl_top1_distance_histogram": {str(d): v for d, v in sorted(top1_dists.items())},
        "n_targets_best_inside_two_hops": inside2_best,
        "n_targets": sum(1 for r in per_target if r["comparable"]),
    }


def summarise(per_target: list[dict], rows: list[dict], dropped_degenerate: int, dropped_majority: int) -> dict:
    comp = [r for r in per_target if r["comparable"]]
    regrets = [r["top1_regret"] for r in comp if r["top1_regret"] is not None]
    ranges = [r["delta_range"] for r in comp if r["delta_range"] is not None]
    rhos = [r["within_target_spearman"] for r in per_target if r["within_target_spearman"] is not None]
    deltas = [r["delta_acc"] for r in rows]
    overall = spearmanr([r["kl_sym"] for r in rows], deltas)
    return {
        "n_pairs_total": len(rows) + dropped_degenerate + dropped_majority,
        "n_pairs_clean": len(rows),
        "n_dropped_degenerate": dropped_degenerate,
        "n_dropped_majority": dropped_majority,
        "n_targets": len(per_target),
        "n_targets_comparable": len(comp),
        "mean_candidates_per_target": statistics.mean([r["n_candidates"] for r in per_target]),
        "mean_delta_acc_clean": statistics.mean(deltas),
        "negative_transfer_fraction_clean": sum(1 for d in deltas if d < 0) / len(deltas),
        "mean_top1_regret": statistics.mean(regrets) if regrets else None,
        "median_top1_regret": statistics.median(regrets) if regrets else None,
        "max_top1_regret": max(regrets) if regrets else None,
        "n_targets_top1_is_best": sum(1 for r in comp if r["kl_top1_candidate"] == r["best_candidate"]),
        "mean_delta_range_within_target": statistics.mean(ranges) if ranges else None,
        "median_delta_range_within_target": statistics.median(ranges) if ranges else None,
        "n_targets_top1_beats_second": sum(1 for r in comp if (r["top1_minus_second_delta_acc"] or 0) > 0),
        "mean_within_target_spearman": statistics.mean(rhos) if rhos else None,
        "median_within_target_spearman": statistics.median(rhos) if rhos else None,
        "n_within_target_spearman_negative": sum(1 for x in rhos if x < 0),
        "n_within_target_spearman": len(rhos),
        "overall_spearman_kl_vs_delta": float(overall.statistic),
        "graph_distance": graph_distance_analysis(rows, per_target),
        "two_hop_subset": two_hop_summary(rows),
    }


def two_hop_summary(rows: list[dict]) -> dict | None:
    """Restrict to candidates the original rule can see (graph distance <= 2)."""
    if not rows or rows[0].get("hop") in ("", None):
        return None
    by_target: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if float(row["hop"]) <= 2:
            by_target[row["target"]].append(row)

    ranges, regrets, top1_is_best = [], [], 0
    for cands in by_target.values():
        if len(cands) < 2:
            continue
        deltas = [c["delta_acc"] for c in cands]
        ranges.append(max(deltas) - min(deltas))
        top1 = min(cands, key=lambda c: c["kl_sym"])
        best = max(deltas)
        regrets.append(best - top1["delta_acc"])
        if abs(top1["delta_acc"] - best) < 1e-12:
            top1_is_best += 1
    if not ranges:
        return None
    return {
        "n_targets": len(ranges),
        "mean_delta_range": statistics.mean(ranges),
        "max_delta_range": max(ranges),
        "mean_top1_regret": statistics.mean(regrets),
        "n_targets_top1_is_best": top1_is_best,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pp(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.2f} pp" if signed else f"{value * 100:.2f} pp"


def num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def distance_tables(summary: dict) -> list[str]:
    gd = summary.get("graph_distance")
    if not gd:
        return ["- CSV 无 hop 列，未做图距离分析。"]
    means = gd["mean_delta_by_distance"]
    counts = gd["count_by_distance"]
    rows = ["| 图距离 | 配对数 | 平均 Δacc |", "|---:|---:|---:|"]
    for d in means:
        rows.append(f"| {d} | {counts[d]} | {pp(means[d])} |")
    hist = gd["best_distance_histogram"]
    hist_text = "、".join(f"距离 {d}: {v} 个" for d, v in hist.items())
    top1_hist = "、".join(f"距离 {d}: {v} 个" for d, v in gd["kl_top1_distance_histogram"].items())
    return rows + [
        "",
        f"- 事后最优候选的图距离分布：{hist_text}",
        f"- KL top-1 候选的图距离分布：{top1_hist}",
        f"- 最优候选落在两跳内的 target：{gd['n_targets_best_inside_two_hops']} / {gd['n_targets']}",
    ]


def build_report(label: str, summary: dict, per_target: list[dict], source: Path, note: str) -> str:
    comp = summary["n_targets_comparable"]
    mean_rho = summary["mean_within_target_spearman"]
    rho_text = "—" if mean_rho is None else f"{mean_rho:+.3f}"
    head = [
        f"# 同一目标客户端内的候选差异（{label}）",
        "",
        f"- 输入：`{source.name}`",
        f"- 配对总数 {summary['n_pairs_total']}，退化 head 剔除 {summary['n_dropped_degenerate']}，"
        f"过度集中剔除 {summary['n_dropped_majority']}，可用 {summary['n_pairs_clean']}",
        f"- 目标客户端 {summary['n_targets']} 个，其中候选数 ≥2 可比较 {comp} 个，"
        f"平均每目标 {summary['mean_candidates_per_target']:.1f} 个候选",
        f"- 条件：{note}",
        "",
        "## 一、同一目标内，收益差别有多大",
        "",
        f"- 目标内 Δacc 极差：均值 {pp(summary['mean_delta_range_within_target'], False)}，"
        f"中位 {pp(summary['median_delta_range_within_target'], False)}",
        f"- 可用配对平均 Δacc：{pp(summary['mean_delta_acc_clean'])}，"
        f"负迁移比例 {summary['negative_transfer_fraction_clean'] * 100:.1f}%",
        "",
        "## 二、KL 规则选中的伙伴是不是最好的",
        "",
        f"- 事后最优与 KL top-1 的差距（regret）：均值 {pp(summary['mean_top1_regret'])}，"
        f"中位 {pp(summary['median_top1_regret'])}，最大 {pp(summary['max_top1_regret'])}",
        f"- KL top-1 恰好就是最优的 target：{summary['n_targets_top1_is_best']} / {comp}",
        f"- KL top-1 优于 KL 第二名的 target：{summary['n_targets_top1_beats_second']} / {comp}",
        f"- 目标内 KL 与 Δacc 的 Spearman：均值 {rho_text}，中位 {num(summary['median_within_target_spearman'])}，"
        f"为负的 target {summary['n_within_target_spearman_negative']} / {summary['n_within_target_spearman']}",
        f"- 全部配对混算的 Spearman：{num(summary['overall_spearman_kl_vs_delta'])}（仅供对照，不作为结论）",
        "",
        "## 三、逐目标明细",
        "",
        "| target | 候选数 | KL top-1 | top-1 Δacc | 最优候选 | 最优 Δacc | regret | 目标内极差 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table = [
        f"| {r['target']} | {r['n_candidates']} | {r['kl_top1_candidate']} | "
        f"{pp(r['kl_top1_delta_acc'])} | {r['best_candidate']} | {pp(r['best_delta_acc'])} | "
        f"{pp(r['top1_regret'])} | {pp(r['delta_range'], False)} |"
        for r in per_target
    ]
    two = summary.get("two_hop_subset")
    if two:
        two_lines = [
            f"- 只看图距离 ≤2 的候选（原规则可见范围）：{two['n_targets']} 个 target 有 ≥2 个候选",
            f"- 这些候选之间的 Δacc 极差：均值 {pp(two['mean_delta_range'], False)}，"
            f"最大 {pp(two['max_delta_range'], False)}",
            f"- 其中 KL top-1 与事后最优的差：均值 {pp(two['mean_top1_regret'])}，"
            f"top-1 恰好最优 {two['n_targets_top1_is_best']} / {two['n_targets']}",
        ]
    else:
        two_lines = ["- 无足够的两跳内候选，未统计。"]
    tail = ["", "## 四、图距离", ""] + distance_tables(summary) + [
        "",
        "## 五、只看原规则可见的两跳内候选",
        "",
    ] + two_lines + [
        "",
        "## 六、边界",
        "",
        "- 单伙伴、单轮聚合：不是原方法的多伙伴集合，不能当作在线效果。",
        "- 同一目标的多个候选不独立，所有统计先按目标聚合，再取均值。",
        "- 图距离是快照邻接图上的最短路径长度，不等于多跳转发次数。",
        "- 结论只覆盖该快照的条件（轮次、种子、聚合范围）；跨种子或跨轮次需另行重复。",
        "",
    ]
    return "\n".join(head + table + tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--max-majority-fraction", type=float, default=1.0)
    args = parser.parse_args()

    source = Path(args.csv).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows, dropped_degenerate, dropped_majority = load_rows(source, args.max_majority_fraction)
    per_target, per_candidate = analyse(rows)
    summary = summarise(per_target, rows, dropped_degenerate, dropped_majority)

    write_csv(out / "per_candidate.csv", per_candidate, list(per_candidate[0].keys()))
    write_csv(out / "per_target.csv", per_target, list(per_target[0].keys()))
    (out / "summary.json").write_text(
        json.dumps({**summary, "label": args.label, "source": str(source)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "report.md").write_text(build_report(args.label, summary, per_target, source, args.note), encoding="utf-8")

    printable = {k: v for k, v in summary.items() if k != "graph_distance"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
