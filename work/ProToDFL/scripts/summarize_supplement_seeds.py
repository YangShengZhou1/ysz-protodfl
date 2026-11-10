"""把多个 seed 的同目标候选分析结果并列成一张对照表。

输入是 `same_target_candidate_analysis.py` 产出的 summary.json（每个设置一份），
输出 markdown 表到指定路径，并在终端打印同一张表。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = [
    ("n_pairs_clean", "可用配对", "{:d}"),
    ("n_targets_comparable", "可比较目标", "{:d}"),
    ("mean_delta_acc_clean", "平均 Δacc", "{:+.2%}"),
    ("negative_transfer_fraction_clean", "负迁移比例", "{:.1%}"),
    ("mean_delta_range_within_target", "目标内极差均值", "{:.2%}"),
    ("mean_top1_regret", "KL top-1 regret 均值", "{:.2%}"),
    ("n_targets_top1_is_best", "top-1 即最优", "{:d}"),
    ("mean_within_target_spearman", "目标内 Spearman 均值", "{:+.3f}"),
    ("overall_spearman_kl_vs_delta", "全部混算 Spearman", "{:+.3f}"),
]

TWO_HOP_ROWS = [
    ("n_targets", "两跳内目标", "{:d}"),
    ("mean_delta_range", "两跳内极差均值", "{:.2%}"),
    ("mean_top1_regret", "两跳内 regret 均值", "{:.2%}"),
    ("n_targets_top1_is_best", "两跳内 top-1 即最优", "{:d}"),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, spec: str) -> str:
    if value is None:
        return "—"
    if spec.endswith("d}"):
        return spec.format(int(value))
    return spec.format(float(value))


def build_table(summaries: list[tuple[str, dict]], rows, getter) -> list[str]:
    header = "| 指标 | " + " | ".join(name for name, _ in summaries) + " |"
    sep = "|---|" + "---:|" * len(summaries)
    lines = [header, sep]
    for key, label, spec in rows:
        cells = [fmt(getter(data, key), spec) for _, data in summaries]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="label=path/to/summary.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="同目标候选分析：多 seed 对照")
    args = parser.parse_args()

    summaries = []
    for item in args.inputs:
        label, path = item.split("=", 1)
        summaries.append((label, load(Path(path))))

    main_lines = build_table(summaries, ROWS, lambda d, k: d.get(k))
    hop_lines = build_table(summaries, TWO_HOP_ROWS, lambda d, k: (d.get("two_hop_subset") or {}).get(k))

    text = "\n".join(
        [f"# {args.title}", "", "## 全图口径", "", *main_lines, "", "## 两跳内口径", "", *hop_lines, ""]
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"written: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
