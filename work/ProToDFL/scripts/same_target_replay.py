"""Short-range replay for same-target partner selection.

For each target client the local training indices are split once into a fixed
selection half and a check half. One candidate at a time is aggregated into the
target's base, then the aggregated model is trained for a small number of local
epochs on the selection half and evaluated on the check half. Two quantities are
reported per candidate:

    delta_agg    = accuracy after aggregation - accuracy before aggregation
    delta_replay = accuracy after replay training - accuracy before aggregation

`delta_agg` mirrors the offline CSV check; `delta_replay` shows whether the
ranking survives one local epoch. Selection never touches the check half.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fedml_api.data_preprocessing.cifar10.data_loader import CIFAR10_truncated, _data_transforms_cifar10
from kl_benefit_distribution import kl_between
from paired_aggregation_check import aggregate, build_model, evaluate_state, is_head_key


def split_indices(indices: list[int], frac: float, seed: int) -> tuple[list[int], list[int]]:
    arr = np.asarray(indices)
    order = np.random.default_rng(seed).permutation(len(arr))
    cut = max(1, min(len(arr) - 1, int(round(len(arr) * frac))))
    return arr[order[:cut]].tolist(), arr[order[cut:]].tolist()


def make_loader(data_dir, indices, transform, batch_size, shuffle):
    dataset = CIFAR10_truncated(
        data_dir, dataidxs=indices, train=True, transform=transform, download=False
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_state(state, loader, device, train_args, epochs, round_idx):
    model = build_model(class_num=10).to(device)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.train()
    lr = train_args["lr"] * (train_args["lr_decay"] ** round_idx)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=train_args["momentum"],
        weight_decay=train_args["wd"],
    )
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def summarise(rows: list[dict], key: str) -> dict:
    clean = [r for r in rows if not r["degenerate"]]
    if not clean:
        return {"n": 0}
    per_target: dict[int, list[dict]] = {}
    for row in clean:
        per_target.setdefault(row["target"], []).append(row)

    ranges, regrets, rho_values = [], [], []
    top1_is_best = 0
    for subset in per_target.values():
        if len(subset) < 2:
            continue
        values = [r[key] for r in subset]
        ranges.append(max(values) - min(values))
        top1 = min(subset, key=lambda r: r["kl_sym"])
        best = max(values)
        regrets.append(best - top1[key])
        if abs(top1[key] - best) < 1e-12:
            top1_is_best += 1
        if len(subset) >= 3:
            rho = spearmanr([r["kl_sym"] for r in subset], values, nan_policy="omit")
            if rho.statistic == rho.statistic:
                rho_values.append(float(rho.statistic))

    return {
        "n": len(clean),
        "n_targets_with_multiple": len(ranges),
        "mean": float(np.mean([r[key] for r in clean])),
        "negative_fraction": float(np.mean([r[key] < 0 for r in clean])),
        "mean_range_within_target": float(np.mean(ranges)) if ranges else None,
        "max_range_within_target": float(np.max(ranges)) if ranges else None,
        "mean_top1_regret": float(np.mean(regrets)) if regrets else None,
        "n_top1_is_best": top1_is_best,
        "mean_within_target_spearman": float(np.mean(rho_values)) if rho_values else None,
    }


def fmt_pp(value) -> str:
    return "—" if value is None else f"{value * 100:+.2f} pp"


def build_report(rows, summary_agg, summary_replay, args, metadata) -> str:
    rho_value = summary_replay.get("mean_within_target_spearman")
    rho_text = "—" if rho_value is None else f"{rho_value:.3f}"
    lines = [
        "# 同目标候选短程重放（选择集 / 核查集固定二分）",
        "",
        f"- 快照：`{args.checkpoint}`",
        f"- 聚合范围：`{args.scope}`；重放本地 epoch：{args.replay_epochs}",
        f"- 二分：选择集 {args.split_frac:.2f} / 核查集 {1 - args.split_frac:.2f}，split seed {args.split_seed}",
        f"- 候选上限：{args.max_candidates}（取图距离 ≤2 且 KL 最小的若干候选）",
        f"- 训练超参（取自快照 args）：lr {metadata['lr']} × decay {metadata['lr_decay']}^round，"
        f"momentum {metadata['momentum']}，wd {metadata['wd']}",
        "",
        "## 一、聚合后（不训练）",
        "",
        f"- 候选数 {summary_agg.get('n', 0)}；平均 Δacc {fmt_pp(summary_agg.get('mean'))}；"
        f"变差比例 {summary_agg.get('negative_fraction', 0) * 100:.1f}%",
        f"- 目标内极差均值 {fmt_pp(summary_agg.get('mean_range_within_target'))}；"
        f"KL top-1 与最优差 {fmt_pp(summary_agg.get('mean_top1_regret'))}；"
        f"top-1 即最优 {summary_agg.get('n_top1_is_best', 0)} / {summary_agg.get('n_targets_with_multiple', 0)}",
        "",
        "## 二、重放 1 个本地 epoch 后",
        "",
        f"- 候选数 {summary_replay.get('n', 0)}；平均 Δacc {fmt_pp(summary_replay.get('mean'))}；"
        f"仍变差比例 {summary_replay.get('negative_fraction', 0) * 100:.1f}%",
        f"- 目标内极差均值 {fmt_pp(summary_replay.get('mean_range_within_target'))}；"
        f"KL top-1 与最优差 {fmt_pp(summary_replay.get('mean_top1_regret'))}；"
        f"top-1 即最优 {summary_replay.get('n_top1_is_best', 0)} / {summary_replay.get('n_targets_with_multiple', 0)}",
        f"- 目标内 KL 与 Δacc 的 Spearman 均值 {rho_text}",
        "",
        "## 三、逐候选",
        "",
        "| target | 候选 | 图距离 | KL | 聚合前 | 聚合后 | Δ聚合 | 重放后 | Δ重放 | 退化 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r["target"], r["candidate"])):
        lines.append(
            f"| {row['target']} | {row['candidate']} | {row['hop']} | {row['kl_sym']:.4f} | "
            f"{row['pre_acc'] * 100:.2f}% | {row['agg_acc'] * 100:.2f}% | {fmt_pp(row['delta_agg'])} | "
            f"{row['replay_acc'] * 100:.2f}% | {fmt_pp(row['delta_replay'])} | "
            f"{'是' if row['degenerate'] else '否'} |"
        )
    lines += [
        "",
        "## 四、边界",
        "",
        "- 重放用核查集一半的样本训练、另一半评估，训练量小于原训练，Δacc 幅度不能与完整训练直接比。",
        "- 单伙伴、单轮、单种子；候选之间不独立，统计按目标聚合。",
        "- 选择集只用于挑选伙伴，所有报出的 Δacc 都来自核查集。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--scope", choices=["base", "all"], default="base")
    parser.add_argument("--replay-epochs", type=int, default=1)
    parser.add_argument("--split-frac", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--targets", default="", help="comma-separated target ids; empty means all")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint).resolve()
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    train_args = metadata["args"]
    clients = {int(c["client_idx"]): c for c in metadata["clients"]}
    device = torch.device(args.device)

    def resolve(path_str: str) -> str:
        path = Path(path_str)
        if path.exists():
            return str(path)
        return str(checkpoint_dir / path.name)

    states = {
        idx: torch.load(resolve(client["model_file"]), map_location="cpu", weights_only=True)
        for idx, client in clients.items()
    }
    prototypes = {
        idx: torch.load(resolve(client["proto_file"]), map_location="cpu", weights_only=True)
        for idx, client in clients.items()
    }
    graph = nx.Graph()
    graph.add_nodes_from(clients)
    graph.add_edges_from(metadata.get("graph_edges", []))

    train_tf, val_tf = _data_transforms_cifar10()
    targets = sorted(clients)
    if args.targets.strip():
        targets = [int(x) for x in args.targets.split(",") if x.strip()]
    if args.max_targets > 0:
        targets = targets[:args.max_targets]

    rows = []
    for target in targets:
        client = clients[target]
        select_idx, check_idx = split_indices(
            client["train_indices"], args.split_frac, args.split_seed
        )
        check_loader = make_loader(args.data_dir, check_idx, val_tf, args.batch_size, False)
        pre = evaluate_state(states[target], check_loader, device)

        hops = nx.single_source_shortest_path_length(graph, target)
        candidates = [
            idx for idx in sorted(clients)
            if idx != target and hops.get(idx, 99) <= 2
        ]
        candidates.sort(key=lambda idx: kl_between(prototypes[target], prototypes[idx])[2])
        candidates = candidates[:args.max_candidates]
        if not candidates:
            continue

        print(f"target {target}: {len(candidates)} candidates, "
              f"select {len(select_idx)} / check {len(check_idx)} samples", flush=True)
        select_loader = make_loader(args.data_dir, select_idx, train_tf, args.batch_size, True)

        for candidate in candidates:
            candidate_n = int(clients[candidate]["train_num"])
            aggregated = aggregate(
                states[target], [states[candidate]], int(client["train_num"]), [candidate_n], args.scope
            )
            agg_metrics = evaluate_state(aggregated, check_loader, device)
            replayed = train_state(
                aggregated, select_loader, device, train_args, args.replay_epochs, int(metadata["round"])
            )
            replay_metrics = evaluate_state(replayed, check_loader, device)
            kl_ij, kl_ji, kl_sym = kl_between(prototypes[target], prototypes[candidate])
            degenerate = int(
                pre["unique_predictions"] < min(2, pre["true_unique_classes"])
                or replay_metrics["unique_predictions"] < min(2, pre["true_unique_classes"])
            )
            rows.append({
                "target": target,
                "candidate": candidate,
                "hop": int(hops[candidate]),
                "kl_ij": kl_ij,
                "kl_ji": kl_ji,
                "kl_sym": kl_sym,
                "pre_acc": pre["accuracy"],
                "agg_acc": agg_metrics["accuracy"],
                "replay_acc": replay_metrics["accuracy"],
                "delta_agg": agg_metrics["accuracy"] - pre["accuracy"],
                "delta_replay": replay_metrics["accuracy"] - pre["accuracy"],
                "training_gain": replay_metrics["accuracy"] - agg_metrics["accuracy"],
                "pre_loss": pre["loss"],
                "agg_loss": agg_metrics["loss"],
                "replay_loss": replay_metrics["loss"],
                "degenerate": degenerate,
            })
            print(f"  candidate {candidate}: agg {fmt_pp(rows[-1]['delta_agg'])}, "
                  f"replay {fmt_pp(rows[-1]['delta_replay'])}", flush=True)

    if not rows:
        raise SystemExit("no candidate rows produced")

    summary_agg = summarise(rows, "delta_agg")
    summary_replay = summarise(rows, "delta_replay")
    result = {
        "checkpoint": str(checkpoint_dir),
        "scope": args.scope,
        "replay_epochs": args.replay_epochs,
        "split_frac": args.split_frac,
        "split_seed": args.split_seed,
        "max_candidates": args.max_candidates,
        "summary_agg": summary_agg,
        "summary_replay": summary_replay,
        "rows": rows,
    }

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "per_candidate.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output / "report.md").write_text(
        build_report(rows, summary_agg, summary_replay, args, train_args), encoding="utf-8"
    )
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
