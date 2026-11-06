"""Offline KL-vs-validation-benefit distribution for same-snapshot partners.

For each target client, aggregate its base with one candidate at a time under a
fixed aggregation rule (base-only by default), evaluate on the target's held-out
validation indices, and compare the observed gain with prototype KL and hop
distance. No training and no test-set selection.
"""

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fedml_api.data_preprocessing.cifar10.data_loader import CIFAR10_truncated, _data_transforms_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import CIFAR100_truncated, _data_transforms_cifar100
from paired_aggregation_check import aggregate, build_model, evaluate_state, is_head_key


def kl_between(proto_i, proto_j):
    p = F.softmax(proto_i.reshape(1, -1), dim=1)
    q = F.softmax(proto_j.reshape(1, -1), dim=1)
    forward = torch.sum(p * torch.log(p / q)) * 1000
    backward = torch.sum(q * torch.log(q / p)) * 1000
    forward = torch.nan_to_num(forward, nan=1e12, posinf=1e12, neginf=-1e12).item()
    backward = torch.nan_to_num(backward, nan=1e12, posinf=1e12, neginf=-1e12).item()
    return float(forward), float(backward), float((forward + backward) / 2)


def bin_summary(kls, deltas, n_bins=8):
    if len(kls) == 0:
        return []
    order = np.argsort(kls)
    chunks = np.array_split(order, min(n_bins, len(order)))
    bins = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        d = np.asarray([deltas[i] for i in chunk], dtype=float)
        k = np.asarray([kls[i] for i in chunk], dtype=float)
        bins.append({
            "count": int(len(chunk)),
            "kl_min": float(k.min()),
            "kl_max": float(k.max()),
            "delta_mean": float(d.mean()),
            "delta_median": float(np.median(d)),
            "delta_std": float(d.std(ddof=0)),
            "delta_min": float(d.min()),
            "delta_max": float(d.max()),
            "negative_transfer_fraction": float((d < 0).mean()),
        })
    return bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="round_X directory containing metadata.json")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--scope", choices=["base", "all"], default="base")
    parser.add_argument("--max_targets", type=int, default=0)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--num_classes", type=int, default=0, help="0 means infer from dataset")
    parser.add_argument("--max_candidates", type=int, default=0,
                        help="randomly sample at most N candidates per target; 0 means all")
    parser.add_argument("--candidate_seed", type=int, default=0,
                        help="seed for candidate subsampling, keeps the pool identical across targets")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint).resolve()
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    clients = {int(c["client_idx"]): c for c in metadata["clients"]}
    targets = sorted(clients)
    if args.max_targets > 0:
        targets = targets[:args.max_targets]
    all_candidates = sorted(clients)
    device = torch.device(args.device)

    states = {
        idx: torch.load(client["model_file"], map_location="cpu", weights_only=True)
        for idx, client in clients.items()
    }
    prototypes = {
        idx: torch.load(client["proto_file"], map_location="cpu", weights_only=True)
        for idx, client in clients.items()
    }

    graph = nx.Graph()
    graph.add_nodes_from(clients)
    graph.add_edges_from(metadata.get("graph_edges", []))
    if args.dataset == "cifar100":
        dataset_cls = CIFAR100_truncated
        _, val_transform = _data_transforms_cifar100()
        class_num = args.num_classes or 100
    else:
        dataset_cls = CIFAR10_truncated
        _, val_transform = _data_transforms_cifar10()
        class_num = args.num_classes or 10

    rows = []
    for target in targets:
        client = clients[target]
        val_indices = client["val_indices"]
        if not val_indices:
            raise ValueError(
                "client {} has no val_indices; the checkpoint was produced without validation_ratio".format(target)
            )
        dataset = dataset_cls(
            args.data_dir, dataidxs=val_indices, train=True,
            transform=val_transform, download=False,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        pre = evaluate_state(states[target], loader, device, class_num=class_num)
        hops = nx.single_source_shortest_path_length(graph, target)

        target_n = int(client["train_num"])
        if args.max_candidates > 0 and len(all_candidates) - 1 > args.max_candidates:
            pool = [c for c in all_candidates if c != target]
            candidates_iter = sorted(
                np.random.default_rng(args.candidate_seed).choice(
                    pool, size=args.max_candidates, replace=False
                ).tolist()
            )
        else:
            candidates_iter = all_candidates
        for candidate in candidates_iter:
            if candidate == target:
                continue
            candidate_client = clients[candidate]
            neighbor_state = states[candidate]
            candidate_n = int(candidate_client["train_num"])
            aggregated = aggregate(
                states[target], [neighbor_state], target_n, [candidate_n], args.scope
            )
            metrics = evaluate_state(aggregated, loader, device, class_num=class_num)
            kl_ij, kl_ji, kl_sym = kl_between(prototypes[target], prototypes[candidate])
            rows.append({
                "target": target,
                "candidate": candidate,
                "hop": int(hops.get(candidate, -1)),
                "kl_ij": kl_ij,
                "kl_ji": kl_ji,
                "kl_sym": kl_sym,
                "pre_acc": pre["accuracy"],
                "pre_loss": pre["loss"],
                "agg_acc": metrics["accuracy"],
                "agg_loss": metrics["loss"],
                "delta_acc": metrics["accuracy"] - pre["accuracy"],
                "delta_loss": metrics["loss"] - pre["loss"],
                "negative_transfer": int(metrics["accuracy"] < pre["accuracy"]),
                "pre_unique_predictions": pre["unique_predictions"],
                "agg_unique_predictions": metrics["unique_predictions"],
                "pre_majority_prediction_fraction": pre["majority_prediction_fraction"],
                "agg_majority_prediction_fraction": metrics["majority_prediction_fraction"],
                "pre_true_classes": pre["true_unique_classes"],
                "true_majority_fraction": pre["true_majority_fraction"],
                "degenerate_head": int(
                    pre["unique_predictions"] < min(2, pre["true_unique_classes"])
                    or metrics["unique_predictions"] < min(2, pre["true_unique_classes"])
                ),
            })

    clean_rows = [row for row in rows if not row["degenerate_head"]]
    kls = [row["kl_sym"] for row in clean_rows]
    deltas = [row["delta_acc"] for row in clean_rows]
    losses = [row["delta_loss"] for row in clean_rows]
    rho = spearmanr(kls, deltas, nan_policy="omit")
    tau = kendalltau(kls, deltas, nan_policy="omit")
    loss_rho = spearmanr(kls, losses, nan_policy="omit")

    per_target = []
    top1_regret = []
    for target in targets:
        subset = [row for row in clean_rows if row["target"] == target]
        if len(subset) < 3:
            continue
        r = spearmanr(
            [row["kl_sym"] for row in subset],
            [row["delta_acc"] for row in subset],
            nan_policy="omit",
        )
        best = max(subset, key=lambda row: row["delta_acc"])
        top1 = min(subset, key=lambda row: row["kl_sym"])
        top1_regret.append(best["delta_acc"] - top1["delta_acc"])
        per_target.append({
            "target": target,
            "spearman_kl_vs_delta": float(r.statistic) if r.statistic == r.statistic else None,
            "top1_kl_candidate": top1["candidate"],
            "top1_kl_delta": top1["delta_acc"],
            "best_candidate": best["candidate"],
            "best_delta": best["delta_acc"],
            "top1_regret": best["delta_acc"] - top1["delta_acc"],
        })

    summary = {
        "checkpoint": str(checkpoint_dir),
        "scope": args.scope,
        "n_pairs": len(rows),
        "n_degenerate_pairs": len(rows) - len(clean_rows),
        "n_pairs_clean": len(clean_rows),
        "overall_spearman_kl_vs_delta_clean": float(rho.statistic),
        "overall_kendall_kl_vs_delta_clean": float(tau.statistic),
        "overall_spearman_kl_vs_delta_loss_clean": float(loss_rho.statistic),
        "mean_delta_clean": float(np.mean(deltas)),
        "median_delta_clean": float(np.median(deltas)),
        "negative_transfer_fraction_clean": float(np.mean([d < 0 for d in deltas])),
        "zero_delta_fraction_clean": float(np.mean([abs(d) < 1e-12 for d in deltas])),
        "mean_top1_regret_clean": float(np.mean(top1_regret)) if top1_regret else None,
        "per_target": per_target,
        "kl_bins_clean": bin_summary(kls, deltas),
        "rows": rows,
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    import csv
    with open(output.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    printable = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
