"""Paired same-snapshot aggregation check: all parameters vs base-only.

This script does not train. It loads models saved before aggregation, builds two
aggregated copies with identical neighbor/weight rules, verifies that base
parameters are identical and that the base-only branch keeps the local head,
then evaluates both on the target's held-out validation indices.
"""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fedml_api.data_preprocessing.cifar10.data_loader import CIFAR10_truncated, _data_transforms_cifar10
from fedml_api.model.cv.resnet import BaseHeadSplit, customized_resnet18


def build_model(class_num=10):
    base = customized_resnet18(class_num=class_num)
    head = copy.deepcopy(base.linear)
    base.linear = nn.Identity()
    return BaseHeadSplit(base=base, head=head)


def is_head_key(key):
    return key.startswith("head.") or key.startswith("linear.")


def aggregate(target_state, neighbor_states, target_n, neighbor_ns, scope):
    total = target_n + sum(neighbor_ns)
    out = {}
    for key, value in target_state.items():
        if scope == "base" and is_head_key(key):
            out[key] = value.detach().clone()
            continue
        acc = value.float() * (target_n / total)
        for state, n_j in zip(neighbor_states, neighbor_ns):
            acc = acc + state[key].float() * (n_j / total)
        out[key] = acc
    return out


def evaluate_state(state, loader, device, class_num=10):
    model = build_model(class_num=class_num).to(device)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    total = 0
    loss_sum = 0.0
    predictions = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
            predictions.append(preds.cpu().numpy())
            labels.append(y.cpu().numpy())
    predictions = np.concatenate(predictions) if predictions else np.asarray([])
    labels = np.concatenate(labels) if labels else np.asarray([])
    counts = np.bincount(predictions, minlength=10) if len(predictions) else np.zeros(10)
    label_counts = np.bincount(labels, minlength=10) if len(labels) else np.zeros(10)
    return {
        "accuracy": correct / max(1, total),
        "loss": loss_sum / max(1, total),
        "samples": total,
        "unique_predictions": int((counts > 0).sum()),
        "majority_prediction_fraction": float(counts.max() / max(1, total)),
        "true_unique_classes": int((label_counts > 0).sum()),
        "true_majority_fraction": float(label_counts.max() / max(1, total)),
    }


def max_abs_diff(state_a, state_b, keys):
    diffs = []
    for key in keys:
        diffs.append((state_a[key].float() - state_b[key].float()).abs().max().item())
    return max(diffs) if diffs else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="round_X directory containing metadata.json")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--clients", default="", help="comma-separated target client ids; empty means all")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint).resolve()
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    clients = {int(c["client_idx"]): c for c in metadata["clients"]}
    targets = [int(x) for x in args.clients.split(",") if x.strip()] or sorted(clients)
    device = torch.device(args.device)

    state_cache = {}
    for idx, client in clients.items():
        state_cache[idx] = torch.load(client["model_file"], map_location="cpu", weights_only=True)

    _, val_transform = _data_transforms_cifar10()
    loader_cache = {}
    for idx in targets:
        val_indices = clients[idx]["val_indices"]
        if not val_indices:
            raise ValueError(f"client {idx} has no validation indices")
        dataset = CIFAR10_truncated(
            args.data_dir, dataidxs=val_indices, train=True,
            transform=val_transform, download=False,
        )
        loader_cache[idx] = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
        )

    rows = []
    for idx in targets:
        client = clients[idx]
        neighbors = [int(j) for j in client["selected_neighbors"] if int(j) in clients]
        neighbor_states = [state_cache[j] for j in neighbors]
        target_n = int(client["train_num"])
        neighbor_ns = [int(clients[j]["train_num"]) for j in neighbors]
        target_state = state_cache[idx]

        all_state = aggregate(target_state, neighbor_states, target_n, neighbor_ns, "all")
        base_state = aggregate(target_state, neighbor_states, target_n, neighbor_ns, "base")

        base_keys = [k for k in target_state if not is_head_key(k)]
        head_keys = [k for k in target_state if is_head_key(k)]
        base_diff = max_abs_diff(all_state, base_state, base_keys)
        head_to_pre = max_abs_diff(base_state, target_state, head_keys)
        head_all_to_pre = max_abs_diff(all_state, target_state, head_keys)

        pre = evaluate_state(target_state, loader_cache[idx], device)
        agg_all = evaluate_state(all_state, loader_cache[idx], device)
        agg_base = evaluate_state(base_state, loader_cache[idx], device)

        rows.append({
            "client": idx,
            "neighbors": neighbors,
            "neighbor_train_nums": neighbor_ns,
            "target_train_num": target_n,
            "pre_acc": pre["accuracy"],
            "pre_loss": pre["loss"],
            "agg_all_acc": agg_all["accuracy"],
            "agg_all_loss": agg_all["loss"],
            "agg_base_acc": agg_base["accuracy"],
            "agg_base_loss": agg_base["loss"],
            "delta_all_acc": agg_all["accuracy"] - pre["accuracy"],
            "delta_base_acc": agg_base["accuracy"] - pre["accuracy"],
            "delta_all_loss": agg_all["loss"] - pre["loss"],
            "delta_base_loss": agg_base["loss"] - pre["loss"],
            "base_max_abs_diff": base_diff,
            "base_head_max_abs_diff_vs_pre": head_to_pre,
            "all_head_max_abs_diff_vs_pre": head_all_to_pre,
            "pre_unique_predictions": pre["unique_predictions"],
            "agg_all_unique_predictions": agg_all["unique_predictions"],
            "agg_base_unique_predictions": agg_base["unique_predictions"],
            "pre_majority_prediction_fraction": pre["majority_prediction_fraction"],
            "agg_all_majority_prediction_fraction": agg_all["majority_prediction_fraction"],
            "agg_base_majority_prediction_fraction": agg_base["majority_prediction_fraction"],
            "pre_true_classes": pre["true_unique_classes"],
            "true_majority_fraction": pre["true_majority_fraction"],
            "pre_degenerate": int(pre["unique_predictions"] < min(2, pre["true_unique_classes"])),
            "agg_base_degenerate": int(agg_base["unique_predictions"] < min(2, pre["true_unique_classes"])),
        })

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(checkpoint_dir), "rows": rows}, f, ensure_ascii=False, indent=2)
    with open(output.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "checkpoint": str(checkpoint_dir),
        "targets": targets,
        "base_max_abs_diff": max(row["base_max_abs_diff"] for row in rows),
        "base_head_max_abs_diff_vs_pre": max(row["base_head_max_abs_diff_vs_pre"] for row in rows),
        "mean_pre_acc": sum(row["pre_acc"] for row in rows) / len(rows),
        "mean_agg_all_acc": sum(row["agg_all_acc"] for row in rows) / len(rows),
        "mean_agg_base_acc": sum(row["agg_base_acc"] for row in rows) / len(rows),
        "rows": rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
