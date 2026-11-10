"""补 11-05 指导第 51 行：保存全部候选的训练后输出，统计延迟收益线索。

候选列表生成方式与 same_target_replay_paper100.py 完全一致，便于和三个固定分支对照。
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

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fedml_api.data_preprocessing.cifar10.data_loader import CIFAR10_truncated, _data_transforms_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import CIFAR100_truncated, _data_transforms_cifar100
from same_target_replay_paper100 import evaluate, kl_sym, replay_one_epoch, stratified_split
from torch.utils.data import DataLoader, Subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--class_num", type=int, default=0)
    ap.add_argument("--max_targets", type=int, default=0)
    ap.add_argument("--n_candidates", type=int, default=4)
    ap.add_argument("--candidate_seed", type=int, default=0)
    ap.add_argument("--select_frac", type=float, default=0.5)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--replay_epochs", type=int, default=1)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    meta = json.loads((ckpt / "metadata.json").read_text(encoding="utf-8"))
    clients = {int(c["client_idx"]): c for c in meta["clients"]}
    device = torch.device(args.device)

    if args.dataset == "cifar100":
        ds_cls = CIFAR100_truncated
        train_transform, val_transform = _data_transforms_cifar100()
        class_num = args.class_num or 100
    else:
        ds_cls = CIFAR10_truncated
        train_transform, val_transform = _data_transforms_cifar10()
        class_num = args.class_num or 10

    states = {i: torch.load(c["model_file"], map_location="cpu", weights_only=True)
              for i, c in clients.items()}
    protos = {i: torch.load(c["proto_file"], map_location="cpu", weights_only=True)
              for i, c in clients.items()}
    ns = {i: int(c["train_num"]) for i, c in clients.items()}

    graph = nx.Graph()
    graph.add_nodes_from(clients)
    graph.add_edges_from(meta.get("graph_edges", []))

    targets = sorted(clients)
    if args.max_targets > 0:
        targets = targets[: args.max_targets]

    rows = []
    for target in targets:
        c = clients[target]
        val_idx = c["val_indices"]
        if not val_idx:
            continue
        val_ds = ds_cls(args.data_dir, dataidxs=val_idx, train=True,
                        transform=val_transform, download=False)
        labels = [int(x) for x in val_ds.target]
        sel_pos, ver_pos = stratified_split(labels, args.select_frac, args.split_seed + target)
        sel_loader = DataLoader(Subset(val_ds, sel_pos), batch_size=args.eval_batch_size, shuffle=False)
        ver_loader = DataLoader(Subset(val_ds, ver_pos), batch_size=args.eval_batch_size, shuffle=False)
        train_ds = ds_cls(args.data_dir, dataidxs=c["train_indices"], train=True,
                          transform=train_transform, download=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)

        hops = nx.single_source_shortest_path_length(graph, target)
        base_state = states[target]

        none_before = evaluate(base_state, ver_loader, device, class_num)
        none_state = base_state
        for _ in range(args.replay_epochs):
            none_state = replay_one_epoch(none_state, train_loader, device, class_num,
                                          args.lr, args.weight_decay)
        none_after = evaluate(none_state, ver_loader, device, class_num)

        order = sorted((i for i in clients if i != target),
                       key=lambda j: kl_sym(protos[target], protos[j]))
        chosen = order[:2]
        rng = np.random.default_rng(args.candidate_seed + target)
        rest = [i for i in order[2:] if np.isfinite(kl_sym(protos[target], protos[i]))]
        pick = rng.choice(rest, size=min(len(rest), max(0, args.n_candidates - len(chosen))),
                          replace=False) if rest else []
        candidates = sorted(set(chosen + [int(i) for i in pick]))[: args.n_candidates]

        for rank, cand in enumerate(candidates, start=1):
            start = {}
            for key, value in base_state.items():
                if key.startswith("head.") or key.startswith("linear."):
                    start[key] = value.detach().clone()
                else:
                    acc = value.float() * (ns[target] / (ns[target] + ns[cand]))
                    acc = acc + states[cand][key].float() * (ns[cand] / (ns[target] + ns[cand]))
                    start[key] = acc

            before_sel = evaluate(start, sel_loader, device, class_num)
            before_ver = evaluate(start, ver_loader, device, class_num)
            after_state = start
            for _ in range(args.replay_epochs):
                after_state = replay_one_epoch(after_state, train_loader, device, class_num,
                                               args.lr, args.weight_decay)
            after_sel = evaluate(after_state, sel_loader, device, class_num)
            after_ver = evaluate(after_state, ver_loader, device, class_num)

            gain_before = before_ver["accuracy"] - none_before["accuracy"]
            gain_after = after_ver["accuracy"] - none_after["accuracy"]
            rows.append({
                "target": target,
                "candidate": cand,
                "kl_rank": rank,
                "kl_sym": kl_sym(protos[target], protos[cand]),
                "hop": int(hops.get(cand, -1)),
                "two_hop_reachable": int(hops.get(cand, 99) <= 2),
                "select_loss_before": before_sel["loss"],
                "ver_acc_before": before_ver["accuracy"],
                "ver_acc_after": after_ver["accuracy"],
                "gain_before_pp": gain_before * 100,
                "gain_after_pp": gain_after * 100,
                "delayed_gain": int(gain_before <= 1e-12 and gain_after > 1e-12),
                "reversed_gain": int(gain_before > 1e-12 and gain_after < -1e-12),
                "none_ver_acc_before": none_before["accuracy"],
                "none_ver_acc_after": none_after["accuracy"],
            })

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_target = {}
    for r in rows:
        by_target.setdefault(r["target"], []).append(r)
    best_after = [max(sub, key=lambda r: r["ver_acc_after"]) for sub in by_target.values()]
    kl_rank_of_best = int(np.mean([
        next(r["kl_rank"] for r in by_target[t] if r["candidate"] == b["candidate"])
        for t, b in zip(by_target, best_after)
    ]) * 100) / 100

    summary = {
        "checkpoint": str(ckpt),
        "n_targets": len(by_target),
        "n_rows": len(rows),
        "delayed_gain": f"{sum(r['delayed_gain'] for r in rows)}/{len(rows)}",
        "reversed_gain": f"{sum(r['reversed_gain'] for r in rows)}/{len(rows)}",
        "mean_gain_before_pp": float(np.mean([r["gain_before_pp"] for r in rows])),
        "mean_gain_after_pp": float(np.mean([r["gain_after_pp"] for r in rows])),
        "mean_kl_rank_of_best_after": kl_rank_of_best,
        "targets_with_delayed_candidate": int(sum(
            1 for sub in by_target.values() if any(r["delayed_gain"] for r in sub)
        )),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
