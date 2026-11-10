"""确定性重复重放: 三分支各独立跑 3 遍, 先确认同一个伙伴成绩能对上。

训练图片预先按固定 seed 增强成固定批次, 三组三遍共用同一批。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fedml_api.data_preprocessing.cifar10.data_loader import CIFAR10_truncated, _data_transforms_cifar10
from paired_aggregation_check import aggregate, build_model
from same_target_replay_paper100 import evaluate, kl_sym, stratified_split


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fixed_batches(loader, seed):
    reset_rng(seed)
    return [(x.clone(), y.clone()) for x, y in loader]


def replay_fixed(state, batches, device, class_num, lr, wd, seed):
    reset_rng(seed)
    model = build_model(class_num=class_num).to(device)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    for x, y in batches:
        x, y = x.to(device), y.to(device).long()
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--targets", default="0,1,2,3,4")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n_candidates", type=int, default=4)
    ap.add_argument("--candidate_seed", type=int, default=0)
    ap.add_argument("--select_frac", type=float, default=0.5)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    args = ap.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    ckpt = Path(args.checkpoint).resolve()
    meta = json.loads((ckpt / "metadata.json").read_text(encoding="utf-8"))
    clients = {int(c["client_idx"]): c for c in meta["clients"]}
    targets = [int(x) for x in args.targets.split(",") if x.strip()]

    ds_cls = CIFAR10_truncated
    train_transform, val_transform = _data_transforms_cifar10()
    class_num = 10

    states = {i: torch.load(c["model_file"], map_location="cpu", weights_only=True)
              for i, c in clients.items()}
    protos = {i: torch.load(c["proto_file"], map_location="cpu", weights_only=True)
              for i, c in clients.items()}
    ns = {i: int(c["train_num"]) for i, c in clients.items()}

    graph = nx.Graph()
    graph.add_nodes_from(clients)
    graph.add_edges_from(meta.get("graph_edges", []))

    rows = []
    for target in targets:
        c = clients[target]
        val_ds = ds_cls(args.data_dir, dataidxs=c["val_indices"], train=True,
                        transform=val_transform, download=False)
        sel_pos, ver_pos = stratified_split([int(x) for x in val_ds.target],
                                            args.select_frac, args.split_seed + target)
        sel_loader = DataLoader(Subset(val_ds, sel_pos), batch_size=args.eval_batch_size, shuffle=False)
        ver_loader = DataLoader(Subset(val_ds, ver_pos), batch_size=args.eval_batch_size, shuffle=False)
        train_ds = ds_cls(args.data_dir, dataidxs=c["train_indices"], train=True,
                          transform=train_transform, download=False)
        batches = fixed_batches(DataLoader(train_ds, batch_size=args.batch_size, shuffle=False),
                                1000 + target)

        hops = nx.single_source_shortest_path_length(graph, target)
        base = states[target]
        order = sorted((i for i in clients if i != target),
                       key=lambda j: kl_sym(protos[target], protos[j]))
        chosen = order[:2]
        rng = np.random.default_rng(args.candidate_seed + target)
        rest = [i for i in order[2:] if np.isfinite(kl_sym(protos[target], protos[i]))]
        pick = rng.choice(rest, size=min(len(rest), max(0, args.n_candidates - len(chosen))),
                          replace=False) if rest else []
        candidates = sorted(set(chosen + [int(i) for i in pick]))[: args.n_candidates]

        select_loss = {}
        for cand in candidates:
            agg = aggregate(base, [states[cand]], ns[target], [ns[cand]], "base")
            select_loss[cand] = evaluate(agg, sel_loader, device, class_num)["loss"]
        kl_top1 = order[0]
        select_best = min(candidates, key=lambda i: select_loss[i])
        branches = {"none": None, "kl_top1": kl_top1, "select_best": select_best}

        for name, cand in branches.items():
            if cand is None:
                start = {k: v.detach().clone() for k, v in base.items()}
            else:
                start = aggregate(base, [states[cand]], ns[target], [ns[cand]], "base")
            before = evaluate(start, ver_loader, device, class_num)
            for rep in range(1, args.repeats + 1):
                after_state = replay_fixed(start, batches, device, class_num,
                                           args.lr, args.weight_decay, 2000 + target)
                after = evaluate(after_state, ver_loader, device, class_num)
                rows.append({
                    "target": target,
                    "branch": name,
                    "candidate": "" if cand is None else cand,
                    "repeat": rep,
                    "ver_acc_before": before["accuracy"],
                    "ver_loss_before": before["loss"],
                    "ver_acc_after": after["accuracy"],
                    "ver_loss_after": after["loss"],
                    "kl_top1": kl_top1,
                    "select_best": select_best,
                    "candidates": "|".join(str(x) for x in candidates),
                })

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    groups = {}
    for r in rows:
        groups.setdefault((r["target"], r["branch"]), []).append(r)
    exact = 0
    max_loss_gap = 0.0
    for group in groups.values():
        accs = [r["ver_acc_after"] for r in group]
        losses = [r["ver_loss_after"] for r in group]
        if len(set(accs)) == 1:
            exact += 1
        max_loss_gap = max(max_loss_gap, max(losses) - min(losses))

    first = {k: next(r for r in v if r["repeat"] == 1) for k, v in groups.items()}
    branches_summary = {}
    for name in ("none", "kl_top1", "select_best"):
        sub = [r for (t, b), r in first.items() if b == name]
        branches_summary[name] = {
            "ver_acc_after_mean": float(np.mean([r["ver_acc_after"] for r in sub])),
            "ver_acc_before_mean": float(np.mean([r["ver_acc_before"] for r in sub])),
        }
    for name in ("kl_top1", "select_best"):
        none_map = {t: r["ver_acc_after"] for (t, b), r in first.items() if b == "none"}
        gains = [r["ver_acc_after"] - none_map[t]
                 for (t, b), r in first.items() if b == name]
        branches_summary[name]["gain_vs_none_mean_pp"] = float(np.mean(gains) * 100)

    summary = {
        "checkpoint": str(ckpt),
        "targets": targets,
        "repeats": args.repeats,
        "n_groups": len(groups),
        "groups_with_identical_acc": exact,
        "max_ver_loss_gap_across_repeats": max_loss_gap,
        "branches": branches_summary,
    }
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
