"""按 2026-11-05 指导方案第三节：同目标候选比较 + 相同起点 1-epoch 重放 + 选择/核查分离。

固定条件：
- 只聚合 base，保留目标自己的 head；
- 所有分支从完全相同的本地模型出发，优化器每次重建（不是原训练的精确续跑）；
- 留出验证集按类别分层固定切成选择/核查两半，伙伴择优只看选择集；
- 样本顺序固定（shuffle=False），评价统一 eval 模式；
- 记录候选跳数与两跳可达性，超出两跳的候选标注为离线诊断可见。
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fedml_api.data_preprocessing.cifar10.data_loader import CIFAR10_truncated, _data_transforms_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import CIFAR100_truncated, _data_transforms_cifar100
from paired_aggregation_check import aggregate, build_model, evaluate_state


def kl_sym(a, b):
    p = F.softmax(a.reshape(1, -1), dim=1)
    q = F.softmax(b.reshape(1, -1), dim=1)
    f = torch.nan_to_num(torch.sum(p * torch.log(p / q)) * 1000, nan=1e12, posinf=1e12).item()
    g = torch.nan_to_num(torch.sum(q * torch.log(q / p)) * 1000, nan=1e12, posinf=1e12).item()
    return (f + g) / 2


def stratified_split(labels, frac, seed):
    rng = np.random.default_rng(seed)
    by_cls = {}
    for pos, lab in enumerate(labels):
        by_cls.setdefault(int(lab), []).append(pos)
    select, verify = [], []
    for poss in by_cls.values():
        rng.shuffle(poss)
        k = max(1, int(round(len(poss) * frac)))
        select.extend(poss[:k])
        verify.extend(poss[k:])
    return sorted(select), sorted(verify)


def evaluate(state, loader, device, class_num):
    return evaluate_state(state, loader, device, class_num=class_num)


def replay_one_epoch(state, train_loader, device, class_num, lr, wd):
    model = build_model(class_num=class_num).to(device)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    for x, y in train_loader:
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

    states = {
        i: torch.load(c["model_file"], map_location="cpu", weights_only=True)
        for i, c in clients.items()
    }
    protos = {
        i: torch.load(c["proto_file"], map_location="cpu", weights_only=True)
        for i, c in clients.items()
    }
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
        order = sorted(
            (i for i in clients if i != target),
            key=lambda j: kl_sym(protos[target], protos[j]),
        )
        chosen = order[:2]
        rng = np.random.default_rng(args.candidate_seed + target)
        rest = [i for i in order[2:] if np.isfinite(kl_sym(protos[target], protos[i]))]
        pick = rng.choice(rest, size=min(len(rest), max(0, args.n_candidates - len(chosen))), replace=False) if rest else []
        candidates = sorted(set(chosen + [int(i) for i in pick]))[: args.n_candidates]

        base_state = states[target]
        cand_info = []
        select_loss = {}
        for cand in candidates:
            agg = aggregate(base_state, [states[cand]], ns[target], [ns[cand]], "base")
            m = evaluate(agg, sel_loader, device, class_num)
            select_loss[cand] = m["loss"]
            cand_info.append({
                "candidate": cand,
                "kl_sym": kl_sym(protos[target], protos[cand]),
                "hop": int(hops.get(cand, -1)),
                "two_hop_reachable": int(hops.get(cand, 99) <= 2),
                "select_loss": m["loss"],
            })

        kl_top1 = order[0]
        select_best = min(candidates, key=lambda i: select_loss[i])
        branches = {
            "none": None,
            "kl_top1": kl_top1,
            "select_best": select_best,
        }

        for name, cand in branches.items():
            if cand is None:
                start = {k: v.detach().clone() for k, v in base_state.items()}
            else:
                start = aggregate(base_state, [states[cand]], ns[target], [ns[cand]], "base")

            before_sel = evaluate(start, sel_loader, device, class_num)
            before_ver = evaluate(start, ver_loader, device, class_num)
            after_state = start
            for _ in range(args.replay_epochs):
                after_state = replay_one_epoch(after_state, train_loader, device, class_num,
                                               args.lr, args.weight_decay)
            after_sel = evaluate(after_state, sel_loader, device, class_num)
            after_ver = evaluate(after_state, ver_loader, device, class_num)

            rows.append({
                "target": target,
                "branch": name,
                "candidate": "" if cand is None else cand,
                "candidate_kl": "" if cand is None else kl_sym(protos[target], protos[cand]),
                "candidate_hop": "" if cand is None else int(hops.get(cand, -1)),
                "select_n": len(sel_pos),
                "verify_n": len(ver_pos),
                "sel_acc_before": before_sel["accuracy"],
                "sel_loss_before": before_sel["loss"],
                "ver_acc_before": before_ver["accuracy"],
                "ver_loss_before": before_ver["loss"],
                "sel_acc_after": after_sel["accuracy"],
                "sel_loss_after": after_sel["loss"],
                "ver_acc_after": after_ver["accuracy"],
                "ver_loss_after": after_ver["loss"],
                "n_candidates": len(candidates),
                "kl_top1": kl_top1,
                "select_best": select_best,
                "select_best_is_kl_top1": int(select_best == kl_top1),
                "candidates_json": json.dumps(cand_info),
            })

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"checkpoint": str(ckpt), "n_targets": len(targets), "n_rows": len(rows), "branches": {}}
    for name in ("none", "kl_top1", "select_best"):
        sub = [r for r in rows if r["branch"] == name]
        if not sub:
            continue
        col = lambda k: np.array([r[k] for r in sub], dtype=float)
        summary["branches"][name] = {
            "n": len(sub),
            "verify_acc_before_mean": float(col("ver_acc_before").mean()),
            "verify_acc_after_mean": float(col("ver_acc_after").mean()),
            "verify_loss_before_mean": float(col("ver_loss_before").mean()),
            "verify_loss_after_mean": float(col("ver_loss_after").mean()),
            "select_acc_before_mean": float(col("sel_acc_before").mean()),
            "select_acc_after_mean": float(col("sel_acc_after").mean()),
        }
    none_after = {r["target"]: r["ver_acc_after"] for r in rows if r["branch"] == "none"}
    for name in ("kl_top1", "select_best"):
        gains = np.array([r["ver_acc_after"] - none_after[r["target"]] for r in rows if r["branch"] == name])
        summary["branches"][name].update({
            "gain_vs_none_after_mean_pp": float(gains.mean() * 100),
            "improved": int((gains > 1e-12).sum()),
            "degraded": int((gains < -1e-12).sum()),
            "flat": int((np.abs(gains) <= 1e-12).sum()),
        })
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
