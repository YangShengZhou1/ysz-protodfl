import argparse
import json
import logging
import os
import platform
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torchvision.datasets import CIFAR10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.standalone.ProToDFL.protoDFL_api import ProToDFLAPI


def check_multihop(graph, graph_name, max_k=4):
    api = ProToDFLAPI.__new__(ProToDFLAPI)
    api.graph = graph
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    rows = []
    for source in graph.nodes:
        for k in range(1, max_k + 1):
            actual = set(api.get_k_hop_neighbors(source, k))
            exact = {node for node in graph.nodes if distances[source][node] == k}
            within = {node for node in graph.nodes if 0 < distances[source][node] <= k}
            rows.append({
                "source": int(source),
                "k": k,
                "actual": sorted(map(int, actual)),
                "exact_k_hop": sorted(map(int, exact)),
                "within_k_hop": sorted(map(int, within)),
                "matches_exact": actual == exact,
                "matches_within": actual == within,
            })
    return {
        "graph": graph_name,
        "edges": sorted(tuple(sorted(map(int, edge))) for edge in graph.edges),
        "rows": rows,
    }


def check_cifar10_partition(data_dir, output_dir, client_num, alpha, batch_size, seed, logger):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = CIFAR10(str(data_dir), train=True, download=False)
    test_dataset = CIFAR10(str(data_dir), train=False, download=False)
    loaded = load_partition_data_cifar10(
        str(data_dir), "dir", alpha, client_num, batch_size, logger
    )
    train_num_dict, train_dict, test_dict, class_counts = loaded[4], loaded[5], loaded[6], loaded[7]

    train_indices = {
        client: np.asarray(train_dict[client].dataset.dataidxs, dtype=np.int64)
        for client in range(client_num)
    }
    test_indices = {
        client: np.asarray(test_dict[client].dataset.dataidxs, dtype=np.int64)
        for client in range(client_num)
    }
    all_train = np.concatenate(list(train_indices.values()))
    all_test = np.concatenate(list(test_indices.values()))
    client_rows = []
    for client in range(client_num):
        counts = np.asarray(class_counts[client], dtype=np.int64)
        client_rows.append({
            "client": client,
            "train_samples": int(train_num_dict[client]),
            "test_samples": int(len(test_indices[client])),
            "train_class_counts": counts.tolist(),
            "train_class_proportions": (counts / counts.sum()).round(6).tolist(),
        })

    partition_path = output_dir / "cifar10_dir_partition.npz"
    np.savez_compressed(
        partition_path,
        **{f"train_{client}": indices for client, indices in train_indices.items()},
        **{f"test_{client}": indices for client, indices in test_indices.items()},
    )
    return {
        "seed": seed,
        "partition": "dir",
        "alpha": alpha,
        "client_num": client_num,
        "official_train_samples": len(train_dataset),
        "official_test_samples": len(test_dataset),
        "assigned_train_samples": int(len(all_train)),
        "unique_assigned_train_samples": int(len(np.unique(all_train))),
        "duplicate_train_assignments": int(len(all_train) - len(np.unique(all_train))),
        "unassigned_train_samples": int(len(train_dataset) - len(np.unique(all_train))),
        "assigned_test_samples_with_overlap": int(len(all_test)),
        "unique_assigned_test_samples": int(len(np.unique(all_test))),
        "train_test_pools": "CIFAR-10 official disjoint train/test pools; indices belong to separate pools",
        "clients": client_rows,
        "partition_file": str(partition_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--client_num", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("startup_checks")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output_dir / "startup_checks.log", mode="w", encoding="utf-8")
    logger.addHandler(handler)

    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "multihop": [
            check_multihop(nx.path_graph(5), "five_node_chain"),
            check_multihop(nx.cycle_graph(5), "five_node_ring"),
        ],
        "cifar10": check_cifar10_partition(
            Path(args.data_dir).resolve(), output_dir, args.client_num,
            args.alpha, args.batch_size, args.seed, logger
        ),
    }
    report_path = output_dir / "startup_checks.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "within_k_mismatches": {
            item["graph"]: [
                {"source": row["source"], "k": row["k"], "actual": row["actual"],
                 "within_k_hop": row["within_k_hop"]}
                for row in item["rows"] if not row["matches_within"]
            ]
            for item in report["multihop"]
        },
        "cifar10_summary": {k: v for k, v in report["cifar10"].items() if k != "clients"},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
