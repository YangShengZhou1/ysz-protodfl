"""Compare validation true-class counts with saved prediction diversity."""

import argparse
import json
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10

TAGS = [
    "srv20-e5-all-s0", "srv20-e5-base-s0",
    "srv20-e5-all-s1", "srv20-e5-base-s1",
    "srv20-e5-all-s2", "srv20-e5-base-s2",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    args = parser.parse_args()

    root = Path(args.project)
    data = CIFAR10(str(root / "datasets/cifar10"), train=True, download=False)
    labels = np.asarray(getattr(data, "targets", getattr(data, "target", None)))
    checks = root / "outputs/checks/server-paired"

    for tag in TAGS:
        ckpt_meta = next((root / "outputs/checkpoints").glob(f"*{tag}/round_1/metadata.json"))
        health_path = checks / f"{tag}.json"
        if not ckpt_meta.exists() or not health_path.exists():
            print(tag, "MISSING")
            continue
        meta = json.loads(ckpt_meta.read_text(encoding="utf-8"))
        health = json.loads(health_path.read_text(encoding="utf-8"))
        pred_unique = {
            int(row["client"]): int(row["pre_unique_predictions"])
            for row in health["rows"]
        }
        single_true = 0
        degenerate = []
        for client in meta["clients"]:
            idx = int(client["client_idx"])
            true_classes = int(np.unique(labels[np.asarray(client["val_indices"], dtype=np.int64)]).size)
            threshold = min(2, true_classes)
            if true_classes == 1:
                single_true += 1
            if pred_unique[idx] < threshold:
                degenerate.append((idx, true_classes, pred_unique[idx]))
        print(tag, "single_true_clients", single_true, "degenerate", degenerate)


if __name__ == "__main__":
    main()
