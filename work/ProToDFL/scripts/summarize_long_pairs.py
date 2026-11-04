"""Summarize paired extended runs from downloaded server logs."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

PATTERN = re.compile(
    r"\{'global_test_acc': ([0-9.]+), 'global_test_loss': ([0-9.]+)\}"
)


def parse_one(log_dir, tag):
    logs = sorted(log_dir.glob(f"*{tag}.log"))
    comms = sorted(log_dir.glob(f"*{tag}_comm.json"))
    if not logs:
        return None
    accs, losses = [], []
    for line in logs[0].read_text(encoding="utf-8", errors="ignore").splitlines():
        match = PATTERN.search(line)
        if match:
            accs.append(float(match.group(1)))
            losses.append(float(match.group(2)))
    comm = {}
    if comms:
        data = json.loads(comms[0].read_text(encoding="utf-8"))
        comm = {
            "model_param_count": data.get("model_param_count"),
            "model_bytes_per_client": data.get("model_bytes_per_client"),
            "proto_bytes_per_client": data.get("proto_bytes_per_client"),
            "total_model_bytes": data.get("total_model_bytes"),
            "total_proto_bytes": data.get("total_proto_bytes"),
            "total_selected_directed_links": data.get("total_selected_directed_links"),
            "total_proto_accesses": data.get("total_proto_accesses"),
            "initial_model_sha256": data.get("initial_model_sha256"),
            "data_split_sha256": data.get("data_split_sha256"),
        }
    return {
        "tag": tag,
        "rounds": len(accs),
        "acc": accs,
        "loss": losses,
        "final_acc": accs[-1] if accs else None,
        "final_loss": losses[-1] if losses else None,
        **comm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    rows = []
    for seed in range(3):
        for scope in ("all", "base"):
            tag = f"srv20-e5-r10-{scope}-s{seed}"
            item = parse_one(log_dir, tag)
            if item:
                rows.append(item)

    summary = {"rows": rows, "paired_final": []}
    for seed in range(3):
        all_row = next((r for r in rows if r["tag"].endswith(f"all-s{seed}")), None)
        base_row = next((r for r in rows if r["tag"].endswith(f"base-s{seed}")), None)
        if all_row and base_row and all_row["final_acc"] is not None and base_row["final_acc"] is not None:
            summary["paired_final"].append({
                "seed": seed,
                "all_final": all_row["final_acc"],
                "base_final": base_row["final_acc"],
                "base_minus_all": base_row["final_acc"] - all_row["final_acc"],
            })
    if summary["paired_final"]:
        diffs = [x["base_minus_all"] for x in summary["paired_final"]]
        summary["mean_base_minus_all"] = statistics.mean(diffs)
        summary["stdev_base_minus_all"] = statistics.stdev(diffs) if len(diffs) > 1 else 0.0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if rows:
        fields = [
            "tag", "rounds", "final_acc", "final_loss", "total_model_bytes",
            "total_proto_bytes", "total_selected_directed_links",
            "total_proto_accesses", "initial_model_sha256", "data_split_sha256",
        ]
        with open(output.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print("rows", len(rows))


if __name__ == "__main__":
    main()
