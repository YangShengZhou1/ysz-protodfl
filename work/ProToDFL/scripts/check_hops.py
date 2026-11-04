import argparse
import json
import os
import sys
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hop import layer_summary, neighbors_exact_k, neighbors_within_k
from fedml_api.standalone.ProToDFL.protoDFL_api import ProToDFLAPI


def _api_with_graph(graph):
    api = ProToDFLAPI.__new__(ProToDFLAPI)
    api.graph = graph
    return api


def verify_correctness():
    checks = []
    for name, graph in [("path5", nx.path_graph(5)), ("ring5", nx.cycle_graph(5))]:
        lengths = dict(nx.all_pairs_shortest_path_length(graph))
        for source in graph.nodes:
            for k in range(1, 5):
                exact_expected = {node for node in graph.nodes if lengths[source][node] == k}
                within_expected = {node for node in graph.nodes if 0 < lengths[source][node] <= k}
                assert neighbors_exact_k(graph, source, k) == exact_expected
                assert neighbors_within_k(graph, source, k) == within_expected
                checks.append({"graph": name, "source": int(source), "k": k, "ok": True})
    return checks


def trainer_mismatch_report(graph, graph_name, kmax):
    api = _api_with_graph(graph)
    rows = []
    for source in graph.nodes:
        for k in range(1, kmax + 1):
            trainer = set(api.get_k_hop_neighbors(source, k))
            exact = neighbors_exact_k(graph, source, k)
            within = neighbors_within_k(graph, source, k)
            rows.append({
                "source": int(source),
                "k": k,
                "trainer": sorted(map(int, trainer)),
                "exact": sorted(map(int, exact)),
                "within": sorted(map(int, within)),
                "trainer_equals_exact": trainer == exact,
                "trainer_equals_within": trainer == within,
            })
    return {"graph": graph_name, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ring_nodes", type=int, default=20)
    parser.add_argument("--kmax", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ring = nx.cycle_graph(args.ring_nodes)
    source = 0
    report = {
        "ring_nodes": args.ring_nodes,
        "kmax": args.kmax,
        "correctness": verify_correctness(),
        "layer_summary_source0": layer_summary(ring, source, args.kmax),
        "trainer_ring": trainer_mismatch_report(ring, "ring", args.kmax),
    }
    path = report["trainer_mismatch_report"] = trainer_mismatch_report(nx.path_graph(5), "path5", 4)

    report_path = output_dir / "hop_semantics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "layer_summary": report["layer_summary_source0"],
        "trainer_mismatch_counts": {
            item["graph"]: sum(1 for row in item["rows"] if not row["trainer_equals_within"])
            for item in [report["trainer_ring"], path]
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
