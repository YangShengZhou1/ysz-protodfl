"""Correct hop-set semantics for the multi-hop search experiments.

The research guidance defines the search radius as: all nodes whose shortest-path
distance from the source on the base graph is at most ``k`` hops, excluding the
source itself. This module keeps that definition separate from the legacy
``get_k_hop_neighbors`` used inside the training code, so radius experiments can
be designed and validated independently before changing the trainer.
"""

from collections import deque

import networkx as nx


def _shortest_path_distances(graph, source):
    if not nx.is_connected(graph):
        raise ValueError("hop sets require a connected base graph")
    if source not in graph:
        raise KeyError(f"source {source} not in graph")
    lengths = nx.single_source_shortest_path_length(graph, source)
    return lengths


def neighbors_exact_k(graph, source, k):
    """Nodes at shortest-path distance exactly ``k`` from ``source``."""
    lengths = _shortest_path_distances(graph, source)
    return {node for node, dist in lengths.items() if dist == k}


def neighbors_within_k(graph, source, k):
    """Nodes at shortest-path distance in ``[1, k]`` from ``source``."""
    lengths = _shortest_path_distances(graph, source)
    return {node for node, dist in lengths.items() if 0 < dist <= k}


def layer_summary(graph, source, kmax):
    """Per-layer candidate counts and a forwarding-cost proxy.

    The proxy ``reach_cost`` sums, over all nodes within radius ``k``, the
    shortest-path hop count needed to reach them. It is a topology-only proxy
    for discovery/forwarding cost, not a byte measurement.
    """
    lengths = _shortest_path_distances(graph, source)
    within = set()
    layers = []
    for k in range(1, kmax + 1):
        exact = neighbors_exact_k(graph, source, k)
        within.update(exact)
        layers.append({
            "k": k,
            "exact_count": len(exact),
            "within_count": len(within),
            "reach_cost_proxy": sum(lengths[node] for node in within),
        })
    return layers
