"""Shared fixtures, adapters, and graph utilities for E2E MWC tests."""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import pytest

# Ensure project and reference modules are on sys.path
PROJECT_ROOT = "/scratch/hs9hd/mwc_certified_pruning"
REFERENCE_DIR = os.path.join(PROJECT_ROOT, "reference")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets", "realnets")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if REFERENCE_DIR not in sys.path:
    sys.path.insert(0, REFERENCE_DIR)

import reference.gen as gen  # noqa: E402
from reference.mwc import (  # noqa: E402
    INF,
    CertificationError,
    MWCResult,
    biconnected_components,
    check_graph,
    cycle_weight,
    is_simple_cycle,
    kappa_of,
    mwc,
    mwc_oracle,
    mwc_transversal,
    spanning_forest,
    two_core,
    _close,
    _index_map,
    _transversal_for,
    _truncated_dijkstra,
)

TOL = 1e-9


def approx_eq(a: float, b: float, tol: float = TOL) -> bool:
    """Floating-point approximate equality honoring infinity and relative tolerances."""
    if a == b:
        return True
    if math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def to_adj_and_multigraph_info(
    G: Any, weight_key: str = "weight", default_weight: float = 1.0
) -> Tuple[Dict[Any, Dict[Any, float]], Optional[Tuple[float, Tuple[Any, Any, Any]]], List[Tuple]]:
    """Convert a graph (nx.Graph, nx.MultiGraph, dict-of-dicts, or edge list)
    into a simple adjacency dictionary, while extracting any multigraph 2-cycles
    (Pass 1 of Proposition 1.2).

    Returns:
        (simple_adj, lightest_2cycle, all_edges)
        where lightest_2cycle is (w1 + w2, (u, v, u)) or None.
    """
    all_edges: List[Tuple[Any, Any, float]] = []
    nodes: Set[Any] = set()

    if isinstance(G, nx.MultiGraph):
        nodes.update(G.nodes())
        for u, v, k, d in G.edges(keys=True, data=True):
            w = float(d.get(weight_key, default_weight))
            all_edges.append((u, v, w))
    elif isinstance(G, nx.Graph):
        nodes.update(G.nodes())
        for u, v, d in G.edges(data=True):
            w = float(d.get(weight_key, default_weight))
            all_edges.append((u, v, w))
    elif isinstance(G, dict):
        nodes.update(G.keys())
        for u, nbrs in G.items():
            if isinstance(nbrs, dict):
                for v, w in nbrs.items():
                    nodes.add(v)
                    if str(u) <= str(v):
                        all_edges.append((u, v, float(w)))
            elif isinstance(nbrs, (list, tuple, set)):
                for v in nbrs:
                    nodes.add(v)
                    if str(u) <= str(v):
                        all_edges.append((u, v, float(default_weight)))
    elif isinstance(G, (list, tuple)):
        for item in G:
            if len(item) == 3:
                u, v, w = item
                all_edges.append((u, v, float(w)))
            elif len(item) == 2:
                u, v = item
                all_edges.append((u, v, float(default_weight)))
            nodes.add(u)
            nodes.add(v)
    else:
        raise TypeError(f"Unsupported graph input type: {type(G)}")

    pair_weights: Dict[Tuple[Any, Any], List[float]] = {}
    for u, v, w in all_edges:
        if u == v:
            continue
        pair = (u, v) if str(u) < str(v) else (v, u)
        pair_weights.setdefault(pair, []).append(w)

    lightest_2cycle: Optional[Tuple[float, Tuple[Any, Any, Any]]] = None
    simple_adj: Dict[Any, Dict[Any, float]] = {n: {} for n in nodes}

    for (u, v), weights in pair_weights.items():
        weights.sort()
        min_w = weights[0]
        simple_adj[u][v] = min_w
        simple_adj[v][u] = min_w
        if len(weights) >= 2:
            two_cyc_w = weights[0] + weights[1]
            if lightest_2cycle is None or two_cyc_w < lightest_2cycle[0]:
                lightest_2cycle = (two_cyc_w, (u, v, u))

    return simple_adj, lightest_2cycle, all_edges


def find_zero_weight_cycle(adj: Dict[Any, Dict[Any, float]]) -> Optional[Tuple[Any, ...]]:
    """Pass 2 of Proposition 1.2: DFS cycle check on zero-weight subgraph G[E_0].
    Returns simple zero-weight cycle as tuple of vertices if one exists, else None.
    """
    zero_adj: Dict[Any, List[Any]] = {u: [] for u in adj}
    for u, nbrs in adj.items():
        for v, w in nbrs.items():
            if w == 0.0:
                zero_adj[u].append(v)

    visited: Set[Any] = set()
    parent: Dict[Any, Any] = {}

    for root in zero_adj:
        if root in visited:
            continue
        stack = [(root, None)]
        while stack:
            u, p = stack.pop()
            if u not in visited:
                visited.add(u)
                parent[u] = p
                for v in zero_adj[u]:
                    if v == p:
                        continue
                    if v in visited:
                        cyc = [v, u]
                        cur = u
                        while cur != v and cur is not None:
                            cur = parent.get(cur)
                            if cur is not None and cur != v:
                                cyc.append(cur)
                        if len(cyc) >= 3:
                            return tuple(cyc)
                    else:
                        stack.append((v, u))
    return None


def is_acyclic_forest(adj: Dict[Any, Dict[Any, float]]) -> bool:
    """Pass 3: Spanning forest check. True if |E| == |V| - c (i.e. acyclic forest)."""
    n = len(adj)
    if n == 0:
        return True
    m = sum(len(nbrs) for nbrs in adj.values()) // 2
    seen = set()
    c = 0
    for node in adj:
        if node not in seen:
            c += 1
            seen.add(node)
            queue = [node]
            while queue:
                u = queue.pop()
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        queue.append(v)
    return m == (n - c)


def solve_mwc(
    G: Any,
    method: str = "certified_pruning",
    alpha: float = 0.0,
    beta: float = 0.0,
    K: float = 2.0,
    root_order: Optional[Sequence[Any]] = None,
    allow_zero_weights: bool = True,
    tol: float = TOL,
) -> Tuple[float, Optional[Tuple[Any, ...]]]:
    """Complete 3-Pass Degeneracy Preprocessing & MWC Certified Pruning Solver.

    Pass 1: Multigraph 2-cycles. If w1 + w2 == 0, halts immediately.
            Else sets Gamma_0 = w1 + w2.
    Pass 2: Zero-weight cycle check via DFS. If found, halts returning (0, C0).
    Pass 3: Spanning forest check. If acyclic, returns Gamma_0 (if 2-cycle exists) or (inf, None).
    Core:   Executes certified pruning or oracle on the simple graph with seeded Gamma_0.
    """
    simple_adj, lightest_2cycle, _ = to_adj_and_multigraph_info(G)

    # Pass 1: Lightest 2-cycle check
    gamma0 = INF
    cyc0 = None
    if lightest_2cycle is not None:
        gamma0, cyc0 = lightest_2cycle
        if gamma0 == 0.0:
            return 0.0, cyc0

    # Pass 2: Zero-weight DFS check
    zero_cyc = find_zero_weight_cycle(simple_adj)
    if zero_cyc is not None:
        return 0.0, zero_cyc

    # Pass 3: Spanning forest check
    if is_acyclic_forest(simple_adj):
        if gamma0 < INF:
            return gamma0, cyc0
        return INF, None

    # Core Execution
    if method == "exact_oracle":
        oracle_len, oracle_cyc = mwc_oracle(simple_adj)
        if gamma0 <= oracle_len and cyc0 is not None:
            return gamma0, cyc0
        return oracle_len, oracle_cyc
    elif method == "transversal":
        res = mwc_transversal(simple_adj, allow_zero_weights=allow_zero_weights, tol=tol)
        if gamma0 <= res.length and cyc0 is not None:
            return gamma0, cyc0
        return res.length, res.cycle
    else:  # default certified_pruning
        c0_to_pass = cyc0 if (cyc0 is not None and len(cyc0) >= 3 and is_simple_cycle(simple_adj, cyc0)) else None
        res = mwc(
            simple_adj,
            alpha=alpha,
            beta=beta,
            root_order=root_order,
            gamma0=gamma0 if gamma0 < INF else INF,
            cycle0=c0_to_pass,
            certify=True,
            collect_stats=True,
            allow_zero_weights=allow_zero_weights,
            tol=tol,
        )
        if gamma0 <= res.length and cyc0 is not None:
            return gamma0, cyc0
        return res.length, res.cycle


def assert_valid_cycle(
    G: Any, cycle: Optional[Sequence[Any]], expected_length: Optional[float] = None, tol: float = TOL
) -> None:
    """Opaque verification that cycle is simple (or 2-cycle in multigraph) and
    has the asserted weight.
    """
    if expected_length is not None and math.isinf(expected_length):
        assert cycle is None or len(cycle) == 0, f"Expected acyclic, got cycle {cycle!r}"
        return

    assert cycle is not None, "Expected finite cycle, got None"
    k = len(cycle)
    assert k >= 2, f"Cycle must have >= 2 vertices, got {cycle!r}"

    simple_adj, lightest_2cycle, all_edges = to_adj_and_multigraph_info(G)

    if k == 2 or (k == 3 and cycle[0] == cycle[-1]):
        u, v = cycle[0], cycle[1]
        pair_weights = [w for a, b, w in all_edges if (a == u and b == v) or (a == v and b == u)]
        assert len(pair_weights) >= 2, f"2-cycle requires multiple edges between {u} and {v}"
        pair_weights.sort()
        w_2cyc = pair_weights[0] + pair_weights[1]
        if expected_length is not None:
            assert approx_eq(w_2cyc, expected_length, tol), (
                f"2-cycle weight mismatch: {w_2cyc} vs expected {expected_length}"
            )
        return

    nodes_in_cycle = cycle[:-1] if cycle[0] == cycle[-1] else cycle
    assert len(set(nodes_in_cycle)) == len(nodes_in_cycle), f"Non-simple cycle: {cycle!r}"
    assert len(nodes_in_cycle) >= 3, f"Simple cycle requires >= 3 vertices: {cycle!r}"

    recomputed = 0.0
    k_cyc = len(nodes_in_cycle)
    for i in range(k_cyc):
        u = nodes_in_cycle[i]
        v = nodes_in_cycle[(i + 1) % k_cyc]
        assert v in simple_adj.get(u, {}), f"Edge ({u}, {v}) in cycle does not exist in graph"
        recomputed += simple_adj[u][v]

    if expected_length is not None:
        assert approx_eq(recomputed, expected_length, tol), (
            f"Cycle weight mismatch: recomputed {recomputed} vs expected {expected_length}"
        )


def assert_mwc_matches_oracle(G: Any, tol: float = TOL) -> Tuple[float, Optional[Tuple[Any, ...]]]:
    """Solves MWC via certified pruning and verifies exact match with exact oracle."""
    oracle_len, oracle_cyc = solve_mwc(G, method="exact_oracle", tol=tol)
    mwc_len, mwc_cyc = solve_mwc(G, method="certified_pruning", tol=tol)

    assert approx_eq(mwc_len, oracle_len, tol), (
        f"MWC certified pruning {mwc_len} != oracle {oracle_len}"
    )

    if math.isinf(oracle_len):
        assert mwc_cyc is None or len(mwc_cyc) == 0
    else:
        assert_valid_cycle(G, mwc_cyc, mwc_len, tol=tol)

    return mwc_len, mwc_cyc


def load_realnet_adj(name: str) -> Dict[int, Dict[int, float]]:
    """Load real benchmark network from .edges file into an adjacency dictionary."""
    path = os.path.join(DATASETS_DIR, f"{name}.edges")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Realnet dataset {path} not found")
    adj: Dict[int, Dict[int, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
            adj.setdefault(u, {})[v] = w
            adj.setdefault(v, {})[u] = w
    return adj


def load_realnet_graph(name: str) -> nx.Graph:
    """Load real benchmark network from .edges file into a NetworkX Graph."""
    adj = load_realnet_adj(name)
    G = nx.Graph()
    for u, nbrs in adj.items():
        for v, w in nbrs.items():
            if u < v:
                G.add_edge(u, v, weight=w)
    return G


def build_g3_gadget(alpha: float = 0.3, beta: float = 0.0, eps: float = 0.01) -> Dict[str, Dict[str, float]]:
    """Construct the 10-vertex degree-3 tightness gadget G_3."""
    import reference.tightness_witness as tw
    adj, _, _ = tw.build(alpha, beta, eps, hub=True)
    return adj
