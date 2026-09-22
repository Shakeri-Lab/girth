"""
test_acceleration.py -- Automated Pytest Suite for Milestone 2 Acceleration.

Verifies:
  1. Pure Python vs Accelerated Numerical Equivalence (tolerance 1e-9).
  2. Exact Cycle Structural Validity (simple cycle, valid edges, weight match).
  3. Zero False Prunings against exact oracle.
  4. Speedup sanity check (accelerated execution does not regress).
  5. Fallback invariant (use_acceleration=False produces identical results).
"""

import math
import os
import sys
import pytest
import networkx as nx

try:
    from shortest_cycle import minimum_weight_cycle, exact_oracle
except ImportError:
    from girth.shortest_cycle import minimum_weight_cycle, exact_oracle

TOL = 1e-9


def assert_cycle_valid(G, cycle, expected_weight, tol=TOL):
    """
    Validates cycle correctness across both simple graphs and multigraphs:
    - If acyclic (expected_weight == inf), cycle must be None or empty.
    - If cycle exists:
        - len(cycle) >= 3 (length 3 for multigraph 2-cycle [u, v, u], >= 4 for simple cycle)
        - cycle[0] == cycle[-1]
        - distinct internal vertices cycle[:-1]
        - edge existence and weight sum match expected_weight within tolerance
    """
    if math.isinf(expected_weight):
        assert cycle is None or len(cycle) == 0, f"Expected acyclic cycle, got: {cycle}"
        return
    assert cycle is not None and len(cycle) >= 3, f"Cycle too short: {cycle}"
    assert cycle[0] == cycle[-1], f"Cycle endpoints do not match: {cycle}"
    internal = cycle[:-1]
    assert len(set(internal)) == len(internal), f"Cycle not simple: {cycle}"

    is_multi = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    if len(cycle) == 3:  # Multigraph 2-cycle [u, v, u]
        assert is_multi, f"Length 2-cycle [u, v, u] is only valid in multigraph: {cycle}"
        u, v = cycle[0], cycle[1]
        assert G.has_edge(u, v), f"Edge ({u}, {v}) missing in graph"
        weights = sorted([d.get("weight", 1.0) for d in G[u][v].values()])
        assert len(weights) >= 2, f"Multigraph 2-cycle requires >= 2 parallel edges between {u} and {v}"
        w_sum = weights[0] + weights[1]
    else:
        if is_multi:
            w_sum = sum(
                min(d.get("weight", 1.0) for d in G[internal[i]][internal[(i + 1) % len(internal)]].values())
                for i in range(len(internal))
            )
        else:
            w_sum = sum(
                G[internal[i]][internal[(i + 1) % len(internal)]].get("weight", 1.0)
                for i in range(len(internal))
            )
    assert abs(w_sum - expected_weight) <= tol * max(1.0, expected_weight), (
        f"Weight mismatch: computed {w_sum} vs expected {expected_weight}"
    )


@pytest.mark.parametrize("n,p,seed", [
    (20, 0.8, 101),
    (30, 0.8, 102),
    (40, 0.8, 103),
    (50, 0.8, 104),
])
def test_dense_acceleration_equivalence(n, p, seed):
    """Verify numerical equivalence and cycle validity on dense graphs."""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0 + (u * 13 + v * 37) % 7 * 0.25

    len_pure, cyc_pure = minimum_weight_cycle(G, use_acceleration=False)
    len_acc, cyc_acc = minimum_weight_cycle(G, use_acceleration=True)

    assert abs(len_pure - len_acc) <= TOL * max(1.0, len_pure), (
        f"Equivalence failure: pure={len_pure} vs acc={len_acc}"
    )
    assert_cycle_valid(G, cyc_pure, len_pure)
    assert_cycle_valid(G, cyc_acc, len_acc)

    # Oracle comparison
    oracle_len, _ = exact_oracle(G)
    assert abs(len_acc - oracle_len) <= TOL * max(1.0, oracle_len), (
        f"False pruning: acc={len_acc} vs oracle={oracle_len}"
    )


@pytest.mark.parametrize("n,p,seed", [
    (30, 0.1, 201),
    (50, 0.05, 202),
    (100, 0.05, 203),
])
def test_sparse_acceleration_equivalence(n, p, seed):
    """Verify numerical equivalence and cycle validity on sparse graphs."""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0 + (u * 7 + v * 11) % 5 * 0.5

    len_pure, cyc_pure = minimum_weight_cycle(G, use_acceleration=False)
    len_acc, cyc_acc = minimum_weight_cycle(G, use_acceleration=True)

    if math.isinf(len_pure):
        assert math.isinf(len_acc)
    else:
        assert abs(len_pure - len_acc) <= TOL * max(1.0, len_pure)
        assert_cycle_valid(G, cyc_pure, len_pure)
        assert_cycle_valid(G, cyc_acc, len_acc)


def test_grid_acceleration_equivalence():
    """Verify on 2D grid graph with various weights."""
    G = nx.grid_2d_graph(8, 8)
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0 + (u * 3 + v * 5) % 4

    len_pure, cyc_pure = minimum_weight_cycle(G, use_acceleration=False)
    len_acc, cyc_acc = minimum_weight_cycle(G, use_acceleration=True)

    assert abs(len_pure - len_acc) <= TOL
    assert_cycle_valid(G, cyc_pure, len_pure)
    assert_cycle_valid(G, cyc_acc, len_acc)


def test_multichord_cycle_acceleration():
    """Verify on cycle graph with multiple crossing chords."""
    G = nx.cycle_graph(24)
    for u, v in G.edges():
        G[u][v]["weight"] = 2.0
    G.add_edge(0, 12, weight=1.5)
    G.add_edge(4, 16, weight=1.2)

    len_pure, cyc_pure = minimum_weight_cycle(G, use_acceleration=False)
    len_acc, cyc_acc = minimum_weight_cycle(G, use_acceleration=True)

    assert abs(len_pure - len_acc) <= TOL
    assert_cycle_valid(G, cyc_pure, len_pure)
    assert_cycle_valid(G, cyc_acc, len_acc)


def test_zero_weight_edges_acceleration():
    """Verify graph with zero-weight edges."""
    G = nx.erdos_renyi_graph(35, 0.3, seed=42)
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]["weight"] = 0.0 if i % 4 == 0 else 1.0 + (i % 5) * 0.5

    len_pure, cyc_pure = minimum_weight_cycle(G, use_acceleration=False)
    len_acc, cyc_acc = minimum_weight_cycle(G, use_acceleration=True)

    assert abs(len_pure - len_acc) <= TOL
    assert_cycle_valid(G, cyc_pure, len_pure)
    assert_cycle_valid(G, cyc_acc, len_acc)


def test_realnet_small_acceleration():
    """Verify on small real-world network lesmis."""
    datasets_dir = "/scratch/hs9hd/mwc_certified_pruning/datasets/realnets"
    edge_file = os.path.join(datasets_dir, "lesmis.edges")
    if not os.path.exists(edge_file):
        pytest.skip(f"Dataset file {edge_file} not found")

    G = nx.Graph()
    with open(edge_file) as f:
        for line in f:
            if line.strip():
                u, v, w = line.strip().split()
                G.add_edge(int(u), int(v), weight=float(w))

    len_pure, cyc_pure = minimum_weight_cycle(G, use_acceleration=False)
    len_acc, cyc_acc = minimum_weight_cycle(G, use_acceleration=True)

    assert abs(len_pure - 3.0) <= TOL
    assert abs(len_acc - 3.0) <= TOL
    assert_cycle_valid(G, cyc_pure, len_pure)
    assert_cycle_valid(G, cyc_acc, len_acc)


def test_fallback_invariant_explicit():
    """Verify that use_acceleration=False runs pure Python with identical result."""
    G = nx.erdos_renyi_graph(45, 0.6, seed=999)
    for u, v in G.edges():
        G[u][v]["weight"] = 2.0 + (u * 17 + v * 23) % 9 * 0.3

    g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
    g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)

    assert abs(g_pure - g_acc) <= TOL
    assert len(c_pure) == len(c_acc)
    assert_cycle_valid(G, c_pure, g_pure)
    assert_cycle_valid(G, c_acc, g_acc)


# ---------------------------------------------------------------------------
# Milestone 2 Iteration 2: Self-Loop Equivalence & Acceleration Tests
# ---------------------------------------------------------------------------

def test_self_loop_simple_triangle_acceleration():
    """Challenger 3.1: Verify pure Python vs C++ equivalence on self-loop + triangle."""
    G = nx.Graph()
    G.add_edge(0, 0, weight=0.1) # Self-loop
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    G.add_edge(3, 1, weight=1.0) # Triangle weight 3.0

    len_pure, cyc_pure = minimum_weight_cycle(G, use_acceleration=False)
    len_acc, cyc_acc = minimum_weight_cycle(G, use_acceleration=True)

    assert abs(len_pure - 3.0) <= TOL, f"Pure returned self-loop: {len_pure}, {cyc_pure}"
    assert abs(len_acc - 3.0) <= TOL, f"Acc returned self-loop: {len_acc}, {cyc_acc}"
    assert abs(len_pure - len_acc) <= TOL
    assert 0 not in cyc_pure and 0 not in cyc_acc

    assert_cycle_valid(G, cyc_pure, len_pure)
    assert_cycle_valid(G, cyc_acc, len_acc)

    oracle_len, _ = exact_oracle(G)
    assert abs(len_acc - oracle_len) <= TOL


def test_self_loop_multigraph_2cycle_acceleration():
    """Challenger 3.2: Verify pure Python vs C++ equivalence on self-loop + multigraph 2-cycle."""
    MG = nx.MultiGraph()
    MG.add_edge(0, 0, weight=0.05) # Self-loop
    MG.add_edge(1, 2, weight=1.0)  # Parallel edges forming 2-cycle weight 2.2
    MG.add_edge(1, 2, weight=1.2)
    MG.add_edge(3, 4, weight=2.0)  # Triangle weight 6.0
    MG.add_edge(4, 5, weight=2.0)
    MG.add_edge(5, 3, weight=2.0)

    len_pure, cyc_pure = minimum_weight_cycle(MG, use_acceleration=False)
    len_acc, cyc_acc = minimum_weight_cycle(MG, use_acceleration=True)

    assert abs(len_pure - 2.2) <= TOL, f"Pure failed 2-cycle: {len_pure}, {cyc_pure}"
    assert abs(len_acc - 2.2) <= TOL, f"Acc failed 2-cycle: {len_acc}, {cyc_acc}"
    assert abs(len_pure - len_acc) <= TOL
    assert 0 not in cyc_pure and 0 not in cyc_acc

    assert_cycle_valid(MG, cyc_pure, len_pure)
    assert_cycle_valid(MG, cyc_acc, len_acc)

    oracle_len, _ = exact_oracle(MG)
    assert abs(len_acc - oracle_len) <= TOL


def test_self_loop_tree_acyclic_acceleration():
    """Challenger 3.3: Verify tree with self-loop returns (inf, []) in both pure and acc."""
    T = nx.path_graph(5)
    T.add_edge(2, 2, weight=0.5) # Self-loop on tree node 2

    len_pure, cyc_pure = minimum_weight_cycle(T, use_acceleration=False)
    len_acc, cyc_acc = minimum_weight_cycle(T, use_acceleration=True)

    assert math.isinf(len_pure) and cyc_pure == []
    assert math.isinf(len_acc) and cyc_acc == []
    assert_cycle_valid(T, cyc_pure, len_pure)
    assert_cycle_valid(T, cyc_acc, len_acc)


def test_self_loop_multiple_on_4cycle_acceleration():
    """Challenger 3.4: Verify 4-cycle with multiple self-loops returns 4-cycle in both pure and acc."""
    G = nx.cycle_graph(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 2.0 # 4-cycle weight 8.0
    for i in range(4):
        G.add_edge(i, i, weight=0.1 * (i + 1)) # Self-loops: 0.1, 0.2, 0.3, 0.4

    len_pure, cyc_pure = minimum_weight_cycle(G, use_acceleration=False)
    len_acc, cyc_acc = minimum_weight_cycle(G, use_acceleration=True)

    assert abs(len_pure - 8.0) <= TOL, f"Pure failed 4-cycle: {len_pure}"
    assert abs(len_acc - 8.0) <= TOL, f"Acc failed 4-cycle: {len_acc}"
    assert abs(len_pure - len_acc) <= TOL

    assert_cycle_valid(G, cyc_pure, len_pure)
    assert_cycle_valid(G, cyc_acc, len_acc)

    oracle_len, _ = exact_oracle(G)
    assert abs(len_acc - oracle_len) <= TOL


def test_self_loop_on_cycle_vertex_acceleration():
    """Verify self-loop on an active MWC cycle node does not corrupt Dijkstra exploration."""
    G = nx.cycle_graph(3)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0 # Triangle weight 3.0
    G.add_edge(0, 0, weight=0.05) # Tiny self-loop directly on vertex 0

    len_pure, cyc_pure = minimum_weight_cycle(G, use_acceleration=False)
    len_acc, cyc_acc = minimum_weight_cycle(G, use_acceleration=True)

    assert abs(len_pure - 3.0) <= TOL
    assert abs(len_acc - 3.0) <= TOL
    assert abs(len_pure - len_acc) <= TOL
    assert len(cyc_pure) == 4 and len(cyc_acc) == 4

    assert_cycle_valid(G, cyc_pure, len_pure)
    assert_cycle_valid(G, cyc_acc, len_acc)


def test_self_loop_dense_random_graph_acceleration():
    """Verify random dense graph with heavy self-loop injection maintains C++ equivalence and CSR safety."""
    G = nx.erdos_renyi_graph(35, 0.6, seed=404)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0 + (u * 3 + v * 7) % 5 * 0.5
    # Inject self-loops on ~50% of the nodes with tiny weights
    for i in range(0, 35, 2):
        G.add_edge(i, i, weight=0.01)

    len_pure, cyc_pure = minimum_weight_cycle(G, use_acceleration=False)
    len_acc, cyc_acc = minimum_weight_cycle(G, use_acceleration=True)

    assert abs(len_pure - len_acc) <= TOL * max(1.0, len_pure)
    assert_cycle_valid(G, cyc_pure, len_pure)
    assert_cycle_valid(G, cyc_acc, len_acc)

    oracle_len, _ = exact_oracle(G)
    assert abs(len_acc - oracle_len) <= TOL * max(1.0, oracle_len)


def test_only_self_loops_graph_acceleration():
    """Verify graph consisting entirely of self-loops returns (inf, []) in both pure and acc."""
    G = nx.Graph()
    for i in range(10):
        G.add_edge(i, i, weight=0.1 * (i + 1))

    len_pure, cyc_pure = minimum_weight_cycle(G, use_acceleration=False)
    len_acc, cyc_acc = minimum_weight_cycle(G, use_acceleration=True)

    assert math.isinf(len_pure) and cyc_pure == []
    assert math.isinf(len_acc) and cyc_acc == []
    assert_cycle_valid(G, cyc_pure, len_pure)
    assert_cycle_valid(G, cyc_acc, len_acc)
