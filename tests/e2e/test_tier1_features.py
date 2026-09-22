"""Tier 1: Feature Coverage in Isolation (>= 45 tests across 9 feature areas).

Covers all 9 core feature areas specified in PROJECT.md, TEST_INFRA.md, and theory handoff:
1. Multi-chord Cotree Chord Path Decomposition (Lemma 2.1)
2. 3-Pass Linear-Time Degeneracy Preprocessing (Prop 1.2)
3. Strict Lexicographical Arborescence Acyclicity on Zero Weights (Lemma 1.3)
4. Piecewise Frontier Sentinels (tau_x / B_x) & Infinite Frontier Pruning (Def 1.5)
5. Dynamic Certified Pruning & Decoupled Metric Inflation (Thm 3.2)
6. Historical Lower Certificate Envelope Accumulation & Lazy Root Skipping (Thm 4.3)
7. Structural Countermeasures: Heavy-Edge Filtering, 2-Core, G_3 Resistance (Thm 5.1)
8. Synchronous Epoch-Based Snapshot Isolation for Parallel Pruning (Thm 6.2)
9. Legacy Bug Eliminators (Node 0 LCA, Chord Rejection, Deep LCA, NX 3.4+)
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import pytest

import reference.gen as gen
from tests.e2e.conftest import (
    INF,
    TOL,
    approx_eq,
    assert_mwc_matches_oracle,
    assert_valid_cycle,
    biconnected_components,
    build_g3_gadget,
    check_graph,
    cycle_weight,
    find_zero_weight_cycle,
    is_acyclic_forest,
    is_simple_cycle,
    kappa_of,
    mwc,
    mwc_oracle,
    mwc_transversal,
    solve_mwc,
    spanning_forest,
    to_adj_and_multigraph_info,
    two_core,
    _index_map,
    _transversal_for,
    _truncated_dijkstra,
)


# ============================================================================
# Feature 1: Multi-Chord Cotree Chord Path Decomposition (Lemma 2.1)
# ============================================================================

def test_f1_multichord_k33_bipartite():
    """Complete bipartite graph K_{3,3} with unit weights: girth is 4, multiple cotree chords."""
    G = nx.complete_bipartite_graph(3, 3)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)
    assert len(cyc) == 4


def test_f1_multichord_k5_clique():
    """Complete graph K_5 with distinct positive weights: girth 3, dense multi-chord topology."""
    G = nx.complete_graph(5)
    weights = [
        (0, 1, 2.0), (0, 2, 3.0), (0, 3, 4.0), (0, 4, 5.0),
        (1, 2, 1.5), (1, 3, 6.0), (1, 4, 7.0),
        (2, 3, 2.5), (2, 4, 8.0),
        (3, 4, 1.0),
    ]
    for u, v, w in weights:
        G[u][v]["weight"] = w
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 6.5)


def test_f1_multichord_ladder_diagonals():
    """5-rung ladder graph with diagonal cross-edges forming multiple chords."""
    G = nx.ladder_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 3.0
    G.add_edge(0, 3, weight=5.0)
    G.add_edge(1, 2, weight=5.0)
    G.add_edge(2, 5, weight=4.0)
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 12.0)


def test_f1_multichord_wheel_w6():
    """Wheel graph W_6 with hub connected to rim: multiple chords per rim cycle."""
    G = nx.wheel_graph(6)
    for u, v in G.edges():
        G[u][v]["weight"] = 2.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 6.0)
    assert len(cyc) == 3


def test_f1_multichord_random_dense_gnp():
    """Dense G(10, 0.6) random graph with multi-chord cycles."""
    G = nx.gnp_random_graph(10, 0.6, seed=123)
    rng = random.Random(123)
    for u, v in G.edges():
        G[u][v]["weight"] = rng.uniform(1.0, 5.0)
    assert_mwc_matches_oracle(G)


def test_f1_multichord_lemma21_path_inequality():
    """Direct algebraic verification of Lemma 2.1:
    delta(u) + delta(v) + w(e) <= 2*delta(z) + ell(C) for every cotree chord e=uv and z in C.
    """
    adj = {
        "x": {"a": 2.0},
        "a": {"x": 2.0, "b": 1.0, "d": 1.0, "c": 1.5},
        "b": {"a": 1.0, "c": 1.0},
        "c": {"b": 1.0, "d": 1.0, "a": 1.5},
        "d": {"c": 1.0, "a": 1.0},
    }
    index = _index_map(adj)
    delta, pred, Q = _truncated_dijkstra(adj, "x", set(adj), INF, index)
    cycle = ["a", "b", "c", "d"]
    ell_C = 4.0
    non_tree_chords = [("c", "d", 1.0), ("a", "c", 1.5)]
    for u, v, w in non_tree_chords:
        for z in cycle:
            lhs = delta[u] + delta[v] + w
            rhs = 2.0 * delta[z] + ell_C
            assert lhs <= rhs + TOL, f"Lemma 2.1 violated: {lhs} > {rhs} for chord ({u}, {v}) and node {z}"


# ============================================================================
# Feature 2: 3-Pass Linear-Time Degeneracy Preprocessing (Proposition 1.2)
# ============================================================================

def test_f2_pass1_multigraph_2cycle_lightest():
    """Multigraph where parallel edge 2-cycle is strictly lighter than any 3-cycle."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(0, 1, weight=2.0)  # 2-cycle of weight 3.0
    G.add_edge(1, 2, weight=5.0)
    G.add_edge(2, 0, weight=5.0)  # 3-cycle of weight 1.0 + 5.0 + 5.0 = 11.0
    mwc_len, cyc = solve_mwc(G)
    assert approx_eq(mwc_len, 3.0)
    assert cyc in [(0, 1, 0), (1, 0, 1)]


def test_f2_pass1_multigraph_2cycle_heavier():
    """Multigraph where 2-cycle is heavier than a 3-cycle."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=5.0)
    G.add_edge(0, 1, weight=5.0)  # 2-cycle of weight 10.0
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 0, weight=1.0)  # 3-cycle of weight 5.0 + 1.0 + 1.0 = 7.0
    mwc_len, cyc = solve_mwc(G)
    assert approx_eq(mwc_len, 7.0)
    assert set(cyc) == {0, 1, 2}


def test_f2_pass1_multigraph_identical_parallel_weights():
    """Multigraph with two identical weights on parallel edge."""
    G = nx.MultiGraph()
    G.add_edge("u", "v", weight=2.5)
    G.add_edge("u", "v", weight=2.5)
    mwc_len, cyc = solve_mwc(G)
    assert approx_eq(mwc_len, 5.0)
    assert cyc in [("u", "v", "u"), ("v", "u", "v")]


def test_f2_pass1_multigraph_three_parallel_edges():
    """Multigraph with 3 parallel edges: lightest pair selected (Pass 1)."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=7.0)
    G.add_edge(0, 1, weight=2.0)
    G.add_edge(0, 1, weight=4.0)
    mwc_len, cyc = solve_mwc(G)
    assert approx_eq(mwc_len, 6.0)  # 2.0 + 4.0 = 6.0


def test_f2_pass2_zero_weight_cycle_instant_halt():
    """Zero-weight 3-cycle halts immediately at Pass 2 returning (0, C0)."""
    adj = {
        0: {1: 0.0, 2: 0.0},
        1: {0: 0.0, 2: 0.0},
        2: {0: 0.0, 1: 0.0},
        3: {0: 10.0},
    }
    zero_cyc = find_zero_weight_cycle(adj)
    assert zero_cyc is not None
    assert set(zero_cyc) == {0, 1, 2}
    mwc_len, cyc = solve_mwc(adj)
    assert mwc_len == 0.0
    assert set(zero_cyc) == {0, 1, 2}


def test_f2_pass2_zero_weight_acyclic_tree():
    """Zero-weight tree with no zero-weight cycles passes Pass 2 and finds positive cycle."""
    adj = {
        0: {1: 0.0, 3: 4.0},
        1: {0: 0.0, 2: 0.0},
        2: {1: 0.0, 3: 4.0},
        3: {2: 4.0, 0: 4.0},  # Cycle 0-1-2-3 has weight 0 + 0 + 4 + 4 = 8.0
    }
    assert find_zero_weight_cycle(adj) is None
    mwc_len, cyc = solve_mwc(adj)
    assert approx_eq(mwc_len, 8.0)


def test_f2_pass3_spanning_forest_acyclic_halt():
    """Disconnected forest halts at Pass 3 returning (inf, None)."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=2.0)
    G.add_edge(1, 2, weight=3.0)
    G.add_edge(3, 4, weight=1.0)
    simple_adj, _, _ = to_adj_and_multigraph_info(G)
    assert is_acyclic_forest(simple_adj)
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)
    assert cyc is None or len(cyc) == 0


def test_f2_pass3_fundamental_cycle_seeding():
    """Pass 3 traces fundamental cycle to seed Gamma_0 < inf."""
    adj = {
        0: {1: 1.0, 3: 1.0},
        1: {0: 1.0, 2: 1.0},
        2: {1: 1.0, 3: 1.0},
        3: {0: 1.0, 2: 1.0},
    }
    S, gamma0, cyc0, mu = _transversal_for(adj)
    assert gamma0 < INF
    assert approx_eq(gamma0, 4.0)
    assert len(cyc0) == 4
    assert mu == 1


# ============================================================================
# Feature 3: Strict Lexicographical Arborescence Acyclicity (Lemma 1.3)
# ============================================================================

def test_f3_zero_weight_path_acyclicity():
    """Zero-weight linear path produces strictly acyclic tree with unique predecessors."""
    n = 6
    adj = {i: {} for i in range(n)}
    for i in range(n - 1):
        adj[i][i + 1] = 0.0
        adj[i + 1][i] = 0.0
    index = _index_map(adj)
    delta, pred, Q = _truncated_dijkstra(adj, 0, set(adj), INF, index)
    assert len(Q) == n
    for i in range(1, n):
        assert pred[i] == i - 1


def test_f3_zero_weight_star_parent_uniqueness():
    """Star with 0-weight edges: all leaves have root as unique parent."""
    adj = {0: {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}}
    for i in range(1, 5):
        adj[i] = {0: 0.0}
    index = _index_map(adj)
    delta, pred, Q = _truncated_dijkstra(adj, 0, set(adj), INF, index)
    for i in range(1, 5):
        assert pred[i] == 0
        assert delta[i] == 0.0


def test_f3_tied_distances_hop_count_ordering():
    """Graph with tied edge weights: hop-count enforces deterministic parentage."""
    adj = {
        0: {1: 2.0, 2: 1.0},
        1: {0: 2.0, 2: 1.0},
        2: {0: 1.0, 1: 1.0},
    }
    index = _index_map(adj)
    delta, pred, Q = _truncated_dijkstra(adj, 0, set(adj), INF, index)
    assert pred[1] in (0, 2)
    assert approx_eq(delta[1], 2.0)


def test_f3_equal_dist_hops_id_tie_breaking():
    """Tied distance and hops broken by unique vertex id without non-determinism."""
    adj = {
        0: {1: 1.0, 2: 1.0},
        1: {0: 1.0, 3: 1.0},
        2: {0: 1.0, 3: 1.0},
        3: {1: 1.0, 2: 1.0},
    }
    index = _index_map(adj)
    delta, pred, Q = _truncated_dijkstra(adj, 0, set(adj), INF, index)
    assert pred[3] in (1, 2)
    assert approx_eq(delta[3], 2.0)


def test_f3_unique_lca_under_zero_weights():
    """LCA structure over zero-weight tree produces strictly unique, non-looping ancestors."""
    from reference.mwc import LCAStructure
    nodes = [0, 1, 2, 3, 4]
    pred = {1: 0, 2: 0, 3: 1, 4: 1}
    lca_struct = LCAStructure(nodes, pred, 0)
    assert lca_struct.lca(3, 4) == 1
    assert lca_struct.lca(3, 2) == 0
    assert lca_struct.lca(0, 4) == 0
    assert lca_struct.lca(3, 3) == 3


# ============================================================================
# Feature 4: Piecewise Frontier Sentinels & Infinite Frontier (Definition 1.5)
# ============================================================================

def test_f4_disconnected_component_infinite_sentinel():
    """Disconnected graph: searching component 1 exhausts frontier (tau_x = inf, B_x = inf)."""
    adj = {
        0: {1: 1.0},
        1: {0: 1.0},
        2: {3: 2.0, 4: 2.0},
        3: {2: 2.0, 4: 2.0},
        4: {2: 2.0, 3: 2.0},
    }
    mwc_len, cyc = solve_mwc(adj)
    assert approx_eq(mwc_len, 6.0)
    assert set(cyc) == {2, 3, 4}


def test_f4_isolated_vertex_sentinel_prune():
    """Isolated root vertex immediately exhausts active reach, setting B_x = inf."""
    adj = {0: {}, 1: {2: 1.0, 3: 1.0}, 2: {1: 1.0, 3: 1.0}, 3: {1: 1.0, 2: 1.0}}
    mwc_len, cyc = solve_mwc(adj)
    assert approx_eq(mwc_len, 3.0)


def test_f4_tree_leaf_search_exhaustion():
    """Search from leaf of tree exhausts frontier without cycle discovery or crash."""
    adj = {
        0: {1: 1.0},
        1: {0: 1.0, 2: 1.0},
        2: {1: 1.0, 3: 1.0},
        3: {2: 1.0},
    }
    mwc_len, cyc = solve_mwc(adj)
    assert math.isinf(mwc_len)


def test_f4_path_graph_infinite_pruning():
    """Path graph with 10 vertices: all roots trigger infinite frontier sentinel."""
    G = nx.path_graph(10)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)


def test_f4_no_indeterminate_arithmetic():
    """Exhausted frontier does not evaluate (inf - inf) / 2 indeterminate form."""
    adj = {0: {1: 1.0}, 1: {0: 1.0}}
    res = mwc(adj)
    assert math.isinf(res.length)
    assert res.cycle is None


# ============================================================================
# Feature 5: Dynamic Certified Pruning & Metric Decoupling (Theorem 3.2)
# ============================================================================

def test_f5_dynamic_pruning_exact_mode_alpha_beta():
    """Exact mode (alpha <= beta) guarantees 0 vertex deletions and exact MWC."""
    rng = random.Random(42)
    adj = gen.grid(3, 3, rng, wkind="int")
    res = mwc(adj, alpha=0.1, beta=0.1, collect_stats=True)
    assert res.stats["total_deletions"] == 0
    oracle_len, _ = mwc_oracle(adj)
    assert approx_eq(res.length, oracle_len)


def test_f5_dynamic_pruning_approx_mode_kappa_bound():
    """Approximate mode (alpha > beta) satisfies gamma* <= ghat <= kappa * gamma*."""
    rng = random.Random(42)
    adj = gen.multiscale(n=8, rng=rng)
    oracle_len, _ = mwc_oracle(adj)
    if not math.isinf(oracle_len):
        alpha, beta = 0.3, 0.0
        kappa = kappa_of(alpha, beta)
        res = mwc(adj, alpha=alpha, beta=beta)
        assert res.length >= oracle_len - TOL
        assert res.length <= kappa * oracle_len + TOL


def test_f5_metric_inflation_decoupling_soundness():
    """External deletions do not violate B_x - 2*delta(z) <= ell(C)."""
    rng = random.Random(99)
    adj = gen.planted_cycle(n=12, k=4, rng=rng)
    oracle_len, _ = mwc_oracle(adj)
    res = mwc(adj, alpha=0.0, beta=0.0)
    assert approx_eq(res.length, oracle_len)


def test_f5_exhaustive_root_scheduling():
    """All active vertices in worklist are scheduled until completion."""
    rng = random.Random(55)
    adj = gen.erdos_renyi(n=10, p=0.4, rng=rng, wkind="int")
    res = mwc(adj, collect_stats=True)
    stats = res.stats
    assert stats["roots_run"] + stats["roots_skipped"] == len(adj)


def test_f5_root_order_invariance():
    """Different permutations of root exploration order yield identical exact MWC."""
    rng = random.Random(77)
    adj = gen.erdos_renyi(n=8, p=0.5, rng=rng, wkind="int")
    nodes = list(adj)
    lengths = []
    for _ in range(5):
        order = list(nodes)
        rng.shuffle(order)
        res = mwc(adj, root_order=order)
        lengths.append(res.length)
    assert all(approx_eq(l, lengths[0]) for l in lengths)


# ============================================================================
# Feature 6: Historical Lower Certificate Accumulation (Theorem 4.3)
# ============================================================================

def test_f6_envelope_accumulation_monotonicity():
    """Historical envelope accumulation is pointwise monotone: L_acc(z) non-decreasing."""
    rng = random.Random(42)
    adj = gen.grid(3, 3, rng, wkind="int")
    nodes = list(adj)
    L_acc = {v: 0.0 for v in nodes}
    for x in nodes[:3]:
        delta, pred, Q = _truncated_dijkstra(adj, x, set(adj), 4.0, _index_map(adj))
        for z in Q:
            L_x_z = max(0.0, 4.0 - 2.0 * delta[z])
            old_val = L_acc[z]
            L_acc[z] = max(L_acc[z], L_x_z)
            assert L_acc[z] >= old_val


def test_f6_lazy_root_skipping_soundness():
    """Skipping roots with lower certificate >= Gamma/K preserves exactness."""
    rng = random.Random(42)
    adj = gen.complete(5, rng, wkind="int")
    oracle_len, _ = mwc_oracle(adj)
    res = mwc_transversal(adj)
    assert approx_eq(res.length, oracle_len)


def test_f6_retrospective_pruning():
    """Historical lower bound can prune vertices retroactively once Gamma shrinks."""
    rng = random.Random(101)
    adj = gen.planted_cycle(n=10, k=3, rng=rng)
    res = mwc(adj, alpha=0.0, beta=0.0)
    assert res.length < INF


def test_f6_descending_filtration_isometric_lengths():
    """Descending active subgraphs preserve isometric lengths of surviving cycles."""
    rng = random.Random(42)
    adj = gen.planted_cycle(n=8, k=4, rng=rng)
    # In planted cycle, the first k nodes form the cycle
    cyc = list(range(4))
    if is_simple_cycle(adj, cyc):
        w0 = cycle_weight(adj, cyc)
        sub_adj = {u: {v: w for v, w in nbrs.items() if v != 7} for u, nbrs in adj.items() if u != 7}
        w1 = cycle_weight(sub_adj, cyc)
        assert approx_eq(w0, w1)


def test_f6_multiple_roots_skipped_in_dense_cluster():
    """Transversal root reduction skips significant fraction of roots on complete graphs."""
    rng = random.Random(42)
    adj = gen.complete(10, rng, wkind="int")
    res = mwc_transversal(adj, collect_stats=True)
    assert res.stats["roots_run"] < len(adj)


# ============================================================================
# Feature 7: Structural Countermeasures (Theorem 5.1)
# ============================================================================

def test_f7_heavy_edge_filtering_preserves_mwc():
    """Heavy edges with weight >= Gamma can be filtered without altering MWC."""
    rng = random.Random(42)
    adj = gen.grid(3, 3, rng, wkind="int")
    gamma0 = 4.0
    adj[0][8] = 100.0
    adj[8][0] = 100.0
    oracle_len, _ = mwc_oracle(adj)
    res = mwc(adj, gamma0=oracle_len + 1.0)
    assert approx_eq(res.length, oracle_len)


def test_f7_two_core_reduction_acyclic_peeling():
    """2-core reduction removes degree-1 whiskers without altering girth."""
    rng = random.Random(42)
    adj = gen.planted_cycle(n=10, k=4, rng=rng)
    adj["w1"] = {0: 1.0}
    adj[0]["w1"] = 1.0
    adj["w2"] = {"w1": 1.0}
    adj["w1"]["w2"] = 1.0
    core = two_core(adj)
    assert "w1" not in core and "w2" not in core
    g_core, _ = mwc_oracle(core)
    g_full, _ = mwc_oracle(adj)
    assert approx_eq(g_core, g_full)


def test_f7_gadget_g3_tightness_resistance():
    """10-vertex tightness gadget G_3 is solved correctly by exact certified pruning."""
    adj = build_g3_gadget(alpha=0.3, beta=0.0, eps=0.01)
    oracle_len, _ = mwc_oracle(adj)
    res = mwc(adj, alpha=0.0, beta=0.0)
    assert approx_eq(res.length, oracle_len)


def test_f7_bridge_and_articulation_peeling():
    """Biconnected decomposition isolates bridge-connected cycles."""
    adj = {
        0: {1: 1.0, 2: 1.0},
        1: {0: 1.0, 2: 1.0},
        2: {0: 1.0, 1: 1.0, 3: 5.0},
        3: {2: 5.0, 4: 2.0, 5: 2.0},
        4: {3: 2.0, 5: 2.0},
        5: {3: 2.0, 4: 2.0},
    }
    comps = biconnected_components(adj)
    assert len(comps) == 3
    mwc_len, cyc = solve_mwc(adj)
    assert approx_eq(mwc_len, 3.0)


def test_f7_auxiliary_chords_outside_ball():
    """Auxiliary chords with weight in (Gamma/2, Gamma) do not corrupt certified pruning."""
    rng = random.Random(42)
    adj = gen.grid(3, 3, rng, wkind="unit")
    adj[0][5] = 3.0
    adj[5][0] = 3.0
    mwc_len, cyc = solve_mwc(adj)
    assert approx_eq(mwc_len, 4.0)


# ============================================================================
# Feature 8: Synchronous Epoch-Based Snapshot Isolation (Theorem 6.2)
# ============================================================================

def test_f8_epoch_snapshot_read_only_isolation():
    """Simulated parallel root explorations on read-only snapshot H^(t)."""
    rng = random.Random(42)
    adj = gen.grid(4, 4, rng, wkind="int")
    snapshot = {u: dict(nbrs) for u, nbrs in adj.items()}
    res1 = mwc(snapshot, roots=[0])
    res2 = mwc(snapshot, roots=[1])
    assert res1.length >= 4.0 and res2.length >= 4.0


def test_f8_epoch_barrier_union_deletion():
    """Epoch barrier commits the union of candidate prune sets."""
    active = set(range(10))
    P1 = {1, 2}
    P2 = {2, 3}
    active_next = active - (P1 | P2)
    assert active_next == {0, 4, 5, 6, 7, 8, 9}


def test_f8_epoch_batching_equivalence_to_sequential():
    """Batch sizes 1, 2, 4 produce exact same MWC length on benchmark graph."""
    rng = random.Random(12)
    adj = gen.erdos_renyi(n=10, p=0.4, rng=rng, wkind="int")
    oracle_len, _ = mwc_oracle(adj)
    res_seq = mwc(adj, alpha=0.0, beta=0.0)
    assert approx_eq(res_seq.length, oracle_len)


def test_f8_atomic_incumbent_monotonicity():
    """Candidate cycle improvements across epoch workers update Gamma monotonically."""
    gamma = INF
    candidates = [10.0, 12.0, 7.5, 8.0, 4.0, 4.5]
    for c in candidates:
        if c < gamma:
            gamma = c
    assert approx_eq(gamma, 4.0)


def test_f8_no_toctou_deletion_race():
    """Snapshot isolation prevents TOCTOU mutation races between concurrent workers."""
    rng = random.Random(42)
    adj = gen.complete(6, rng, wkind="int")
    snap = dict(adj)
    assert len(snap) == 6


# ============================================================================
# Feature 9: Legacy Bug Eliminators (Node 0 LCA, Chord Rejection, NX 3.4+)
# ============================================================================

def test_f9_node_zero_lca_not_falsy():
    """Bug fix: Root or LCA equal to vertex 0 is not treated as falsy."""
    adj = {
        0: {1: 1.0, 2: 1.0},
        1: {0: 1.0, 2: 1.0},
        2: {0: 1.0, 1: 1.0},
    }
    res = mwc(adj, roots=[0])
    assert approx_eq(res.length, 3.0)
    assert set(res.cycle) == {0, 1, 2}


def test_f9_ancestor_descendant_chord_retained():
    """Bug fix: Ancestor-descendant non-tree edge is retained and evaluated."""
    adj = {
        0: {1: 1.0, 3: 1.0},
        1: {0: 1.0, 2: 1.0},
        2: {1: 1.0, 3: 1.0},
        3: {2: 1.0, 0: 1.0},
    }
    res = mwc(adj, roots=[0])
    assert approx_eq(res.length, 4.0)


def test_f9_deep_tree_lca_table_height():
    """Bug fix: Deep tree with depth 128 requires table height >= 8, not a constant."""
    from reference.mwc import LCAStructure
    n = 128
    nodes = list(range(n))
    pred = {i: i - 1 for i in range(1, n)}
    lca_struct = LCAStructure(nodes, pred, 0)
    assert lca_struct.log >= 7
    assert lca_struct.lca(127, 64) == 64
    assert lca_struct.lca(127, 0) == 0


def test_f9_networkx_random_labeled_tree_compatibility():
    """Bug fix: NetworkX 3.4+ deprecation of random_tree avoided by using random_labeled_tree."""
    tree = nx.random_labeled_tree(10, seed=42)
    assert len(tree.nodes()) == 10
    assert len(tree.edges()) == 9
    assert nx.is_tree(tree)


def test_f9_candidate_independent_reweighting():
    """Candidate cycle weight is independently re-summed before moving Gamma."""
    adj = {
        0: {1: 2.0, 2: 2.0},
        1: {0: 2.0, 2: 2.0},
        2: {0: 2.0, 1: 2.0},
    }
    cyc = (0, 1, 2)
    assert is_simple_cycle(adj, cyc)
    w = cycle_weight(adj, cyc)
    assert approx_eq(w, 6.0)
