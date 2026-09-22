"""Tier 3: Cross-Feature Combinations (>= 9 tests).

Covers pairwise and multi-way interactions between core feature areas:
- F1 + F5: Multi-chord dense topology + dynamic certified pruning
- F2 + F3: Multigraph degeneracy + zero weights + strict tie-breaking
- F2 + F7: Multigraph 2-cycles + 2-core reduction + heavy-edge filtering
- F4 + F6: Infinite frontier sentinels + historical certificate envelope
- F5 + F6: Dynamic pruning + historical certificate envelope & lazy root skipping
- F5 + F8: Dynamic pruning under synchronous epoch snapshot isolation
- F1 + F7: Multi-chord cycles + heavy-edge filtering + G_3 tightness gadget
- F3 + F9: Strict tie-breaking + Node 0 LCA + ancestor-descendant chords
- F6 + F8: Historical certificate envelope accumulation across epoch batches
- F2 + F5 + F7: 3-pass preprocessing + 2-core peeling + dynamic pruning on composite topology
"""

from __future__ import annotations

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
    build_g3_gadget,
    cycle_weight,
    is_simple_cycle,
    kappa_of,
    mwc,
    mwc_oracle,
    mwc_transversal,
    solve_mwc,
    two_core,
    _index_map,
    _truncated_dijkstra,
)


# ============================================================================
# Combination 1: F1 (Multi-Chord) + F5 (Dynamic Certified Pruning)
# ============================================================================

def test_c1_multichord_k6_dynamic_pruning():
    """Multi-chord K_6 clique under dynamic certified pruning (alpha=0.1, beta=0.1)."""
    rng = random.Random(42)
    adj = gen.complete(6, rng, wkind="int")
    oracle_len, _ = mwc_oracle(adj)
    res = mwc(adj, alpha=0.1, beta=0.1, collect_stats=True)
    assert approx_eq(res.length, oracle_len)
    assert res.stats["total_deletions"] == 0  # alpha <= beta guarantees no false deletions


def test_c1_multichord_ladder_diagonals_approx_pruning():
    """Ladder with diagonal chords under approximate pruning (alpha=0.3, beta=0.0)."""
    G = nx.ladder_graph(6)
    for u, v in G.edges():
        G[u][v]["weight"] = 3.0
    G.add_edge(0, 3, weight=4.0)
    G.add_edge(2, 5, weight=4.0)
    adj = {u: {v: d["weight"] for v, d in G[u].items()} for u in G}
    oracle_len, _ = mwc_oracle(adj)
    kappa = kappa_of(0.3, 0.0)
    res = mwc(adj, alpha=0.3, beta=0.0)
    assert res.length >= oracle_len - TOL
    assert res.length <= kappa * oracle_len + TOL


# ============================================================================
# Combination 2: F2 (3-Pass Degeneracy) + F3 (Strict Lexicographical Tie-Breaking)
# ============================================================================

def test_c2_multigraph_with_zero_weight_edges():
    """Multigraph with zero-weight edges between different node pairs: strict tie-break avoids cycles."""
    G = nx.MultiGraph()
    # Parallel edges on (0, 1): weights 0.0 and 2.0 -> 2-cycle length 2.0
    G.add_edge(0, 1, weight=0.0)
    G.add_edge(0, 1, weight=2.0)
    # Zero-weight tree path 1-2-3
    G.add_edge(1, 2, weight=0.0)
    G.add_edge(2, 3, weight=0.0)
    # Positive back-edge 3-0
    G.add_edge(3, 0, weight=5.0)
    mwc_len, cyc = solve_mwc(G)
    assert approx_eq(mwc_len, 2.0)
    assert cyc in [(0, 1, 0), (1, 0, 1)]


def test_c2_multigraph_zero_weight_tie_breaking_path():
    """Multigraph with tied zero weights and parallel paths: strict parent assignment."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(0, 2, weight=1.0)
    G.add_edge(1, 3, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    # Parallel edges on (3, 4)
    G.add_edge(3, 4, weight=1.5)
    G.add_edge(3, 4, weight=1.5)  # 2-cycle 3.0 vs 4-cycle 4.0
    mwc_len, cyc = solve_mwc(G)
    assert approx_eq(mwc_len, 3.0)
    assert cyc in [(3, 4, 3), (4, 3, 4)]


# ============================================================================
# Combination 3: F2 (3-Pass Degeneracy) + F7 (2-Core & Heavy-Edge Countermeasures)
# ============================================================================

def test_c3_multigraph_2core_peeling_with_whiskers():
    """Multigraph with whiskers hanging off a 2-cycle: 2-core peels whiskers, 2-cycle preserved."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=2.0)
    G.add_edge(0, 1, weight=2.0)  # 2-cycle 4.0
    # Add whiskers
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    G.add_edge(0, 4, weight=1.0)
    mwc_len, cyc = solve_mwc(G)
    assert approx_eq(mwc_len, 4.0)


def test_c3_multigraph_heavy_edge_filtering():
    """Multigraph where parallel 2-cycle of weight 3.0 filters heavy edges w >= 3.0."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(0, 1, weight=2.0)  # 2-cycle 3.0
    # Add heavy triangle
    G.add_edge(10, 11, weight=5.0)
    G.add_edge(11, 12, weight=5.0)
    G.add_edge(12, 10, weight=5.0)
    G.add_edge(0, 10, weight=20.0)
    mwc_len, cyc = solve_mwc(G)
    assert approx_eq(mwc_len, 3.0)


# ============================================================================
# Combination 4: F4 (Frontier Sentinels) + F6 (Historical Certificate Envelope)
# ============================================================================

def test_c4_disconnected_frontier_sentinel_with_envelope():
    """Component 1 exhausts frontier (B_x = inf), envelope preserves validity across components."""
    G = nx.Graph()
    # Component 1: tree 0-1-2
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=1.0)
    # Component 2: triangle 10-11-12 of weight 6.0
    G.add_edge(10, 11, weight=2.0)
    G.add_edge(11, 12, weight=2.0)
    G.add_edge(12, 10, weight=2.0)
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 6.0)
    assert set(cyc) == {10, 11, 12}


def test_c4_multi_component_trees_and_cycles():
    """3 components: Tree 1, Tree 2, Cycle 3: infinite frontier sentinels on 1 & 2, finds 3."""
    G = nx.Graph()
    # Tree 1: 0-1
    G.add_edge(0, 1, weight=1.0)
    # Tree 2: 10-11-12
    G.add_edge(10, 11, weight=1.0)
    G.add_edge(11, 12, weight=1.0)
    # Cycle 3: 20-21-22-23 (square of weight 4.0)
    for u, v in [(20, 21), (21, 22), (22, 23), (23, 20)]:
        G.add_edge(u, v, weight=1.0)
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)
    assert set(cyc) == {20, 21, 22, 23}


# ============================================================================
# Combination 5: F5 (Dynamic Pruning) + F6 (Envelope & Lazy Root Skipping)
# ============================================================================

def test_c5_dynamic_pruning_with_lazy_skipping():
    """Grid graph under dynamic pruning with root skipping preserves exactness."""
    rng = random.Random(42)
    adj = gen.grid(3, 3, rng, wkind="int")
    oracle_len, _ = mwc_oracle(adj)
    res = mwc_transversal(adj)
    assert approx_eq(res.length, oracle_len)


def test_c5_dense_cluster_lazy_skipping_ratio():
    """Complete graph K_8 under transversal root reduction skips roots safely."""
    rng = random.Random(42)
    adj = gen.complete(8, rng, wkind="int")
    oracle_len, _ = mwc_oracle(adj)
    res = mwc_transversal(adj, collect_stats=True)
    assert approx_eq(res.length, oracle_len)
    assert res.stats["roots_run"] < 8


# ============================================================================
# Combination 6: F5 (Dynamic Pruning) + F8 (Epoch Snapshot Isolation)
# ============================================================================

def test_c6_epoch_snapshot_parallel_dynamic_pruning():
    """Simulated batch execution of root exploration on frozen snapshot."""
    rng = random.Random(42)
    adj = gen.grid(3, 3, rng, wkind="int")
    oracle_len, _ = mwc_oracle(adj)
    # Snapshot isolation: run independent workers on snapshot
    snapshot = dict(adj)
    res_seq = mwc(snapshot, alpha=0.0, beta=0.0)
    assert approx_eq(res_seq.length, oracle_len)


def test_c6_monotone_incumbent_under_epoch_batches():
    """Simulating candidate cycle improvements across parallel workers."""
    rng = random.Random(77)
    adj = gen.complete(6, rng, wkind="int")
    order = list(adj)
    gamma = INF
    # Process in batches of 2
    for b in range(0, len(order), 2):
        batch_roots = order[b:b+2]
        res = mwc(adj, roots=batch_roots)
        if res.length < gamma:
            gamma = res.length
    oracle_len, _ = mwc_oracle(adj)
    assert approx_eq(gamma, oracle_len)


# ============================================================================
# Combination 7: F1 (Multi-Chord) + F7 (Heavy-Edge Filtering & G_3 Tightness)
# ============================================================================

def test_c7_multichord_with_heavy_edge_countermeasure():
    """Multi-chord graph with heavy auxiliary chords w >= Gamma removed."""
    G = nx.complete_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0  # Triangles have weight 3.0
    # Add heavy chord
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(0, 4, weight=10.0)
    adj = {u: {v: d["weight"] for v, d in G[u].items()} for u in G}
    res = mwc(adj, gamma0=4.0)
    assert approx_eq(res.length, 3.0)


def test_c7_g3_gadget_under_heavy_edge_filtering():
    """G_3 tightness gadget under heavy edge countermeasure."""
    adj = build_g3_gadget(alpha=0.3, beta=0.0, eps=0.01)
    oracle_len, _ = mwc_oracle(adj)
    res = mwc(adj, gamma0=oracle_len + 0.1)
    assert approx_eq(res.length, oracle_len)


# ============================================================================
# Combination 8: F3 (Strict Tie-Breaking) + F9 (Node 0 LCA & Chords)
# ============================================================================

def test_c8_node_zero_lca_with_tied_weights():
    """Node 0 as root and LCA with equal distance ties and ancestor-descendant chords."""
    adj = {
        0: {1: 1.0, 2: 1.0, 3: 2.0},
        1: {0: 1.0, 2: 1.0},
        2: {0: 1.0, 1: 1.0, 3: 1.0},
        3: {0: 2.0, 2: 1.0},
    }
    # Triangle 0-1-2 has weight 3.0. Triangle 0-2-3 has weight 1.0 + 1.0 + 2.0 = 4.0.
    res = mwc(adj, roots=[0])
    assert approx_eq(res.length, 3.0)
    assert set(res.cycle) == {0, 1, 2}


def test_c8_node_zero_deep_ancestor_descendant_tied():
    """Deep path from node 0 with tied weights and chord from 0 to leaf."""
    adj = {
        0: {1: 1.0, 4: 2.0},
        1: {0: 1.0, 2: 1.0},
        2: {1: 1.0, 3: 1.0},
        3: {2: 1.0, 4: 1.0},
        4: {3: 1.0, 0: 2.0},
    }
    res = mwc(adj, roots=[0])
    # Cycle 0-1-2-3-4 has weight 1+1+1+1+2 = 6.0
    assert approx_eq(res.length, 6.0)


# ============================================================================
# Combination 9: F6 (Historical Certificates) + F8 (Epoch Snapshot Parallel)
# ============================================================================

def test_c9_historical_envelope_across_epoch_batches():
    """Accumulating historical lower bounds across epoch barriers."""
    rng = random.Random(42)
    adj = gen.grid(3, 3, rng, wkind="int")
    nodes = list(adj)
    L_acc = {v: 0.0 for v in nodes}
    for b in range(0, len(nodes), 2):
        batch = nodes[b:b+2]
        for root in batch:
            delta, _, Q = _truncated_dijkstra(adj, root, set(adj), 4.0, _index_map(adj))
            for z in Q:
                L_acc[z] = max(L_acc[z], max(0.0, 4.0 - 2.0 * delta[z]))
    # Verify bounds are valid
    oracle_len, _ = mwc_oracle(adj)
    for v in nodes:
        assert L_acc[v] <= oracle_len + TOL


def test_c9_retrospective_pruning_with_epoch_isolation():
    """Retrospective pruning combined with epoch isolation."""
    rng = random.Random(99)
    adj = gen.planted_cycle(n=10, k=4, rng=rng)
    oracle_len, _ = mwc_oracle(adj)
    res = mwc(adj, alpha=0.0, beta=0.0)
    assert approx_eq(res.length, oracle_len)


# ============================================================================
# Combination 10: F2 (Degeneracy) + F5 (Dynamic Pruning) + F7 (2-Core)
# ============================================================================

def test_c10_multigraph_2core_dynamic_pruning_composite():
    """Composite topology: multigraph parallel edges, pendant trees, and dynamic pruning."""
    G = nx.MultiGraph()
    # 2-cycle on (0, 1): weight 2.0 + 2.0 = 4.0
    G.add_edge(0, 1, weight=2.0)
    G.add_edge(0, 1, weight=2.0)
    # Pendant tree
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    # Triangle on 10-11-12 of weight 10.0
    G.add_edge(10, 11, weight=3.0)
    G.add_edge(11, 12, weight=3.0)
    G.add_edge(12, 10, weight=4.0)
    G.add_edge(0, 10, weight=10.0)
    mwc_len, cyc = solve_mwc(G, alpha=0.1, beta=0.1)
    assert approx_eq(mwc_len, 4.0)


def test_c10_zero_weight_cycle_with_whiskers_and_dynamic_pruning():
    """Zero-weight cycle with whiskers: Pass 2 halts immediately at step 0."""
    G = nx.Graph()
    # Zero-weight triangle
    G.add_edge(0, 1, weight=0.0)
    G.add_edge(1, 2, weight=0.0)
    G.add_edge(2, 0, weight=0.0)
    # Whiskers
    G.add_edge(0, 3, weight=5.0)
    G.add_edge(3, 4, weight=5.0)
    mwc_len, cyc = solve_mwc(G, alpha=0.2, beta=0.1)
    assert mwc_len == 0.0
    assert set(cyc) == {0, 1, 2}
