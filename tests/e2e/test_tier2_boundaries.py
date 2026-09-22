"""Tier 2: Boundary and Corner Cases (>= 45 tests).

Covers edge cases, degenerate inputs, topological extremes, and boundary conditions:
- Empty graphs, single vertices, edgeless graphs, single edge
- Multigraphs with identical and distinct weights, multiple parallel pairs
- Zero-weight cycles of various lengths (2-cycle, 3-cycle, 4-cycle, 5-cycle, 6-cycle)
- Zero-weight cycles embedded in large positive graphs
- Zero-weight trees and forests
- Trees and disconnected forests
- Dense complete cliques (K_3, K_4, K_5, K_6, K_7, K_8, K_10, K_16)
- Stars and stars with peripheral chords
- Wheel graphs (W_4, W_5, W_6, W_8)
- Ladder graphs (L_3, L_5, L_8)
- Circular ladder / Prism graphs (C_3 x P_2, C_4 x P_2, C_5 x P_2)
- Möbius ladder graphs (M_4, M_6)
- Grid graphs (2x2, 3x3, 4x4)
- Complete bipartite graphs (K_{2,2}, K_{2,3}, K_{3,3}, K_{3,4})
- Disconnected graphs with MWC in later components
- High-degree tightness gadgets G_3 (eta = 0.1, 0.01, 0.001)
- Extreme weight contrasts (1e-6 vs 1e9)
- Floating-point near-ties (differing by 1e-8)
- Crown graphs and Hypercubes (Q_3, Q_4)
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
    find_zero_weight_cycle,
    is_acyclic_forest,
    mwc_oracle,
    solve_mwc,
)


def make_mobius_ladder(n: int) -> nx.Graph:
    """Construct Möbius ladder graph M_n with 2*n vertices."""
    G = nx.cycle_graph(2 * n)
    for i in range(n):
        G.add_edge(i, i + n)
    return G


# ============================================================================
# 1. Null, Trivial, and Degenerate Small Graphs
# ============================================================================

def test_t2_empty_graph():
    """Empty graph: V = empty, E = empty -> (inf, None)."""
    G = nx.Graph()
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)
    assert cyc is None or len(cyc) == 0


def test_t2_single_isolated_vertex():
    """Single isolated vertex: V = {0}, E = empty -> (inf, None)."""
    G = nx.Graph()
    G.add_node(0)
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)
    assert cyc is None or len(cyc) == 0


def test_t2_multiple_isolated_vertices_small():
    """3 isolated vertices with no edges -> (inf, None)."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)
    assert cyc is None or len(cyc) == 0


def test_t2_multiple_isolated_vertices_large():
    """10 isolated vertices with no edges -> (inf, None)."""
    G = nx.Graph()
    G.add_nodes_from(range(10))
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)
    assert cyc is None or len(cyc) == 0


def test_t2_single_edge_graph():
    """Single edge: V = {0, 1}, E = {(0, 1)} -> (inf, None)."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=5.0)
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)
    assert cyc is None or len(cyc) == 0


# ============================================================================
# 2. Trees and Disconnected Forests (Acyclic)
# ============================================================================

def test_t2_tree_path_5_vertices():
    """Linear path with 5 vertices -> (inf, None)."""
    G = nx.path_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.5
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)


def test_t2_tree_binary_balanced():
    """Balanced binary tree with 7 nodes -> (inf, None)."""
    G = nx.balanced_tree(r=2, h=2)
    for u, v in G.edges():
        G[u][v]["weight"] = 2.0
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)


def test_t2_forest_two_disconnected_trees():
    """Two disconnected trees (path and star) -> (inf, None)."""
    G = nx.Graph()
    # Tree 1: 0-1-2
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=2.0)
    # Tree 2: 10 connected to 11, 12, 13
    G.add_edge(10, 11, weight=1.0)
    G.add_edge(10, 12, weight=1.0)
    G.add_edge(10, 13, weight=1.0)
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)


def test_t2_forest_five_disconnected_trees():
    """Five small trees in forest -> (inf, None)."""
    G = nx.Graph()
    for base in range(0, 50, 10):
        G.add_edge(base, base + 1, weight=1.0)
        G.add_edge(base + 1, base + 2, weight=1.0)
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)


# ============================================================================
# 3. Multigraphs (Identical, Distinct, Dominance, Multiple Pairs)
# ============================================================================

def test_t2_multigraph_identical_weights():
    """Multigraph with 2 parallel edges of weight 3.0 -> 2-cycle length 6.0."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=3.0)
    G.add_edge(0, 1, weight=3.0)
    mwc_len, cyc = solve_mwc(G)
    assert approx_eq(mwc_len, 6.0)
    assert_valid_cycle(G, cyc, 6.0)


def test_t2_multigraph_distinct_weights():
    """Multigraph with 2 parallel edges of weights 2.0 and 5.0 -> 2-cycle length 7.0."""
    G = nx.MultiGraph()
    G.add_edge("a", "b", weight=2.0)
    G.add_edge("a", "b", weight=5.0)
    mwc_len, cyc = solve_mwc(G)
    assert approx_eq(mwc_len, 7.0)
    assert_valid_cycle(G, cyc, 7.0)


def test_t2_multigraph_four_parallel_edges():
    """Multigraph with 4 parallel edges on same pair: sorts weights (1.0, 3.0, 6.0, 9.0) -> 4.0."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=6.0)
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(0, 1, weight=9.0)
    G.add_edge(0, 1, weight=3.0)
    mwc_len, cyc = solve_mwc(G)
    assert approx_eq(mwc_len, 4.0)


def test_t2_multigraph_multiple_parallel_pairs():
    """Multigraph with two pairs of parallel edges: pair (0, 1) has 2+3=5, pair (2, 3) has 1+2=3."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=2.0)
    G.add_edge(0, 1, weight=3.0)
    G.add_edge(2, 3, weight=1.0)
    G.add_edge(2, 3, weight=2.0)
    G.add_edge(1, 2, weight=10.0)
    mwc_len, cyc = solve_mwc(G)
    assert approx_eq(mwc_len, 3.0)
    assert cyc in [(2, 3, 2), (3, 2, 3)]


def test_t2_multigraph_2cycle_vs_3cycle_tie():
    """Multigraph where 2-cycle and 3-cycle have exactly tied length."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=3.0)
    G.add_edge(0, 1, weight=3.0)  # 2-cycle length 6.0
    G.add_edge(1, 2, weight=2.0)
    G.add_edge(2, 0, weight=1.0)  # Triangle 0-1-2 has weight 3.0 + 2.0 + 1.0 = 6.0
    mwc_len, cyc = solve_mwc(G)
    assert approx_eq(mwc_len, 6.0)


def test_t2_multigraph_4cycle_lighter_than_2cycle():
    """Multigraph where a 4-cycle (weight 4.0) is lighter than a 2-cycle (weight 10.0)."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=5.0)
    G.add_edge(0, 1, weight=5.0)  # 2-cycle 10.0
    G.add_edge(2, 3, weight=1.0)
    G.add_edge(3, 4, weight=1.0)
    G.add_edge(4, 5, weight=1.0)
    G.add_edge(5, 2, weight=1.0)
    mwc_len, cyc = solve_mwc(G)
    assert approx_eq(mwc_len, 4.0)
    assert set(cyc) == {2, 3, 4, 5}


# ============================================================================
# 4. Zero-Weight Cycles & Topologies (Pass 2 Preprocessing)
# ============================================================================

def test_t2_zero_weight_2cycle_multigraph():
    """Multigraph with two zero-weight edges between same pair -> 0.0 immediately."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=0.0)
    G.add_edge(0, 1, weight=0.0)
    mwc_len, cyc = solve_mwc(G)
    assert mwc_len == 0.0


def test_t2_zero_weight_3cycle():
    """Simple graph with zero-weight 3-cycle -> 0.0 immediately."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=0.0)
    G.add_edge(1, 2, weight=0.0)
    G.add_edge(2, 0, weight=0.0)
    mwc_len, cyc = solve_mwc(G)
    assert mwc_len == 0.0
    assert set(cyc) == {0, 1, 2}


def test_t2_zero_weight_4cycle():
    """Simple graph with zero-weight 4-cycle -> 0.0."""
    G = nx.cycle_graph(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 0.0
    mwc_len, cyc = solve_mwc(G)
    assert mwc_len == 0.0
    assert len(cyc) == 4


def test_t2_zero_weight_5cycle():
    """Simple graph with zero-weight 5-cycle -> 0.0."""
    G = nx.cycle_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 0.0
    mwc_len, cyc = solve_mwc(G)
    assert mwc_len == 0.0
    assert len(cyc) == 5


def test_t2_zero_weight_6cycle():
    """Simple graph with zero-weight 6-cycle -> 0.0."""
    G = nx.cycle_graph(6)
    for u, v in G.edges():
        G[u][v]["weight"] = 0.0
    mwc_len, cyc = solve_mwc(G)
    assert mwc_len == 0.0
    assert len(cyc) == 6


def test_t2_zero_weight_cycle_embedded_in_heavy_graph():
    """Zero-weight triangle embedded in a graph of weight 1000 edges."""
    G = nx.grid_2d_graph(3, 3)
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        G[u][v]["weight"] = 1000.0
    G.add_edge(10, 11, weight=0.0)
    G.add_edge(11, 12, weight=0.0)
    G.add_edge(12, 10, weight=0.0)
    G.add_edge(0, 10, weight=500.0)
    mwc_len, cyc = solve_mwc(G)
    assert mwc_len == 0.0
    assert set(cyc) == {10, 11, 12}


def test_t2_zero_weight_tree_with_no_cycles():
    """Zero-weight tree (path of 6 nodes) has no zero-weight cycles -> (inf, None)."""
    G = nx.path_graph(6)
    for u, v in G.edges():
        G[u][v]["weight"] = 0.0
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)


def test_t2_zero_weight_star_no_cycles():
    """Star with 8 zero-weight spokes has no cycles -> (inf, None)."""
    G = nx.star_graph(8)
    for u, v in G.edges():
        G[u][v]["weight"] = 0.0
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)


def test_t2_all_zero_weights_complete_k4():
    """All edges zero-weight on K_4 -> 0.0 at Pass 2."""
    G = nx.complete_graph(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 0.0
    mwc_len, cyc = solve_mwc(G)
    assert mwc_len == 0.0


# ============================================================================
# 5. Dense Complete Cliques (K_3 to K_16)
# ============================================================================

def test_t2_clique_k3():
    """Complete graph K_3 with unit weights -> 3.0."""
    G = nx.complete_graph(3)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 3.0)


def test_t2_clique_k4():
    """Complete graph K_4 with unit weights -> 3.0."""
    G = nx.complete_graph(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 3.0)


def test_t2_clique_k5():
    """Complete graph K_5 with unit weights -> 3.0."""
    G = nx.complete_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 3.0)


def test_t2_clique_k6():
    """Complete graph K_6 with random positive weights."""
    G = nx.complete_graph(6)
    rng = random.Random(42)
    for u, v in G.edges():
        G[u][v]["weight"] = rng.uniform(1.0, 5.0)
    assert_mwc_matches_oracle(G)


def test_t2_clique_k7():
    """Complete graph K_7 with random positive weights."""
    G = nx.complete_graph(7)
    rng = random.Random(77)
    for u, v in G.edges():
        G[u][v]["weight"] = rng.uniform(1.0, 10.0)
    assert_mwc_matches_oracle(G)


def test_t2_clique_k8():
    """Complete graph K_8 with random positive weights."""
    G = nx.complete_graph(8)
    rng = random.Random(88)
    for u, v in G.edges():
        G[u][v]["weight"] = rng.uniform(1.0, 10.0)
    assert_mwc_matches_oracle(G)


def test_t2_clique_k10():
    """Complete graph K_10 with random positive weights."""
    G = nx.complete_graph(10)
    rng = random.Random(101)
    for u, v in G.edges():
        G[u][v]["weight"] = rng.uniform(1.0, 10.0)
    assert_mwc_matches_oracle(G)


def test_t2_clique_k16():
    """Complete graph K_16 with unit weights -> 3.0."""
    G = nx.complete_graph(16)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 3.0)


# ============================================================================
# 6. Stars & Stars with Peripheral Chords
# ============================================================================

def test_t2_star_s5():
    """Star graph S_5 has no cycles -> (inf, None)."""
    G = nx.star_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 2.0
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)


def test_t2_star_s10():
    """Star graph S_10 has no cycles -> (inf, None)."""
    G = nx.star_graph(10)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)


def test_t2_star_s5_with_one_chord():
    """Star graph S_5 with a single peripheral chord between leaves 1 and 2."""
    G = nx.star_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 2.0
    G.add_edge(1, 2, weight=1.0)
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 5.0)
    assert set(cyc) == {0, 1, 2}


def test_t2_star_s8_with_multiple_chords():
    """Star graph S_8 with two separate peripheral triangles."""
    G = nx.star_graph(8)
    for u, v in G.edges():
        G[u][v]["weight"] = 3.0
    G.add_edge(1, 2, weight=2.0)
    G.add_edge(3, 4, weight=1.0)
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 7.0)
    assert set(cyc) == {0, 3, 4}


# ============================================================================
# 7. Wheel Graphs (W_4, W_5, W_6, W_8)
# ============================================================================

def test_t2_wheel_w4():
    """Wheel graph W_4 (K_4) with unit weights -> 3.0."""
    G = nx.wheel_graph(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 3.0)


def test_t2_wheel_w5():
    """Wheel graph W_5 with unit weights -> 3.0."""
    G = nx.wheel_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 3.0)


def test_t2_wheel_w6():
    """Wheel graph W_6 with weighted hub and rim."""
    G = nx.wheel_graph(6)
    for u, v in G.edges():
        G[u][v]["weight"] = 2.0
    G[0][1]["weight"] = 0.5
    G[0][2]["weight"] = 0.5
    G[1][2]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 2.0)


def test_t2_wheel_w8():
    """Wheel graph W_8 with unit weights -> 3.0."""
    G = nx.wheel_graph(8)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 3.0)


# ============================================================================
# 8. Ladder and Circular Ladder (Prism) Graphs
# ============================================================================

def test_t2_ladder_l3():
    """Ladder graph L_3 with unit weights -> 4-cycle of weight 4.0."""
    G = nx.ladder_graph(3)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)


def test_t2_ladder_l5():
    """Ladder graph L_5 with unit weights -> 4.0."""
    G = nx.ladder_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)


def test_t2_ladder_l8():
    """Ladder graph L_8 with varying weights."""
    G = nx.ladder_graph(8)
    rng = random.Random(42)
    for u, v in G.edges():
        G[u][v]["weight"] = rng.uniform(1.0, 5.0)
    assert_mwc_matches_oracle(G)


def test_t2_circular_ladder_c3_prism():
    """Prism graph C_3 x P_2 with unit weights -> 3.0 (triangular bases)."""
    G = nx.circular_ladder_graph(3)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 3.0)


def test_t2_circular_ladder_c4_prism():
    """Circular ladder C_4 x P_2 (Cube Q_3) with unit weights -> 4.0."""
    G = nx.circular_ladder_graph(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)


def test_t2_circular_ladder_c5_prism():
    """Circular ladder C_5 x P_2 with unit weights -> 4.0 (square sides)."""
    G = nx.circular_ladder_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)


def test_t2_mobius_ladder_m4():
    """Möbius ladder graph M_4 with unit weights -> 4.0."""
    G = make_mobius_ladder(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)


def test_t2_mobius_ladder_m6():
    """Möbius ladder graph M_6 with unit weights -> 4.0."""
    G = make_mobius_ladder(6)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)


# ============================================================================
# 9. Grid Graphs
# ============================================================================

def test_t2_grid_2x2():
    """2x2 grid graph (single 4-cycle) -> 4.0."""
    G = nx.grid_2d_graph(2, 2)
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)


def test_t2_grid_3x3():
    """3x3 grid graph with unit weights -> 4.0."""
    G = nx.grid_2d_graph(3, 3)
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)


def test_t2_grid_4x4():
    """4x4 grid graph with unit weights -> 4.0."""
    G = nx.grid_2d_graph(4, 4)
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)


# ============================================================================
# 10. Complete Bipartite Graphs
# ============================================================================

def test_t2_bipartite_k22():
    """K_{2,2} (4-cycle) with unit weights -> 4.0."""
    G = nx.complete_bipartite_graph(2, 2)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)


def test_t2_bipartite_k23():
    """K_{2,3} with unit weights -> 4.0."""
    G = nx.complete_bipartite_graph(2, 3)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)


def test_t2_bipartite_k33():
    """K_{3,3} with unit weights -> 4.0."""
    G = nx.complete_bipartite_graph(3, 3)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)


def test_t2_bipartite_k34():
    """K_{3,4} with random weights."""
    G = nx.complete_bipartite_graph(3, 4)
    rng = random.Random(42)
    for u, v in G.edges():
        G[u][v]["weight"] = rng.uniform(1.0, 5.0)
    assert_mwc_matches_oracle(G)


# ============================================================================
# 11. Disconnected Graphs with MWC in Later Components
# ============================================================================

def test_t2_disconnected_mwc_in_second_component():
    """Component 1 is tree; Component 2 has triangle of weight 3.0."""
    G = nx.Graph()
    # Comp 1: 0-1-2
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=1.0)
    # Comp 2: 3-4-5
    G.add_edge(3, 4, weight=1.0)
    G.add_edge(4, 5, weight=1.0)
    G.add_edge(5, 3, weight=1.0)
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 3.0)
    assert set(cyc) == {3, 4, 5}


def test_t2_disconnected_mwc_in_third_component():
    """Component 1 and 2 have heavier cycles; Component 3 has lightest cycle."""
    G = nx.Graph()
    # Comp 1: 4-cycle of weight 40.0
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        G.add_edge(u, v, weight=10.0)
    # Comp 2: 3-cycle of weight 15.0
    for u, v in [(10, 11), (11, 12), (12, 10)]:
        G.add_edge(u, v, weight=5.0)
    # Comp 3: 3-cycle of weight 6.0
    for u, v in [(20, 21), (21, 22), (22, 20)]:
        G.add_edge(u, v, weight=2.0)
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 6.0)
    assert set(cyc) == {20, 21, 22}


def test_t2_disconnected_first_component_acyclic_second_light_third_tree():
    """Comp 1 is star; Comp 2 is triangle; Comp 3 is path."""
    G = nx.Graph()
    # Star 0 to 1,2,3
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(0, 2, weight=1.0)
    G.add_edge(0, 3, weight=1.0)
    # Triangle 10-11-12
    G.add_edge(10, 11, weight=2.0)
    G.add_edge(11, 12, weight=2.0)
    G.add_edge(12, 10, weight=2.0)
    # Path 20-21-22
    G.add_edge(20, 21, weight=1.0)
    G.add_edge(21, 22, weight=1.0)
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 6.0)


# ============================================================================
# 12. High-Degree Tightness Gadgets G_3
# ============================================================================

def test_t2_gadget_g3_eta_01():
    """10-vertex tightness gadget G_3 with eta = 0.1."""
    adj = build_g3_gadget(alpha=0.3, beta=0.0, eps=0.01)
    oracle_len, _ = mwc_oracle(adj)
    mwc_len, cyc = solve_mwc(adj)
    assert approx_eq(mwc_len, oracle_len)


def test_t2_gadget_g3_eta_001():
    """10-vertex tightness gadget G_3 with eta = 0.001."""
    adj = build_g3_gadget(alpha=0.25, beta=0.0, eps=0.002)
    oracle_len, _ = mwc_oracle(adj)
    mwc_len, cyc = solve_mwc(adj)
    assert approx_eq(mwc_len, oracle_len)


def test_t2_gadget_g3_eta_0001():
    """10-vertex tightness gadget G_3 with small epsilon = 0.0005."""
    adj = build_g3_gadget(alpha=0.4, beta=0.0, eps=0.001)
    oracle_len, _ = mwc_oracle(adj)
    mwc_len, cyc = solve_mwc(adj)
    assert approx_eq(mwc_len, oracle_len)


# ============================================================================
# 13. Weight Scale Extremes & Numerical Precision
# ============================================================================

def test_t2_extreme_weight_contrast_micro_vs_giga():
    """Weight scale contrast: cycle of 1e-6 vs edges of 1e9."""
    G = nx.Graph()
    # Micro cycle: 0-1-2 with 1e-6 each -> 3e-6
    G.add_edge(0, 1, weight=1e-6)
    G.add_edge(1, 2, weight=1e-6)
    G.add_edge(2, 0, weight=1e-6)
    # Giga edges
    G.add_edge(0, 3, weight=1e9)
    G.add_edge(3, 4, weight=1e9)
    G.add_edge(4, 0, weight=1e9)
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 3e-6)
    assert set(cyc) == {0, 1, 2}


def test_t2_extreme_weight_large_finite():
    """Graph with weights near 1e12."""
    G = nx.cycle_graph(3)
    for u, v in G.edges():
        G[u][v]["weight"] = 1e12
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 3e12)


def test_t2_nearly_tied_cycles_diff_1e8():
    """Two cycles differing by 1e-8: certified pruning chooses the strictly lighter one."""
    G = nx.Graph()
    # Cycle 1: 0-1-2 with weight 1.0 + 1.0 + 1.0 = 3.0
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 0, weight=1.0)
    # Cycle 2: 10-11-12 with weight 1.0 + 1.0 + 1.00000001 = 3.00000001
    G.add_edge(10, 11, weight=1.0)
    G.add_edge(11, 12, weight=1.0)
    G.add_edge(12, 10, weight=1.00000001)
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 3.0)
    assert set(cyc) == {0, 1, 2}


def test_t2_nearly_tied_squares():
    """Two squares differing by 1e-7."""
    G = nx.Graph()
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        G.add_edge(u, v, weight=2.0)  # total 8.0
    for u, v in [(10, 11), (11, 12), (12, 13), (13, 10)]:
        G.add_edge(u, v, weight=2.00000005)  # total 8.0000002
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 8.0)
    assert set(cyc) == {0, 1, 2, 3}


# ============================================================================
# 14. Path Graphs with Whiskers, Bridges, Crown, and Hypercubes
# ============================================================================

def test_t2_long_path_with_whiskers():
    """Path of 30 vertices with whiskers hanging off every node -> (inf, None)."""
    G = nx.path_graph(30)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    for i in range(30):
        G.add_edge(i, 100 + i, weight=2.0)
    mwc_len, cyc = solve_mwc(G)
    assert math.isinf(mwc_len)


def test_t2_long_bridge_between_two_triangles():
    """Path of 20 edges connecting two triangles."""
    G = nx.Graph()
    for u, v in [(0, 1), (1, 2), (2, 0)]:
        G.add_edge(u, v, weight=2.0)
    for i in range(2, 22):
        G.add_edge(i, i + 1, weight=1.0)
    for u, v in [(22, 23), (23, 24), (24, 22)]:
        G.add_edge(u, v, weight=1.0)
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 3.0)
    assert set(cyc) == {22, 23, 24}


def test_t2_crown_graph():
    """Crown graph on 6 vertices: K_{3,3} minus a perfect matching (6-cycle) -> 6.0."""
    G = nx.complete_bipartite_graph(3, 3)
    G.remove_edge(0, 3)
    G.remove_edge(1, 4)
    G.remove_edge(2, 5)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 6.0)


def test_t2_hypercube_q3():
    """3-dimensional hypercube Q_3 (8 vertices, 12 edges) with unit weights -> 4.0."""
    G = nx.hypercube_graph(3)
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)


def test_t2_hypercube_q4():
    """4-dimensional hypercube Q_4 (16 vertices, 32 edges) with unit weights -> 4.0."""
    G = nx.hypercube_graph(4)
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    mwc_len, cyc = assert_mwc_matches_oracle(G)
    assert approx_eq(mwc_len, 4.0)
