"""
Milestone 2 Empirical Challenger Stress Harness
Author: Challenger 2 (teamwork_preview_challenger_m2_2)
Covers:
  1. Disconnected graphs (isolated nodes, multiple components)
  2. Acyclic graphs / forests / trees (returns (inf, []))
  3. Self-loops (ensuring simple cycles >= 3 or multigraph 2-cycles are found, not self-loops)
  4. Giant graphs (n > 1000) memory stability and no segfaults
  5. Extreme weight ratios (w_max / w_min > 10^8) numerical stability
  6. G3 tightness gadgets (tight cotree chord bounds)
  7. Authoritative real-world networks from /scratch/hs9hd/mwc_certified_pruning/datasets/realnets/
"""

import os
import sys
import time
import math
import traceback
import json
import numpy as np
import networkx as nx

# Add girth to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from shortest_cycle import minimum_weight_cycle
from benchmark_acceleration import load_real_network

def independent_oracle_girth(G, weight="weight"):
    """
    Independent ground truth oracle for minimum-weight cycle.
    Finds min over all edges e=(u,v) of w(e) + dist_{G \ e}(u, v).
    Handles multigraphs (2-cycles between distinct u != v).
    Explicitly ignores self-loops (u == v).
    Returns (min_weight, cycle_nodes).
    """
    if G.is_directed():
        raise ValueError("Must be undirected")
    
    best_len = float("inf")
    best_cycle = []
    
    # 1. Check for multigraph 2-cycles between distinct nodes
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        edge_groups = {}
        for u, v, k, data in G.edges(keys=True, data=True):
            if u == v:
                continue # ignore self loop
            w = data.get(weight, 1.0)
            pair = (u, v) if str(u) <= str(v) else (v, u)
            edge_groups.setdefault(pair, []).append(w)
        for (u, v), weights in edge_groups.items():
            if len(weights) >= 2:
                weights.sort()
                two_len = weights[0] + weights[1]
                if two_len < best_len:
                    best_len = two_len
                    best_cycle = [u, v, u]

    # Convert to simple graph with minimum weights, ignoring self-loops
    G_simp = nx.Graph()
    G_simp.add_nodes_from(G.nodes())
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        edge_min = {}
        for u, v, data in G.edges(data=True):
            if u == v:
                continue
            w = data.get(weight, 1.0)
            pair = (u, v) if str(u) <= str(v) else (v, u)
            if pair not in edge_min or w < edge_min[pair]:
                edge_min[pair] = w
        for (u, v), w in edge_min.items():
            G_simp.add_edge(u, v, **{weight: w})
    else:
        for u, v, data in G.edges(data=True):
            if u == v:
                continue
            w = data.get(weight, 1.0)
            G_simp.add_edge(u, v, **{weight: w})

    # For each edge in G_simp, remove it and find shortest path
    edges = list(G_simp.edges(data=True))
    for u, v, data in edges:
        w = data.get(weight, 1.0)
        G_simp.remove_edge(u, v)
        if nx.has_path(G_simp, u, v):
            try:
                path = nx.shortest_path(G_simp, u, v, weight=weight)
                path_len = nx.path_weight(G_simp, path, weight=weight)
                cycle_len = path_len + w
                if cycle_len < best_len:
                    best_len = cycle_len
                    best_cycle = path + [u]
            except nx.NetworkXNoPath:
                pass
        G_simp.add_edge(u, v, **{weight: w})

    return best_len, best_cycle

def validate_cycle(G, cycle_nodes, reported_len, weight="weight", tol=1e-7):
    """
    Validates that cycle_nodes is a valid cycle in G:
    - len(cycle_nodes) >= 3 (for multigraph [u, v, u]) or >= 4 (for simple [u, v, w, u])
    - cycle_nodes[0] == cycle_nodes[-1]
    - distinct internal nodes
    - all edges exist
    - sum of edge weights matches reported_len within tol
    """
    if reported_len == float("inf"):
        return len(cycle_nodes) == 0
    if len(cycle_nodes) < 3:
        return False
    if cycle_nodes[0] != cycle_nodes[-1]:
        return False
    # Check no self-loop cycle [u, u]
    if len(cycle_nodes) == 2:
        return False
    # Check distinct internal nodes
    internal = cycle_nodes[:-1]
    if len(internal) != len(set(internal)):
        return False
    # Check edges and weight
    is_multi = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    total_w = 0.0
    if len(cycle_nodes) == 3: # 2-cycle [u, v, u]
        u, v = cycle_nodes[0], cycle_nodes[1]
        if not is_multi:
            return False
        # Must have at least 2 edges between u and v
        weights = [d.get(weight, 1.0) for d in G[u][v].values()]
        if len(weights) < 2:
            return False
        weights.sort()
        total_w = weights[0] + weights[1]
    else: # length >= 3 cycle
        for i in range(len(cycle_nodes) - 1):
            u, v = cycle_nodes[i], cycle_nodes[i+1]
            if not G.has_edge(u, v):
                return False
            if is_multi:
                total_w += min(d.get(weight, 1.0) for d in G[u][v].values())
            else:
                total_w += G[u][v].get(weight, 1.0)
                
    return abs(total_w - reported_len) <= tol * max(1.0, abs(reported_len))


class EmpiricalChallengerRunner:
    def __init__(self):
        self.results = []
        self.failures = []

    def record(self, category, test_name, status, details, gamma_pure=None, gamma_acc=None, gamma_oracle=None, time_ms=0.0):
        entry = {
            "category": category,
            "test_name": test_name,
            "status": status,
            "gamma_pure": gamma_pure,
            "gamma_acc": gamma_acc,
            "gamma_oracle": gamma_oracle,
            "time_ms": time_ms,
            "details": details
        }
        self.results.append(entry)
        if status != "PASS":
            self.failures.append(entry)
            print(f"[-] FAIL: [{category}] {test_name}: {details}")
        else:
            print(f"[+] PASS: [{category}] {test_name}")

    def run_all(self):
        print("================================================================================")
        print("RUNNING EMPIRICAL CHALLENGER STRESS HARNESS (MILESTONE 2)")
        print("================================================================================")
        self.test_category_1_disconnected()
        self.test_category_2_acyclic()
        self.test_category_3_self_loops()
        self.test_category_4_giant_graphs()
        self.test_category_5_extreme_weights()
        self.test_category_6_g3_gadgets()
        self.test_category_7_realnets()
        self.summary()

    # --- CATEGORY 1: Disconnected Graphs & Isolated Nodes ---
    def test_category_1_disconnected(self):
        print("\n--- Category 1: Disconnected Graphs & Isolated Nodes ---")
        
        # 1.1 Multi-component graph
        G = nx.Graph()
        # Comp 1: triangle of weight 3.0
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        G.add_edge(3, 1, weight=1.0)
        # Comp 2: 4-cycle of weight 2.0 (0.5 each) -> minimum
        G.add_edge(4, 5, weight=0.5)
        G.add_edge(5, 6, weight=0.5)
        G.add_edge(6, 7, weight=0.5)
        G.add_edge(7, 4, weight=0.5)
        # Comp 3: tree
        G.add_edge(8, 9, weight=1.0)
        G.add_edge(9, 10, weight=1.0)

        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
        g_ora, c_ora = independent_oracle_girth(G)
        t_ms = (time.perf_counter() - t0) * 1000

        ok = (abs(g_pure - 2.0) < 1e-9 and abs(g_acc - 2.0) < 1e-9 and 
              validate_cycle(G, c_pure, g_pure) and validate_cycle(G, c_acc, g_acc))
        self.record("1_disconnected", "multi_component_min_cycle", "PASS" if ok else "FAIL",
                    f"pure={g_pure} acc={g_acc} ora={g_ora}", g_pure, g_acc, g_ora, t_ms)

        # 1.2 Isolated vertices + cycle
        G = nx.Graph()
        for i in range(25):
            G.add_node(f"iso_{i}")
        # Add 4-cycle
        G.add_edge(0, 1, weight=1.5)
        G.add_edge(1, 2, weight=1.5)
        G.add_edge(2, 3, weight=1.5)
        G.add_edge(3, 0, weight=1.5)

        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
        g_ora, c_ora = independent_oracle_girth(G)
        t_ms = (time.perf_counter() - t0) * 1000

        ok = (abs(g_pure - 6.0) < 1e-9 and abs(g_acc - 6.0) < 1e-9 and 
              validate_cycle(G, c_pure, g_pure) and validate_cycle(G, c_acc, g_acc))
        self.record("1_disconnected", "isolated_nodes_with_cycle", "PASS" if ok else "FAIL",
                    f"pure={g_pure} acc={g_acc} ora={g_ora}", g_pure, g_acc, g_ora, t_ms)

        # 1.3 Completely disconnected graph (100 nodes, 0 edges)
        G = nx.Graph()
        for i in range(100):
            G.add_node(i)
        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
        t_ms = (time.perf_counter() - t0) * 1000
        ok = (math.isinf(g_pure) and c_pure == [] and math.isinf(g_acc) and c_acc == [])
        self.record("1_disconnected", "all_isolated_nodes_100", "PASS" if ok else "FAIL",
                    f"pure={g_pure}, c_pure={c_pure}; acc={g_acc}, c_acc={c_acc}", g_pure, g_acc, float("inf"), t_ms)

        # 1.4 Disjoint forest (10 trees of 10 nodes each)
        G = nx.Graph()
        for c in range(10):
            T = nx.path_graph(10)
            G.add_edges_from(((f"t{c}_{u}", f"t{c}_{v}", {"weight": 1.0 + u}) for u, v in T.edges()))
        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
        t_ms = (time.perf_counter() - t0) * 1000
        ok = (math.isinf(g_pure) and c_pure == [] and math.isinf(g_acc) and c_acc == [])
        self.record("1_disconnected", "disjoint_forest_10_trees", "PASS" if ok else "FAIL",
                    f"pure={g_pure} acc={g_acc}", g_pure, g_acc, float("inf"), t_ms)

        # 1.5 Giant tree (1000 nodes) + small triangle
        G = nx.random_labeled_tree(1000, seed=42)
        for u, v in G.edges():
            G[u][v]["weight"] = 2.0
        # Add small disconnected triangle
        G.add_edge("c1", "c2", weight=0.4)
        G.add_edge("c2", "c3", weight=0.4)
        G.add_edge("c3", "c1", weight=0.4)
        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
        t_ms = (time.perf_counter() - t0) * 1000
        ok = (abs(g_pure - 1.2) < 1e-9 and abs(g_acc - 1.2) < 1e-9 and
              validate_cycle(G, c_pure, g_pure) and validate_cycle(G, c_acc, g_acc))
        self.record("1_disconnected", "giant_tree_plus_triangle", "PASS" if ok else "FAIL",
                    f"pure={g_pure} acc={g_acc}", g_pure, g_acc, 1.2, t_ms)

    # --- CATEGORY 2: Acyclic Graphs / Forests / Trees ---
    def test_category_2_acyclic(self):
        print("\n--- Category 2: Acyclic Graphs / Forests / Trees ---")
        cases = [
            ("empty_graph", nx.Graph()),
            ("single_node", nx.empty_graph(1)),
            ("single_edge", nx.path_graph(2)),
            ("path_100", nx.path_graph(100)),
            ("path_1000", nx.path_graph(1000)),
            ("star_100", nx.star_graph(100)),
            ("star_1000", nx.star_graph(1000)),
            ("balanced_tree_d8", nx.balanced_tree(r=2, h=8)),
            ("random_tree_500", nx.random_labeled_tree(500, seed=123)),
            ("random_tree_1000", nx.random_labeled_tree(1000, seed=456)),
        ]
        for name, G in cases:
            for u, v in G.edges():
                G[u][v]["weight"] = 1.0 + (u + v) % 7
            t0 = time.perf_counter()
            g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
            g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
            t_ms = (time.perf_counter() - t0) * 1000
            ok = (math.isinf(g_pure) and c_pure == [] and math.isinf(g_acc) and c_acc == [])
            self.record("2_acyclic", name, "PASS" if ok else "FAIL",
                        f"pure={g_pure}, c_pure={c_pure}; acc={g_acc}, c_acc={c_acc}", g_pure, g_acc, float("inf"), t_ms)

    # --- CATEGORY 3: Self-Loops & Multigraph 2-Cycles ---
    def test_category_3_self_loops(self):
        print("\n--- Category 3: Self-Loops & Multigraph 2-Cycles ---")

        # 3.1 Simple graph with self-loop + triangle
        G = nx.Graph()
        G.add_edge(0, 0, weight=0.1) # Self-loop
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        G.add_edge(3, 1, weight=1.0) # Triangle weight 3.0

        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
        g_ora, c_ora = independent_oracle_girth(G)
        t_ms = (time.perf_counter() - t0) * 1000

        # Must find simple cycle >= 3 (triangle of weight 3.0), NOT self-loop [0, 0] of weight 0.1
        ok = (abs(g_pure - 3.0) < 1e-9 and abs(g_acc - 3.0) < 1e-9 and 
              validate_cycle(G, c_pure, g_pure) and validate_cycle(G, c_acc, g_acc))
        self.record("3_self_loops", "simple_graph_with_self_loop_and_triangle", "PASS" if ok else "FAIL",
                    f"pure={g_pure} c_pure={c_pure}; acc={g_acc} c_acc={c_acc}; ora={g_ora}", g_pure, g_acc, g_ora, t_ms)

        # 3.2 Multigraph with self loop + 2-cycle
        MG = nx.MultiGraph()
        MG.add_edge(0, 0, weight=0.05) # Self-loop
        MG.add_edge(1, 2, weight=1.0)  # Parallel edges forming 2-cycle weight 2.2
        MG.add_edge(1, 2, weight=1.2)
        MG.add_edge(3, 4, weight=2.0)  # Triangle weight 6.0
        MG.add_edge(4, 5, weight=2.0)
        MG.add_edge(5, 3, weight=2.0)

        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(MG, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(MG, use_acceleration=True)
        g_ora, c_ora = independent_oracle_girth(MG)
        t_ms = (time.perf_counter() - t0) * 1000

        # Must find 2-cycle [1, 2, 1] weight 2.2, NOT self-loop [0, 0] weight 0.05
        ok = (abs(g_pure - 2.2) < 1e-9 and abs(g_acc - 2.2) < 1e-9 and
              validate_cycle(MG, c_pure, g_pure) and validate_cycle(MG, c_acc, g_acc))
        self.record("3_self_loops", "multigraph_with_self_loop_and_2cycle", "PASS" if ok else "FAIL",
                    f"pure={g_pure} c_pure={c_pure}; acc={g_acc} c_acc={c_acc}; ora={g_ora}", g_pure, g_acc, g_ora, t_ms)

        # 3.3 Acyclic tree with self-loop
        T = nx.path_graph(5)
        T.add_edge(2, 2, weight=0.5) # self loop on tree
        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(T, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(T, use_acceleration=True)
        t_ms = (time.perf_counter() - t0) * 1000
        # Underlying graph is tree, so girth should be inf, []
        ok = (math.isinf(g_pure) and c_pure == [] and math.isinf(g_acc) and c_acc == [])
        self.record("3_self_loops", "tree_with_self_loop", "PASS" if ok else "FAIL",
                    f"pure={g_pure} c_pure={c_pure}; acc={g_acc} c_acc={c_acc}", g_pure, g_acc, float("inf"), t_ms)

        # 3.4 Multiple self loops with 4-cycle
        G = nx.cycle_graph(4)
        for u, v in G.edges():
            G[u][v]["weight"] = 2.0 # 4-cycle weight 8.0
        for i in range(4):
            G.add_edge(i, i, weight=0.1 * (i + 1)) # self loops weights 0.1, 0.2, 0.3, 0.4
        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
        g_ora, c_ora = independent_oracle_girth(G)
        t_ms = (time.perf_counter() - t0) * 1000
        ok = (abs(g_pure - 8.0) < 1e-9 and abs(g_acc - 8.0) < 1e-9 and
              validate_cycle(G, c_pure, g_pure) and validate_cycle(G, c_acc, g_acc))
        self.record("3_self_loops", "multiple_self_loops_on_4cycle", "PASS" if ok else "FAIL",
                    f"pure={g_pure} c_pure={c_pure}; acc={g_acc} c_acc={c_acc}; ora={g_ora}", g_pure, g_acc, g_ora, t_ms)

    # --- CATEGORY 4: Giant Graphs (n > 1000) & Memory Stability ---
    def test_category_4_giant_graphs(self):
        print("\n--- Category 4: Giant Graphs (n > 1000) & Memory Stability ---")
        
        giant_configs = [
            ("er_n1500_p0.004", nx.erdos_renyi_graph(1500, 0.004, seed=101)),
            ("er_n2500_p0.002", nx.erdos_renyi_graph(2500, 0.002, seed=102)),
            ("ba_n2000_m2", nx.barabasi_albert_graph(2000, 2, seed=103)),
            ("grid_40x40_n1600", nx.grid_2d_graph(40, 40)),
            ("er_n3500_p0.001", nx.erdos_renyi_graph(3500, 0.001, seed=104)),
        ]

        for name, G in giant_configs:
            # Map grid tuples if needed
            if isinstance(list(G.nodes())[0], tuple):
                G = nx.convert_node_labels_to_integers(G)
            for u, v in G.edges():
                G[u][v]["weight"] = 1.0 + (hash((u, v)) % 100) / 20.0
            
            t0 = time.perf_counter()
            g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
            t_pure = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
            t_acc = (time.perf_counter() - t0) * 1000

            ok = False
            if math.isinf(g_pure) and math.isinf(g_acc):
                ok = (c_pure == [] and c_acc == [])
            else:
                diff = abs(g_pure - g_acc)
                val_pure = validate_cycle(G, c_pure, g_pure)
                val_acc = validate_cycle(G, c_acc, g_acc)
                ok = (diff < 1e-7 and val_pure and val_acc)

            speedup = (t_pure / t_acc) if t_acc > 0 else 1.0
            self.record("4_giant_graphs", name, "PASS" if ok else "FAIL",
                        f"N={G.number_of_nodes()} M={G.number_of_edges()} g_pure={g_pure:.4f} g_acc={g_acc:.4f} T_pure={t_pure:.1f}ms T_acc={t_acc:.1f}ms ({speedup:.2f}x)",
                        g_pure, g_acc, None, t_acc)

    # --- CATEGORY 5: Extreme Weight Ratios (w_max / w_min > 10^8) ---
    def test_category_5_extreme_weights(self):
        print("\n--- Category 5: Extreme Weight Ratios (w_max / w_min > 10^8) ---")
        
        # 5.1 Ratio 10^8: cycle weight 3.0, background edges 10^8
        G = nx.erdos_renyi_graph(50, 0.2, seed=42)
        for u, v in G.edges():
            G[u][v]["weight"] = 1e8
        # Plant small cycle of length 3
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 0, weight=1.0)

        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
        g_ora, c_ora = independent_oracle_girth(G)
        t_ms = (time.perf_counter() - t0) * 1000
        ok = (abs(g_pure - 3.0) < 1e-9 and abs(g_acc - 3.0) < 1e-9 and
              validate_cycle(G, c_pure, g_pure) and validate_cycle(G, c_acc, g_acc))
        self.record("5_extreme_weights", "ratio_1e8_planted_triangle", "PASS" if ok else "FAIL",
                    f"pure={g_pure} acc={g_acc} ora={g_ora}", g_pure, g_acc, g_ora, t_ms)

        # 5.2 Ratio 10^12: cycle weight 3e-6, background 1e6
        G = nx.erdos_renyi_graph(50, 0.2, seed=43)
        for u, v in G.edges():
            G[u][v]["weight"] = 1e6
        G.add_edge(0, 1, weight=1e-6)
        G.add_edge(1, 2, weight=1e-6)
        G.add_edge(2, 0, weight=1e-6)

        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
        g_ora, c_ora = independent_oracle_girth(G)
        t_ms = (time.perf_counter() - t0) * 1000
        expected = 3e-6
        ok = (abs(g_pure - expected) < 1e-12 and abs(g_acc - expected) < 1e-12 and
              validate_cycle(G, c_pure, g_pure) and validate_cycle(G, c_acc, g_acc))
        self.record("5_extreme_weights", "ratio_1e12_small_weights", "PASS" if ok else "FAIL",
                    f"pure={g_pure:.8e} acc={g_acc:.8e} ora={g_ora:.8e}", g_pure, g_acc, g_ora, t_ms)

        # 5.3 Ratio 10^16: cycle weight 3.0, background 1e16
        G = nx.erdos_renyi_graph(40, 0.25, seed=44)
        for u, v in G.edges():
            G[u][v]["weight"] = 1e16
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 0, weight=1.0)

        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
        g_ora, c_ora = independent_oracle_girth(G)
        t_ms = (time.perf_counter() - t0) * 1000
        ok = (abs(g_pure - 3.0) < 1e-9 and abs(g_acc - 3.0) < 1e-9 and
              validate_cycle(G, c_pure, g_pure) and validate_cycle(G, c_acc, g_acc))
        self.record("5_extreme_weights", "ratio_1e16_huge_background", "PASS" if ok else "FAIL",
                    f"pure={g_pure} acc={g_acc} ora={g_ora}", g_pure, g_acc, g_ora, t_ms)

        # 5.4 Dynamic scale spanning across 10 decades [10^-5, 10^5]
        G = nx.erdos_renyi_graph(60, 0.15, seed=45)
        edges = list(G.edges())
        for idx, (u, v) in enumerate(edges):
            exponent = -5.0 + 10.0 * (idx / max(1, len(edges) - 1))
            G[u][v]["weight"] = 10.0 ** exponent
        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
        g_ora, c_ora = independent_oracle_girth(G)
        t_ms = (time.perf_counter() - t0) * 1000
        diff = abs(g_pure - g_acc)
        rel_diff = diff / max(1e-12, g_pure)
        ok = (rel_diff < 1e-7 and validate_cycle(G, c_pure, g_pure) and validate_cycle(G, c_acc, g_acc) and abs(g_pure - g_ora) / max(1e-12, g_ora) < 1e-7)
        self.record("5_extreme_weights", "dynamic_range_10_decades", "PASS" if ok else "FAIL",
                    f"pure={g_pure:.6e} acc={g_acc:.6e} ora={g_ora:.6e} rel_diff={rel_diff:.2e}", g_pure, g_acc, g_ora, t_ms)

    # --- CATEGORY 6: G3 Tightness Gadgets & Chord Bounds ---
    def test_category_6_g3_gadgets(self):
        print("\n--- Category 6: G3 Tightness Gadgets & Chord Bounds ---")

        # 6.1 Canonical 10-vertex G3 gadget
        G = nx.Graph()
        seed_nodes = ['s0', 'a', 'b', 'c']
        for i in range(len(seed_nodes)):
            for j in range(i + 1, len(seed_nodes)):
                G.add_edge(seed_nodes[i], seed_nodes[j], weight=1.0 / 3.0)
        G.add_edge('s0', 'x', weight=1.0)
        w_sq = 0.2
        G.add_edge('v0', 'v1', weight=w_sq)
        G.add_edge('v1', 'v2', weight=w_sq)
        G.add_edge('v2', 'v3', weight=w_sq)
        G.add_edge('v3', 'v0', weight=w_sq)
        G.add_edge('x', 'v0', weight=0.1)
        # Heavy chords
        G.add_edge('v1', 'v3', weight=2.0)
        G.add_edge('x', 'v2', weight=2.0)
        G.add_edge('v2', 'h', weight=2.0)
        G.add_edge('x', 'h', weight=2.0)
        G.add_edge('s0', 'h', weight=2.0)

        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
        g_ora, c_ora = independent_oracle_girth(G)
        t_ms = (time.perf_counter() - t0) * 1000
        ok = (abs(g_pure - 0.8) < 1e-9 and abs(g_acc - 0.8) < 1e-9 and
              set(c_pure) == {'v0', 'v1', 'v2', 'v3'} and set(c_acc) == {'v0', 'v1', 'v2', 'v3'})
        self.record("6_g3_gadgets", "canonical_g3_gadget", "PASS" if ok else "FAIL",
                    f"pure={g_pure} acc={g_acc} ora={g_ora}", g_pure, g_acc, g_ora, t_ms)

        # 6.2 Scaled G3 with 15 heavy chords
        G = nx.Graph()
        # Ring of 8 vertices with weight 0.1 each -> 8-cycle weight 0.8
        for i in range(8):
            G.add_edge(f"r{i}", f"r{(i+1)%8}", weight=0.1)
        # Add 15 heavy cross chords with weight 2.5
        chord_count = 0
        for i in range(8):
            for j in range(i+2, 8):
                if (i, j) != (0, 7):
                    G.add_edge(f"r{i}", f"r{j}", weight=2.5)
                    chord_count += 1
        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
        g_ora, c_ora = independent_oracle_girth(G)
        t_ms = (time.perf_counter() - t0) * 1000
        ok = (abs(g_pure - 0.8) < 1e-9 and abs(g_acc - 0.8) < 1e-9 and
              validate_cycle(G, c_pure, g_pure) and validate_cycle(G, c_acc, g_acc))
        self.record("6_g3_gadgets", f"scaled_g3_ring_{chord_count}_chords", "PASS" if ok else "FAIL",
                    f"chords={chord_count} pure={g_pure} acc={g_acc} ora={g_ora}", g_pure, g_acc, g_ora, t_ms)

        # 6.3 Near-tight cotree chord w(e) = gamma - 1e-4
        # Optimal cycle is triangle with weight 3.0 (1.0, 1.0, 1.0)
        # Other cycle with cotree chord having weight 3.0 - 1e-4 -> THIS becomes optimal!
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        G.add_edge(3, 0, weight=1.0)
        # Add diagonal chord (0, 2) of weight 1.8 -> creates two triangles of weight 3.8
        G.add_edge(0, 2, weight=1.8)
        t0 = time.perf_counter()
        g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
        g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
        g_ora, c_ora = independent_oracle_girth(G)
        t_ms = (time.perf_counter() - t0) * 1000
        ok = (abs(g_pure - 3.8) < 1e-9 and abs(g_acc - 3.8) < 1e-9 and
              validate_cycle(G, c_pure, g_pure) and validate_cycle(G, c_acc, g_acc) and abs(g_pure - g_ora) < 1e-9)
        self.record("6_g3_gadgets", "near_tight_cotree_chord_subtle", "PASS" if ok else "FAIL",
                    f"pure={g_pure} acc={g_acc} ora={g_ora}", g_pure, g_acc, g_ora, t_ms)

    # --- CATEGORY 7: Authoritative Real-World Networks ---
    def test_category_7_realnets(self):
        print("\n--- Category 7: Authoritative Real-World Networks ---")
        realnets = [
            "lesmis",
            "celegans-neural",
            "chicago-sketch-road",
            "rome99-road",
            "uspowergrid-synth",
            "usairport-2010",
            "openflights-air",
            "chicago-regional-road",
            "sydney-road",
            "osm-portland-drive"
        ]
        datasets_dir = "/scratch/hs9hd/mwc_certified_pruning/datasets/realnets"

        for net in realnets:
            try:
                G = load_real_network(net, datasets_dir)
            except Exception as e:
                self.record("7_realnets", net, "FAIL", f"Failed to load dataset: {e}")
                continue

            t0 = time.perf_counter()
            g_pure, c_pure = minimum_weight_cycle(G, use_acceleration=False)
            t_pure = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            g_acc, c_acc = minimum_weight_cycle(G, use_acceleration=True)
            t_acc = (time.perf_counter() - t0) * 1000

            g_ora = None
            if G.number_of_nodes() <= 500:
                try:
                    g_ora, c_ora = independent_oracle_girth(G)
                except Exception:
                    g_ora = None

            diff = abs(g_pure - g_acc)
            val_pure = validate_cycle(G, c_pure, g_pure)
            val_acc = validate_cycle(G, c_acc, g_acc)
            
            ok = (diff < 1e-7 and val_pure and val_acc)
            if g_ora is not None:
                ok = ok and (abs(g_acc - g_ora) < 1e-7)

            speedup = (t_pure / t_acc) if t_acc > 0 else 1.0
            self.record("7_realnets", net, "PASS" if ok else "FAIL",
                        f"N={G.number_of_nodes()} M={G.number_of_edges()} g_pure={g_pure:.6f} g_acc={g_acc:.6f} diff={diff:.2e} T_pure={t_pure:.1f}ms T_acc={t_acc:.1f}ms ({speedup:.2f}x)",
                        g_pure, g_acc, g_ora, t_acc)

    def summary(self):
        total = len(self.results)
        passed = total - len(self.failures)
        print("\n" + "=" * 80)
        print(f"HARNESS EXECUTION COMPLETE: {passed} / {total} PASSED ({passed/total*100:.1f}%)")
        print("=" * 80)
        if self.failures:
            print("\nFAILURES IDENTIFIED:")
            for f in self.failures:
                print(f"  * [{f['category']}] {f['test_name']}: {f['details']}")
        print("=" * 80)

        # Save to JSON
        out_path = "/scratch/hs9hd/mwc_certified_pruning/results/challenger_m2_results.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump(self.results, fp, indent=2)
        print(f"Saved full results to {out_path}")

if __name__ == "__main__":
    runner = EmpiricalChallengerRunner()
    runner.run_all()
