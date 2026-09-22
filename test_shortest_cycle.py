import unittest
import os
import pickle
import networkx as nx
import numpy as np
from shortest_cycle import (
    minimum_weight_cycle,
    sota_shortest_cycle,
    shortest_cycle_nodes,
    traditional_shortest_cycle,
    create_grid_graph,
    create_spatial_graph,
    DynamicLCATree,
    LCATree,
    FibonacciHeap,
    dijkstra_base
)
from proposed_algorithm import proposed_algorithm


def _generate_random_tree(n: int, seed: int = 42) -> nx.Graph:
    """Generate a random tree compatible across NetworkX 2.x and 3.x."""
    if hasattr(nx, "random_labeled_tree"):
        return nx.random_labeled_tree(n, seed=seed)
    elif hasattr(nx, "random_tree"):
        return nx.random_tree(n, seed=seed)
    else:
        import random
        rng = random.Random(seed)
        G = nx.Graph()
        G.add_node(0)
        for i in range(1, n):
            parent = rng.randint(0, i - 1)
            G.add_edge(parent, i)
        return G


def _load_or_reconstruct_crashed_graph() -> nx.Graph:
    """
    Loads the serialized Challenger 1 counterexample graph if available,
    or programmatically reconstructs its exact topology and fractional weights.

    Graph specifications:
      - Model: Dense Erdős-Rényi G(n=25, p=0.8, seed=919993), |V|=25, |E|=243
      - Weights: Discrete fractional weights in {1/7, 0.25, 1/3, 0.5, 1.0, 2.5}
      - Girth (Oracle): 3/7 = 0.42857142857142855 (Triangle cycle: 5 - 20 - 9 - 5)
    """
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "../challenger_stress/crashed_graph.pkl"),
        os.path.join(os.path.dirname(__file__), "crashed_graph.pkl"),
        "/scratch/hs9hd/mwc_certified_pruning/challenger_stress/crashed_graph.pkl",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass

    # Programmatic reconstruction fallback (100% self-contained)
    G = nx.erdos_renyi_graph(25, 0.8, seed=919993)
    val_map = [1.0 / 7.0, 0.25, 1.0 / 3.0, 0.5, 1.0, 2.5]
    weight_indices = (
        "424520141034031424300505142403025224352422350024131304155152012045220031"
        "042532441540000221223350252020430552255223454350024322513442440523343141"
        "221050134441255224253235555125041154324025431212042115234404454312003145"
        "213535134250000200024321144"
    )
    for (u, v), ch in zip(G.edges(), weight_indices):
        G[u][v]["weight"] = val_map[int(ch)]
    return G


class TestShortestCycle(unittest.TestCase):
    
    def test_triangle_graph(self):
        """Test a simple triangle graph with different weights."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=2.0)
        G.add_edge(2, 0, weight=3.0)
        
        self.assertEqual(sota_shortest_cycle(G), 6.0)
        self.assertEqual(len(shortest_cycle_nodes(G)), 4)  # 3 nodes plus closing node
        
    def test_square_graph(self):
        """Test a square graph with uniform weights."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        G.add_edge(3, 0, weight=1.0)
        
        self.assertEqual(sota_shortest_cycle(G), 4.0)
        cycle = shortest_cycle_nodes(G)
        self.assertEqual(len(cycle), 5)  # 4 nodes plus closing node
        
    def test_square_with_diagonal(self):
        """Test a square graph with a diagonal edge creating two triangles."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        G.add_edge(3, 0, weight=1.0)
        G.add_edge(0, 2, weight=1.5)  # Diagonal
        
        self.assertEqual(sota_shortest_cycle(G), 3.5)  # Diagonal plus one edge
        
    def test_pentagon_graph(self):
        """Test a pentagon graph with non-uniform weights."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=2.0)
        G.add_edge(2, 3, weight=3.0)
        G.add_edge(3, 4, weight=2.0)
        G.add_edge(4, 0, weight=1.0)
        G.add_edge(0, 3, weight=2.5)  # Shortcut
        
        self.assertEqual(sota_shortest_cycle(G), 5.5)  # 0 -> 1 -> 2 -> 3 -> 0
        
    def test_complete_graph(self):
        """Test a complete graph where every node connects to every other node."""
        G = nx.complete_graph(5)
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
            
        self.assertEqual(sota_shortest_cycle(G), 3.0)
        
    def test_no_cycle(self):
        """Test a tree graph which has no cycles."""
        G = _generate_random_tree(10, seed=42)
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
            
        self.assertIsNone(sota_shortest_cycle(G))
        self.assertIsNone(shortest_cycle_nodes(G))
        
    def test_disconnected_graph(self):
        """Test a disconnected graph with cycles in components."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=2.0)
        G.add_edge(2, 0, weight=3.0)
        
        G.add_edge(3, 4, weight=1.0)
        G.add_edge(4, 5, weight=1.0)
        G.add_edge(5, 6, weight=1.0)
        G.add_edge(6, 3, weight=1.0)
        
        self.assertEqual(sota_shortest_cycle(G), 4.0)
        
    def test_single_node(self):
        """Test a graph with a single node."""
        G = nx.Graph()
        G.add_node(0)
        
        self.assertIsNone(sota_shortest_cycle(G))
        
    def test_empty_graph(self):
        """Test an empty graph."""
        G = nx.Graph()
        
        self.assertIsNone(sota_shortest_cycle(G))
        
    def test_negative_weights(self):
        """Test that the algorithm raises an error for negative weights."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=-1.0)
        G.add_edge(2, 0, weight=3.0)
        
        with self.assertRaises(ValueError):
            sota_shortest_cycle(G)
            
    def test_directed_graph(self):
        """Test that the algorithm raises an error for directed graphs."""
        G = nx.DiGraph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=2.0)
        G.add_edge(2, 0, weight=3.0)
        
        with self.assertRaises(ValueError):
            sota_shortest_cycle(G)
            
    def test_large_random_graph(self):
        """Test the algorithm on a larger random graph."""
        G = nx.gnm_random_graph(100, 200, seed=42)
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
            
        result = sota_shortest_cycle(G)
        self.assertIsNotNone(result)
        
    def test_grid_graph(self):
        """Test on a grid graph which has many equal-length cycles."""
        G = nx.grid_2d_graph(3, 3)
        G = nx.convert_node_labels_to_integers(G)
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
            
        self.assertEqual(sota_shortest_cycle(G), 4.0)
    
    def test_grid_graph_with_hidden_cycle(self):
        """Test a grid graph with a hidden low-weight cycle."""
        G = create_grid_graph(5, default_weight=10, cycle_weight=1)
        self.assertEqual(sota_shortest_cycle(G), 4.0)
    
    def test_spatial_graph_with_hidden_cycle(self):
        """Test a spatial graph with a hidden low-weight cycle."""
        np.random.seed(42)
        G = create_spatial_graph(30, radius=0.3, default_weight=10, cycle_weight=1)
        self.assertEqual(sota_shortest_cycle(G), 4.0)
    
    def test_optimized_vs_traditional(self):
        """Test that optimized algorithm gives same results as traditional approach."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=2.0)
        G.add_edge(1, 2, weight=3.0)
        G.add_edge(2, 3, weight=1.0)
        G.add_edge(3, 0, weight=4.0)
        G.add_edge(0, 2, weight=5.0)
        
        optimized_result = sota_shortest_cycle(G)
        
        traditional_result = None
        shortest_cycle_length = float("inf")
        
        for u, v in G.edges():
            weight = G[u][v].get("weight", 1.0)
            G.remove_edge(u, v)
            try:
                path_length = nx.shortest_path_length(G, u, v, weight="weight")
                cycle_length = path_length + weight
                if cycle_length < shortest_cycle_length:
                    shortest_cycle_length = cycle_length
            except nx.NetworkXNoPath:
                pass
            G.add_edge(u, v, weight=weight)
        
        if shortest_cycle_length != float("inf"):
            traditional_result = shortest_cycle_length
        
        self.assertEqual(optimized_result, traditional_result)


class TestCertifiedPruningFeatures(unittest.TestCase):
    """Isolated unit tests for each of the 8 certified pruning features."""

    # --- Feature 1: Multigraph 2-Cycles ---
    def test_feature1_multigraph_parallel_2cycle_minimum(self):
        """Test that a multigraph with parallel edges finds the 2-cycle when it is the MWC."""
        G = nx.MultiGraph()
        G.add_edge(0, 1, key='e01', weight=10.0)
        G.add_edge(1, 2, key='e12_a', weight=2.0)
        G.add_edge(1, 2, key='e12_b', weight=3.0)
        G.add_edge(2, 3, key='e23', weight=10.0)
        G.add_edge(3, 0, key='e30', weight=10.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 5.0)
        self.assertIn(list(cycle), [[1, 2, 1], [2, 1, 2]])

    def test_feature1_multigraph_distant_root_cotree_collapse(self):
        """Candidate 18 gadget: distant root must not inflate parallel edge 2-cycle."""
        G = nx.MultiGraph()
        G.add_edge('x', 'u', key='xu', weight=5.0)
        G.add_edge('x', 'v', key='xv', weight=5.0)
        G.add_edge('u', 'v', key='uv1', weight=1.0)
        G.add_edge('u', 'v', key='uv2', weight=1.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 2.0)
        self.assertIn(list(cycle), [['u', 'v', 'u'], ['v', 'u', 'v']])

    def test_feature1_multigraph_zero_weight_2cycle_immediate_halt(self):
        """Test that a multigraph with two weight 0.0 parallel edges immediately returns 0.0."""
        G = nx.MultiGraph()
        G.add_edge(0, 1, key='a', weight=0.0)
        G.add_edge(0, 1, key='b', weight=0.0)
        G.add_edge(1, 2, key='c', weight=10.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 0.0)
        self.assertEqual(len(cycle), 3)

    def test_feature1_multigraph_parallel_edges_not_minimum(self):
        """Test that a triangle of weight 3.0 wins over a heavy 2-cycle of weight 7.0."""
        G = nx.MultiGraph()
        G.add_edge(0, 1, key='t1', weight=1.0)
        G.add_edge(1, 2, key='t2', weight=1.0)
        G.add_edge(2, 0, key='t3', weight=1.0)
        G.add_edge(0, 1, key='heavy', weight=6.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 3.0)
        self.assertEqual(len(cycle), 4)

    # --- Feature 2: Zero-Weight Cycles & Trees ---
    def test_feature2_zero_weight_cycle_triangle(self):
        """Test graph with a zero-weight 3-cycle embedded among positive edges."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=0.0)
        G.add_edge(1, 2, weight=0.0)
        G.add_edge(2, 0, weight=0.0)
        G.add_edge(2, 3, weight=5.0)
        G.add_edge(3, 4, weight=5.0)
        G.add_edge(4, 2, weight=5.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 0.0)
        self.assertEqual(len(cycle), 4)
        resum = sum(G[cycle[i]][cycle[i + 1]]['weight'] for i in range(len(cycle) - 1))
        self.assertEqual(resum, 0.0)

    def test_feature2_zero_weight_trees_with_positive_cycle(self):
        """Test zero-weight tree edges leading to a positive-weight cycle."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=0.0)
        G.add_edge(1, 2, weight=0.0)
        G.add_edge(2, 3, weight=0.0)
        G.add_edge(3, 4, weight=1.0)
        G.add_edge(4, 5, weight=1.0)
        G.add_edge(5, 3, weight=1.0)
        G.add_edge(0, 6, weight=4.0)
        G.add_edge(6, 7, weight=4.0)
        G.add_edge(7, 0, weight=4.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 3.0)
        self.assertEqual(set(cycle), {3, 4, 5})

    def test_feature2_zero_weight_forest_acyclic(self):
        """Test graph where zero-weight edges form an acyclic forest, no zero-cycle."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=0.0)
        G.add_edge(2, 3, weight=0.0)
        G.add_edge(1, 2, weight=2.5)
        G.add_edge(3, 0, weight=2.5)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 5.0)
        self.assertEqual(len(cycle), 5)

    # --- Feature 3: Strict Lexicographical Tie-Breaking ---
    def test_feature3_lexicographical_tiebreaking_zero_weight_path(self):
        """Test that zero-weight edge between two nodes at identical distance does not form a loop."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(0, 2, weight=1.0)
        G.add_edge(1, 2, weight=0.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 2.0)
        self.assertEqual(len(cycle), 4)

    def test_feature3_lexicographical_tiebreaking_complete_bipartite_k33(self):
        """Test K_{3,3} with unit weights: distance ties must produce acyclic arborescences."""
        G = nx.complete_bipartite_graph(3, 3)
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
            
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 4.0)
        self.assertEqual(len(cycle), 5)

    def test_feature3_lexicographical_tiebreaking_hypercube_q3(self):
        """Test 3-dimensional hypercube (Q_3): uniform 4-cycles with extensive distance ties."""
        G = nx.hypercube_graph(3)
        G = nx.convert_node_labels_to_integers(G)
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
            
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 4.0)
        self.assertEqual(len(cycle), 5)

    # --- Feature 4: Frontier Sentinels ---
    def test_feature4_frontier_sentinel_exhausted_tree_component(self):
        """Test sentinel branch B_x = infinity on exhausted acyclic tree component."""
        G = nx.Graph()
        for i in range(5):
            G.add_edge(f"t{i}", f"t{i+1}", weight=1.0)
        G.add_edge("c0", "c1", weight=2.0)
        G.add_edge("c1", "c2", weight=2.0)
        G.add_edge("c2", "c0", weight=2.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 6.0)
        self.assertEqual(set(cycle), {"c0", "c1", "c2"})

    def test_feature4_frontier_sentinel_clamping_prevents_overpruning(self):
        """Candidate 02 counterexample: cross-edges to unsettled nodes must clamp tau_x = R."""
        G = nx.Graph()
        for i in range(10):
            G.add_edge(f"A{i}", f"A{(i+1)%10}", weight=1.0)
        G.add_edge("x", "v1", weight=2.0)
        G.add_edge("x", "v2", weight=4.0)
        G.add_edge("v1", "v_far", weight=6.0)
        G.add_edge("v2", "v3", weight=1.0)
        G.add_edge("v3", "v_far", weight=1.0)
        G.add_edge("v_far", "v2", weight=1.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 3.0)

    # --- Feature 5: Dynamic Certified Pruning & Scheduling ---
    def test_feature5_unsafe_pruning_regression_graph(self):
        """Canonical audit counterexample against legacy heuristic (dist + 2*w_min >= gamma)."""
        G = nx.Graph()
        for i in range(10):
            G.add_edge(f"A{i}", f"A{(i+1)%10}", weight=1.0)
        G.add_edge("x", "a", weight=8.0)
        G.add_edge("x", "b", weight=8.0)
        G.add_edge("x", "c", weight=8.0)
        G.add_edge("a", "b", weight=1.0)
        G.add_edge("b", "c", weight=1.0)
        G.add_edge("c", "a", weight=1.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 3.0)
        self.assertEqual(set(cycle), {"a", "b", "c"})

    def test_feature5_exhaustive_root_scheduling_disjoint_components(self):
        """Test 4 disconnected components with cycles of lengths 20, 15, 10, 5."""
        G = nx.Graph()
        for i in range(4):
            G.add_edge(f"c1_{i}", f"c1_{(i+1)%4}", weight=5.0)
        for i in range(3):
            G.add_edge(f"c2_{i}", f"c2_{(i+1)%3}", weight=5.0)
        for i in range(5):
            G.add_edge(f"c3_{i}", f"c3_{(i+1)%5}", weight=2.0)
        G.add_edge("c4_0", "c4_1", weight=1.0)
        G.add_edge("c4_1", "c4_2", weight=2.0)
        G.add_edge("c4_2", "c4_0", weight=2.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 5.0)
        self.assertEqual(set(cycle), {"c4_0", "c4_1", "c4_2"})

    def test_feature5_approximation_factor_k_soundness(self):
        """Test certified pruning with approximation factor K=2.0, K=1.5, K=1.0."""
        G = nx.erdos_renyi_graph(15, 0.4, seed=123)
        for u, v in G.edges():
            G[u][v]['weight'] = float((u * v) % 7 + 1)
            
        exact_len, _ = minimum_weight_cycle(G, method='exact_oracle', weight='weight')
        
        for K_val in [1.0, 1.5, 2.0]:
            k_len, k_cycle = minimum_weight_cycle(G, K=K_val, weight='weight')
            self.assertGreaterEqual(k_len, exact_len - 1e-9)
            self.assertLessEqual(k_len, K_val * exact_len + 1e-9)

    # --- Feature 6: Historical Certificate Accumulation & Lazy Skipping ---
    def test_feature6_historical_certificate_accumulation_monotonicity(self):
        """Verify that certificate accumulation preserves cycle detection."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        G.add_edge(3, 0, weight=1.0)
        G.add_edge(2, 4, weight=3.0)
        G.add_edge(4, 5, weight=3.0)
        G.add_edge(5, 3, weight=1.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 4.0)

    def test_feature6_lazy_root_skipping_preserves_optimality(self):
        """Verify that lazy skipping achieves exact results on barbell graph."""
        G = nx.barbell_graph(6, 2)
        for u, v in G.edges():
            G[u][v]['weight'] = 2.0
        G[0][1]['weight'] = 1.0
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 5.0)

    # --- Feature 7: Heavy-Edge Filtering & Structural Countermeasures ---
    def test_feature7_heavy_edge_filtering_gadget_g3(self):
        """Test 10-vertex gadget G_3: heavy-edge filtering removes W=2 chords."""
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
        
        G.add_edge('v1', 'v3', weight=2.0)
        G.add_edge('x', 'v2', weight=2.0)
        G.add_edge('v2', 'h', weight=2.0)
        G.add_edge('x', 'h', weight=2.0)
        G.add_edge('s0', 'h', weight=2.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertAlmostEqual(length, 0.8, places=6)
        self.assertEqual(set(cycle), {'v0', 'v1', 'v2', 'v3'})

    def test_feature7_heavy_edge_filtering_dynamic_pruning(self):
        """Test that heavy edges are discarded dynamically when a tighter cycle is found."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=2.5)
        G.add_edge(1, 2, weight=2.5)
        G.add_edge(2, 3, weight=2.5)
        G.add_edge(3, 0, weight=2.5)
        G.add_edge(0, 2, weight=8.0)
        G.add_edge(2, 4, weight=1.0)
        G.add_edge(4, 5, weight=1.0)
        G.add_edge(5, 2, weight=1.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 3.0)

    # --- Feature 8: Parallel Snapshot Isolation ---
    def test_feature8_parallel_snapshot_isolation_matches_sequential(self):
        """Test that parallel execution produces exact match with sequential execution."""
        rng = np.random.RandomState(42)
        G = nx.erdos_renyi_graph(25, 0.3, seed=42)
        for u, v in G.edges():
            G[u][v]['weight'] = float(rng.randint(1, 20))
            
        seq_len, seq_cycle = minimum_weight_cycle(G, parallel=False, weight='weight')
        par_len_2, par_cycle_2 = minimum_weight_cycle(G, parallel=True, n_workers=2, weight='weight')
        par_len_4, par_cycle_4 = minimum_weight_cycle(G, parallel=True, n_workers=4, weight='weight')
        
        self.assertAlmostEqual(seq_len, par_len_2, places=7)
        self.assertAlmostEqual(seq_len, par_len_4, places=7)
        if seq_len < float('inf'):
            self.assertEqual(len(par_cycle_2), len(seq_cycle))
            self.assertEqual(len(par_cycle_4), len(seq_cycle))

    def test_feature8_parallel_epoch_barrier_thread_safety(self):
        """Test parallel execution on a symmetric multi-component graph with concurrent deletions."""
        G = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(6))
        G = nx.disjoint_union(G, nx.cycle_graph(7))
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
            
        par_len, par_cycle = minimum_weight_cycle(G, parallel=True, n_workers=4, weight='weight')
        self.assertEqual(par_len, 5.0)
        self.assertEqual(len(par_cycle), 6)


class TestLegacyBugRegressions(unittest.TestCase):
    """Explicit regressions verifying legacy defects in girth are eradicated."""

    def test_legacy_bug_falsy_lca_node_zero(self):
        """Test that a fundamental cycle whose LCA is node 0 is not dropped."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 0, weight=1.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 3.0)
        self.assertIn(0, cycle)

    def test_legacy_bug_falsy_node_zero_interior_lca(self):
        """Test where vertex 0 is an interior LCA for nodes 2 and 4."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(0, 3, weight=1.0)
        G.add_edge(3, 4, weight=1.0)
        G.add_edge(2, 4, weight=1.0)
        G.add_edge(0, 5, weight=20.0)
        G.add_edge(5, 6, weight=20.0)
        G.add_edge(6, 0, weight=20.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 5.0)
        self.assertIn(0, cycle)

    def test_legacy_bug_ancestor_descendant_chord(self):
        """Test cotree edge where one endpoint is an ancestor of the other (p = u)."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        G.add_edge(0, 3, weight=10.0)
        
        length, cycle = minimum_weight_cycle(G, weight='weight')
        self.assertEqual(length, 13.0)

    def test_regression_in_loop_mutation_crashed_graph(self):
        """
        Regression test: Verify certified pruning on the Challenger 1 counterexample graph
        executes without RuntimeError (dictionary changed size during iteration) and matches
        the exact oracle girth and valid simple cycle structure.
        """
        G = _load_or_reconstruct_crashed_graph()
        self.assertEqual(G.number_of_nodes(), 25)
        self.assertEqual(G.number_of_edges(), 243)

        # Certified pruning execution (K=1.0, sequential)
        cp_length, cp_cycle = minimum_weight_cycle(G, K=1.0, weight='weight', parallel=False)

        # Oracle ground truth
        oracle_length, oracle_cycle = minimum_weight_cycle(G, method='exact_oracle', weight='weight')

        # Assertions
        expected_girth = 3.0 / 7.0
        self.assertAlmostEqual(oracle_length, expected_girth, places=7)
        self.assertAlmostEqual(cp_length, oracle_length, places=7)

        # Validate cycle structure
        self.assertIsNotNone(cp_cycle)
        self.assertGreaterEqual(len(cp_cycle), 4)
        self.assertEqual(cp_cycle[0], cp_cycle[-1])
        cycle_edges_weight = sum(G[u][v]['weight'] for u, v in zip(cp_cycle[:-1], cp_cycle[1:]))
        self.assertAlmostEqual(cycle_edges_weight, cp_length, places=7)

        # Also verify parallel mode matches oracle
        par_length, par_cycle = minimum_weight_cycle(G, K=1.0, weight='weight', parallel=True, n_workers=2)
        self.assertAlmostEqual(par_length, oracle_length, places=7)

    def test_regression_random_weighted_graphs_no_mutation_crash(self):
        """
        Regression test: Sweep 50 seeded random Erdős-Rényi graphs with uniform weights in [1.0, 10.0].
        Verifies 0 unhandled exceptions / iteration mutation crashes across all instances and
        verifies differential equivalence against exact oracle.
        """
        crashes = []
        max_diff = 0.0

        for seed in range(50):
            rng = np.random.RandomState(seed)
            G = nx.erdos_renyi_graph(20, 0.4, seed=seed)
            for u, v in G.edges():
                G[u][v]['weight'] = float(rng.uniform(1.0, 10.0))

            try:
                cp_len, cp_cycle = minimum_weight_cycle(G, K=1.0, weight='weight', parallel=False)
            except Exception as e:
                crashes.append((seed, type(e).__name__, str(e)))
                continue

            # Verify valid return
            self.assertTrue(np.isfinite(cp_len) or cp_len == float('inf'))
            if np.isfinite(cp_len) and cp_cycle is not None:
                self.assertGreaterEqual(len(cp_cycle), 4)
                self.assertEqual(cp_cycle[0], cp_cycle[-1])
                cw = sum(G[u][v]['weight'] for u, v in zip(cp_cycle[:-1], cp_cycle[1:]))
                self.assertAlmostEqual(cw, cp_len, places=7)

            # Differential check against exact oracle
            or_len, or_cycle = minimum_weight_cycle(G, method='exact_oracle', weight='weight')
            diff = abs(cp_len - or_len)
            if diff > max_diff:
                max_diff = diff
            self.assertAlmostEqual(cp_len, or_len, places=7,
                                   msg=f"Discrepancy on seed {seed}: cp={cp_len}, oracle={or_len}")

        self.assertEqual(len(crashes), 0, f"Encountered {len(crashes)} crashes: {crashes}")
        self.assertLess(max_diff, 1e-7)


class TestPublicAPIContract(unittest.TestCase):
    """Tests validating parameter boundaries and method consistency."""

    def test_invalid_parameters_raise_appropriate_errors(self):
        G = nx.cycle_graph(4)
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
        with self.assertRaises(ValueError):
            minimum_weight_cycle(G, K=0.5)
        with self.assertRaises(ValueError):
            minimum_weight_cycle(G, method='unsupported_method')
        G_neg = nx.Graph()
        G_neg.add_edge(0, 1, weight=-1.0)
        with self.assertRaises(ValueError):
            minimum_weight_cycle(G_neg)
        G_di = nx.DiGraph()
        G_di.add_edge(0, 1, weight=1.0)
        with self.assertRaises(ValueError):
            minimum_weight_cycle(G_di)

    def test_methods_consistency(self):
        G = nx.erdos_renyi_graph(12, 0.4, seed=99)
        for u, v in G.edges():
            G[u][v]['weight'] = float((u + v) % 5 + 1)
        cp_len, cp_cyc = minimum_weight_cycle(G, method='certified_pruning', K=1.0)
        or_len, or_cyc = minimum_weight_cycle(G, method='exact_oracle')
        dijk_len, dijk_cyc = minimum_weight_cycle(G, method='dijkstra')
        self.assertAlmostEqual(cp_len, or_len)
        self.assertAlmostEqual(cp_len, dijk_len)


class TestProposedAlgorithm(unittest.TestCase):
    """Tests for proposed_algorithm.py interface and behavior."""

    def test_proposed_algorithm_cycle(self):
        G = nx.cycle_graph(4)
        for u, v in G.edges():
            G[u][v]['weight'] = 2.5
        gamma, ops = proposed_algorithm(G)
        self.assertEqual(gamma, 10.0)
        self.assertGreater(ops, 0)

    def test_proposed_algorithm_acyclic(self):
        T = nx.path_graph(6)
        for u, v in T.edges():
            T[u][v]['weight'] = 1.0
        gamma, ops = proposed_algorithm(T)
        self.assertEqual(gamma, float('inf'))
        self.assertGreater(ops, 0)

    def test_proposed_algorithm_invalid_graph(self):
        G_di = nx.DiGraph()
        G_di.add_edge(0, 1, weight=1.0)
        with self.assertRaises(ValueError):
            proposed_algorithm(G_di)


class TestAuxiliaryStructures(unittest.TestCase):
    """Tests for auxiliary data structures and backward compatibility."""

    def test_dynamic_lca_tree_depth_and_lifting(self):
        nodes = [0, 1, 2, 3, 4]
        parent = {1: 0, 2: 1, 3: 0, 4: 3}
        tree = DynamicLCATree(nodes, parent, root=0)
        self.assertEqual(tree.lca(2, 4), 0)
        self.assertEqual(tree.lca(2, 1), 1)
        self.assertEqual(tree.lca(1, 3), 0)
        self.assertEqual(tree.lca(4, 3), 3)

    def test_legacy_lca_tree_compatibility(self):
        nodes = [0, 1, 2]
        parent = {1: 0, 2: 1}
        depth = {0: 0, 1: 1, 2: 2}
        tree = LCATree(parent, depth, nodes)
        self.assertEqual(tree.lca(1, 2), 1)
        stats = tree.get_stats()
        self.assertIn('query_count', stats)

    def test_fibonacci_heap_operations(self):
        heap = FibonacciHeap()
        heap.push('item1', 10)
        heap.push('item2', 5)
        heap.push('item3', 15)
        p, item = heap.pop()
        self.assertEqual(item, 'item2')
        self.assertEqual(p, 5)

    def test_traditional_shortest_cycle_stats(self):
        G = nx.cycle_graph(3)
        for u, v in G.edges():
            G[u][v]['weight'] = 2.0
        length, stats = traditional_shortest_cycle(G, return_stats=True)
        self.assertEqual(length, 6.0)
        self.assertIn('execution_time', stats)


# ---------------------------------------------------------------------------
# Milestone 2 Iteration 2: Self-Loop Filtering & Degeneracy Invariant Tests
# ---------------------------------------------------------------------------

class TestSelfLoopDegeneracy(unittest.TestCase):
    """
    Dedicated test suite for Milestone 2 Iteration 2 verifying that self-loops (u, u)
    are strictly filtered out during degeneracy preprocessing and never returned as
    minimum-weight cycles. Simple cycles (>= 3 nodes), multigraph 2-cycles (distinct u != v),
    or acyclic results (inf, []) must be returned.
    """

    def test_self_loop_and_triangle_prefers_triangle(self):
        """Challenger 3.1: self-loop (0.1) on node 0 with triangle (3.0) on 1, 2, 3 must return triangle."""
        G = nx.Graph()
        G.add_edge(0, 0, weight=0.1) # Self-loop
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        G.add_edge(3, 1, weight=1.0) # Triangle weight 3.0

        length, cycle = minimum_weight_cycle(G, weight="weight")
        self.assertAlmostEqual(length, 3.0, places=7)
        self.assertIsNotNone(cycle)
        self.assertEqual(len(cycle), 4)
        self.assertEqual(cycle[0], cycle[-1])
        self.assertEqual(set(cycle[:-1]), {1, 2, 3})
        self.assertNotIn(0, cycle)

    def test_multigraph_self_loop_and_parallel_2cycle(self):
        """Challenger 3.2: multigraph with self-loop (0.05) and parallel 2-cycle (2.2) must return 2-cycle."""
        MG = nx.MultiGraph()
        MG.add_edge(0, 0, weight=0.05) # Self-loop
        MG.add_edge(1, 2, weight=1.0)  # Parallel edges forming 2-cycle weight 2.2
        MG.add_edge(1, 2, weight=1.2)
        MG.add_edge(3, 4, weight=2.0)  # Triangle weight 6.0
        MG.add_edge(4, 5, weight=2.0)
        MG.add_edge(5, 3, weight=2.0)

        length, cycle = minimum_weight_cycle(MG, weight="weight")
        self.assertAlmostEqual(length, 2.2, places=7)
        self.assertIn(list(cycle), [[1, 2, 1], [2, 1, 2]])
        self.assertNotIn(0, cycle)

    def test_tree_with_self_loop_returns_inf(self):
        """Challenger 3.3: path tree P5 with self-loop (0.5) must return (inf, [])."""
        T = nx.path_graph(5)
        T.add_edge(2, 2, weight=0.5) # Self-loop on node 2

        length, cycle = minimum_weight_cycle(T, weight="weight")
        self.assertEqual(length, float("inf"))
        self.assertEqual(cycle, [])

    def test_multiple_self_loops_on_4cycle_prefers_4cycle(self):
        """Challenger 3.4: 4-cycle (weight 8.0) with self-loops on each node must return 4-cycle."""
        G = nx.cycle_graph(4)
        for u, v in G.edges():
            G[u][v]["weight"] = 2.0 # 4-cycle weight 8.0
        for i in range(4):
            G.add_edge(i, i, weight=0.1 * (i + 1)) # Self-loops: 0.1, 0.2, 0.3, 0.4

        length, cycle = minimum_weight_cycle(G, weight="weight")
        self.assertAlmostEqual(length, 8.0, places=7)
        self.assertEqual(len(cycle), 5)
        self.assertEqual(cycle[0], cycle[-1])
        self.assertEqual(set(cycle[:-1]), {0, 1, 2, 3})

    def test_graph_with_only_self_loops_returns_inf(self):
        """Graph with only self-loops has no simple cycles of length >= 2; must return (inf, [])."""
        G = nx.Graph()
        G.add_edge(0, 0, weight=1.0)
        G.add_edge(1, 1, weight=2.0)
        G.add_edge(2, 2, weight=3.0)

        length, cycle = minimum_weight_cycle(G, weight="weight")
        self.assertEqual(length, float("inf"))
        self.assertEqual(cycle, [])

    def test_multigraph_multiple_self_loops_same_node_not_2cycle(self):
        """Multiple parallel self-loops on the same node must NOT form a 2-cycle [u, u, u]."""
        MG = nx.MultiGraph()
        MG.add_edge(0, 0, key="s1", weight=0.1)
        MG.add_edge(0, 0, key="s2", weight=0.2)
        MG.add_edge(1, 2, weight=2.0)
        MG.add_edge(2, 3, weight=2.0)
        MG.add_edge(3, 1, weight=2.0)

        length, cycle = minimum_weight_cycle(MG, weight="weight")
        self.assertAlmostEqual(length, 6.0, places=7)
        self.assertEqual(set(cycle[:-1]), {1, 2, 3})
        self.assertNotIn(0, cycle)

    def test_zero_weight_self_loop_ignored(self):
        """Zero-weight self-loop (0.0) must not trigger a false zero-weight cycle return."""
        G = nx.Graph()
        G.add_edge(0, 0, weight=0.0)
        G.add_edge(1, 2, weight=1.5)
        G.add_edge(2, 3, weight=1.5)
        G.add_edge(3, 1, weight=1.5)

        length, cycle = minimum_weight_cycle(G, weight="weight")
        self.assertAlmostEqual(length, 4.5, places=7)
        self.assertEqual(set(cycle[:-1]), {1, 2, 3})

        T = nx.path_graph(4)
        T.add_edge(1, 1, weight=0.0)
        length_t, cycle_t = minimum_weight_cycle(T, weight="weight")
        self.assertEqual(length_t, float("inf"))
        self.assertEqual(cycle_t, [])

    def test_self_loop_on_shortest_cycle_vertex(self):
        """Self-loop attached directly to an active MWC vertex must be safely ignored during search."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 0, weight=1.0)
        G.add_edge(0, 0, weight=0.01) # Self-loop directly on vertex 0

        length, cycle = minimum_weight_cycle(G, weight="weight")
        self.assertAlmostEqual(length, 3.0, places=7)
        self.assertEqual(set(cycle[:-1]), {0, 1, 2})
        self.assertEqual(len(cycle), 4)

    def test_preprocess_degeneracy_self_loop_filtering(self):
        """Direct white-box contract: preprocess_degeneracy must never output self-loops in G_simple."""
        from shortest_cycle import preprocess_degeneracy

        MG = nx.MultiGraph()
        MG.add_edge(0, 0, key="loop0", weight=0.1)
        MG.add_edge(1, 1, key="loop1", weight=0.2)
        MG.add_edge(0, 1, key="e01", weight=2.0)
        MG.add_edge(1, 2, key="e12", weight=2.0)
        MG.add_edge(2, 0, key="e20", weight=2.0)

        G_simple, gamma_0, cycle_0, is_done = preprocess_degeneracy(MG, weight="weight")
        self.assertFalse(any(u == v for u, v in G_simple.edges()))
        self.assertNotEqual(gamma_0, 0.1)
        self.assertNotEqual(gamma_0, 0.2)
        if cycle_0 is not None:
            self.assertGreaterEqual(len(cycle_0), 3)
            self.assertNotEqual(cycle_0[0], cycle_0[1])

    def test_self_loop_all_methods_consistency(self):
        """Cross-verify all public methods (certified_pruning, dijkstra, exact_oracle) on self-loops."""
        G = nx.Graph()
        G.add_edge(0, 0, weight=0.1)
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        G.add_edge(3, 1, weight=1.0)

        l_cp, c_cp = minimum_weight_cycle(G, method="certified_pruning")
        l_dijk, c_dijk = minimum_weight_cycle(G, method="dijkstra")
        l_ora, c_ora = minimum_weight_cycle(G, method="exact_oracle")
        sota_len = sota_shortest_cycle(G)
        cycle_nodes = shortest_cycle_nodes(G)
        prop_len, _ = proposed_algorithm(G)

        for l in [l_cp, l_dijk, l_ora, sota_len, prop_len]:
            self.assertAlmostEqual(l, 3.0, places=7)
        self.assertEqual(set(c_cp[:-1]), {1, 2, 3})
        self.assertEqual(set(c_dijk[:-1]), {1, 2, 3})
        self.assertEqual(set(c_ora[:-1]), {1, 2, 3})
        self.assertEqual(set(cycle_nodes[:-1]), {1, 2, 3})


if __name__ == '__main__':
    unittest.main()
