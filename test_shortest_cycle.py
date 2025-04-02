import unittest
import networkx as nx
import numpy as np
from shortest_cycle import (
    sota_shortest_cycle, shortest_cycle_nodes, 
    create_grid_graph, create_spatial_graph
)

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
        # Set all weights to 1.0
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
            
        self.assertEqual(sota_shortest_cycle(G), 3.0)  # Shortest cycle in complete graph is a triangle
        
    def test_no_cycle(self):
        """Test a tree graph which has no cycles."""
        G = nx.random_tree(10, seed=42)
        # Set all weights to 1.0
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
            
        self.assertIsNone(sota_shortest_cycle(G))
        self.assertIsNone(shortest_cycle_nodes(G))
        
    def test_disconnected_graph(self):
        """Test a disconnected graph with cycles in components."""
        G = nx.Graph()
        # Component 1: Triangle
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=2.0)
        G.add_edge(2, 0, weight=3.0)
        
        # Component 2: Square
        G.add_edge(3, 4, weight=1.0)
        G.add_edge(4, 5, weight=1.0)
        G.add_edge(5, 6, weight=1.0)
        G.add_edge(6, 3, weight=1.0)
        
        self.assertEqual(sota_shortest_cycle(G), 4.0)  # Square is shorter than triangle
        
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
        G.add_edge(1, 2, weight=-1.0)  # Negative weight
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
        # Generate a random graph with 100 nodes and 200 edges
        G = nx.gnm_random_graph(100, 200, seed=42)
        # Assign random weights
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
            
        # Just ensure it runs without error and returns something reasonable
        result = sota_shortest_cycle(G)
        # Most random graphs will have cycles
        self.assertIsNotNone(result)
        
    def test_grid_graph(self):
        """Test on a grid graph which has many equal-length cycles."""
        G = nx.grid_2d_graph(3, 3)
        # Convert to integer node labels
        G = nx.convert_node_labels_to_integers(G)
        # Assign weights
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
            
        self.assertEqual(sota_shortest_cycle(G), 4.0)  # Smallest cell in the grid
    
    def test_grid_graph_with_hidden_cycle(self):
        """Test a grid graph with a hidden low-weight cycle."""
        # Create a 5x5 grid graph with a hidden cycle
        G = create_grid_graph(5, default_weight=10, cycle_weight=1)
        
        # The hidden cycle should have length 4 (4 edges of weight 1)
        self.assertEqual(sota_shortest_cycle(G), 4.0)
    
    def test_spatial_graph_with_hidden_cycle(self):
        """Test a spatial graph with a hidden low-weight cycle."""
        # Fix the random seed for reproducibility
        np.random.seed(42)
        
        # Create a spatial graph with 30 nodes and a hidden cycle
        G = create_spatial_graph(30, radius=0.3, default_weight=10, cycle_weight=1)
        
        # The hidden cycle should have length 4 (4 edges of weight 1)
        self.assertEqual(sota_shortest_cycle(G), 4.0)
    
    def test_optimized_vs_traditional(self):
        """Test that optimized algorithm gives same results as traditional approach."""
        # Create a test graph
        G = nx.Graph()
        G.add_edge(0, 1, weight=2.0)
        G.add_edge(1, 2, weight=3.0)
        G.add_edge(2, 3, weight=1.0)
        G.add_edge(3, 0, weight=4.0)
        G.add_edge(0, 2, weight=5.0)
        
        # Get cycle length using optimized algorithm
        optimized_result = sota_shortest_cycle(G)
        
        # Traditional approach for comparison
        traditional_result = None
        shortest_cycle_length = float('inf')
        
        for u, v in G.edges():
            weight = G[u][v].get('weight', 1.0)
            G.remove_edge(u, v)
            
            try:
                path_length = nx.shortest_path_length(G, u, v, weight='weight')
                cycle_length = path_length + weight
                
                if cycle_length < shortest_cycle_length:
                    shortest_cycle_length = cycle_length
            except nx.NetworkXNoPath:
                pass
                
            G.add_edge(u, v, weight=weight)
        
        if shortest_cycle_length != float('inf'):
            traditional_result = shortest_cycle_length
        
        # Both approaches should give the same result
        self.assertEqual(optimized_result, traditional_result)

if __name__ == '__main__':
    unittest.main() 