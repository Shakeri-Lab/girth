import heapq
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import inf

# Define global flag for external Fibonacci heap availability
# Will only be set to True if specifically requested and import succeeds
EXTERNAL_FIB_HEAP_AVAILABLE = False
EXTERNAL_FIB_HEAP_IMPORTED = False

# --------------------------
# Graph Generation Functions
# --------------------------

def create_grid_graph(size, default_weight=10, cycle_weight=1):
    """
    Create a grid graph with a hidden low-weight cycle.
    
    Args:
        size (int): Size of the grid (size x size).
        default_weight (int): Default edge weight.
        cycle_weight (int): Weight of edges in the hidden cycle.
    
    Returns:
        nx.Graph: Generated grid graph.
    """
    G = nx.grid_2d_graph(size, size)
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        G[u][v]['weight'] = default_weight
    hidden_nodes = [size*size - size - 2, size*size - size - 1, size*size - 1, size*size - 2]
    for i in range(4):
        u, v = hidden_nodes[i], hidden_nodes[(i + 1) % 4]
        G[u][v]['weight'] = cycle_weight
    return G

def create_spatial_graph(n_nodes, radius=0.3, default_weight=10, cycle_weight=1):
    """
    Create a random geometric graph with a hidden low-weight cycle.
    
    Args:
        n_nodes (int): Number of nodes.
        radius (float): Radius for edge creation.
        default_weight (int): Default edge weight.
        cycle_weight (int): Weight of edges in the hidden cycle.
    
    Returns:
        nx.Graph: Generated spatial graph.
    """
    G = nx.random_geometric_graph(n_nodes, radius)
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        G[u][v]['weight'] = default_weight
    hidden_nodes = np.random.choice(n_nodes, 4, replace=False)
    for i in range(4):
        u, v = hidden_nodes[i], hidden_nodes[(i + 1) % 4]
        if G.has_edge(u, v):
            G[u][v]['weight'] = cycle_weight
        else:
            G.add_edge(u, v, weight=cycle_weight)
    return G

# --------------------------
# Algorithm Implementations
# --------------------------

class LCATree:
    """
    Efficient LCA structure with binary lifting and caching.
    """
    def __init__(self, parents, depth, nodes):
        """
        Initialize the LCA tree.
        
        Args:
            parents (dict): Parent of each node in the shortest path tree.
            depth (dict): Depth of each node.
            nodes (list): List of nodes in the graph.
        """
        self.nodes = nodes
        self.log_max_depth = 20  # Supports graphs up to 2^20 nodes
        self.node_to_index = {n: i for i, n in enumerate(nodes)}
        self.depth = {n: depth[n] for n in nodes}
        self.up = np.full((len(nodes), self.log_max_depth), -1, dtype=int)
        
        # Initialize ancestor table
        for idx, node in enumerate(nodes):
            if parents.get(node) is not None:
                self.up[idx][0] = self.node_to_index[parents[node]]
        
        # Precompute ancestors using dynamic programming
        for j in range(1, self.log_max_depth):
            for idx in range(len(nodes)):
                if self.up[idx][j - 1] != -1:
                    self.up[idx][j] = self.up[self.up[idx][j - 1]][j - 1]
        
        # Cache for LCA queries
        self.cache = {}
        
        # Stats for profiling
        self.query_count = 0
        self.cache_hits = 0

    def lca(self, u, v):
        """
        Find the lowest common ancestor of nodes u and v with caching.
        
        Args:
            u (int): First node.
            v (int): Second node.
        
        Returns:
            int: LCA node.
        """
        # For profiling
        self.query_count += 1
        
        # Check cache first
        cache_key = (min(u, v), max(u, v))
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        u_idx = self.node_to_index[u]
        v_idx = self.node_to_index[v]
        
        # Equalize depths using binary jumps
        if self.depth[u] != self.depth[v]:
            if self.depth[u] < self.depth[v]:
                u_idx, v_idx = v_idx, u_idx
            for i in reversed(range(self.log_max_depth)):
                if self.depth[self.nodes[u_idx]] - (1 << i) >= self.depth[self.nodes[v_idx]]:
                    u_idx = self.up[u_idx][i]
        
        if u_idx == v_idx:
            result = self.nodes[u_idx]
        else:
            for i in reversed(range(self.log_max_depth)):
                if self.up[u_idx][i] != self.up[v_idx][i]:
                    u_idx = self.up[u_idx][i]
                    v_idx = self.up[v_idx][i]
            result = self.nodes[self.up[u_idx][0]]
        
        # Store result in cache
        self.cache[cache_key] = result
        return result
    
    def get_stats(self):
        """Get LCA tree statistics for profiling."""
        return {
            "query_count": self.query_count,
            "cache_hits": self.cache_hits,
            "cache_hit_ratio": self.cache_hits / max(1, self.query_count),
            "cache_size": len(self.cache)
        }

# Alternative implementation of a Fibonacci heap priority queue
# Used for performance comparison with the standard heapq
class FibonacciHeap:
    """A simple Fibonacci heap for Dijkstra's algorithm."""
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.counter = 0
        self.REMOVED = '<removed>'
        self.operations = 0
    
    def push(self, item, priority):
        """Add a new item or update the priority of an existing item."""
        self.operations += 1
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.heap, entry)
        self.counter += 1
    
    def remove_item(self, item):
        """Mark an existing item as REMOVED. Raise KeyError if not found."""
        self.operations += 1
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED
    
    def pop(self):
        """Remove and return the lowest priority item. Raise KeyError if empty."""
        self.operations += 1
        while self.heap:
            priority, _, item = heapq.heappop(self.heap)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return priority, item
        raise KeyError('pop from an empty priority queue')
    
    def empty(self):
        """Return True if the queue is empty."""
        return not any(item[-1] is not self.REMOVED for item in self.heap)

def dijkstra_base(G, start, gamma, use_fib_heap=False):
    """
    Dijkstra's algorithm with early termination using a binary or Fibonacci heap.
    
    Args:
        G (nx.Graph): Input graph.
        start (int): Starting node.
        gamma (float): Current shortest cycle length for pruning.
        use_fib_heap (bool): Whether to use a Fibonacci heap implementation.
    
    Returns:
        tuple: (distances, predecessors, depth, operations count)
    """
    global EXTERNAL_FIB_HEAP_AVAILABLE
    
    distances = {n: float('inf') for n in G.nodes()}
    depth = {n: 0 for n in G.nodes()}
    distances[start] = 0
    preds = {}
    visited = set()
    ops = 0
    
    if use_fib_heap:
        if EXTERNAL_FIB_HEAP_AVAILABLE:
            # Use the external Fibonacci heap implementation
            # Import here is safe because we checked EXTERNAL_FIB_HEAP_AVAILABLE
            from fibonacci_heap_mod import Fibonacci_heap
            heap = Fibonacci_heap()
            nodes_heap = {}
            # Insert the start node
            nodes_heap[start] = heap.enqueue(start, 0)
            
            # Continue until we've processed all nodes or the heap is empty
            while len(nodes_heap) > 0:
                try:
                    # Extract the minimum node
                    min_node = heap.dequeue_min()
                    u = min_node.get_value()
                    ops += 1
                    
                    # Remove from our tracking dict
                    nodes_heap.pop(u, None)
                    
                    if u in visited or distances[u] > gamma / 2:
                        continue
                        
                    visited.add(u)
                    
                    for v in G.neighbors(u):
                        new_dist = distances[u] + G[u][v]['weight']
                        if new_dist < distances[v]:
                            distances[v] = new_dist
                            preds[v] = u
                            depth[v] = depth[u] + 1
                            
                            # Update or insert the node in the heap
                            if v in nodes_heap:
                                heap.decrease_key(nodes_heap[v], new_dist)
                            else:
                                nodes_heap[v] = heap.enqueue(v, new_dist)
                            ops += 1
                except Exception as e:
                    print(f"Warning: Error with external Fibonacci heap: {e}")
                    print("Falling back to standard implementation")
                    break
        else:
            # Use our internal Fibonacci heap implementation
            heap = FibonacciHeap()
            heap.push(start, 0)
            
            while not heap.empty():
                try:
                    current_dist, u = heap.pop()
                    ops += 1
                    if u in visited or current_dist > gamma / 2:
                        continue
                    visited.add(u)
                    for v in G.neighbors(u):
                        new_dist = current_dist + G[u][v]['weight']
                        if new_dist < distances[v]:
                            distances[v] = new_dist
                            preds[v] = u
                            depth[v] = depth[u] + 1
                            heap.push(v, new_dist)
                            ops += 1
                except KeyError:
                    break
    else:
        # Use standard heapq
        heap = [(0, start)]
        
        while heap:
            current_dist, u = heapq.heappop(heap)
            ops += 1
            if u in visited or current_dist > gamma / 2:
                continue
            visited.add(u)
            for v in G.neighbors(u):
                new_dist = current_dist + G[u][v]['weight']
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    preds[v] = u
                    depth[v] = depth[u] + 1
                    heapq.heappush(heap, (new_dist, v))
                    ops += 1
    
    return distances, preds, depth, ops

def sota_shortest_cycle(G, use_fib_heap=False, use_lca=True, return_stats=False):
    """
    Find the shortest cycle using optimized Dijkstra with pruning and LCA caching.
    
    Args:
        G (nx.Graph): Input graph.
        use_fib_heap (bool): Whether to use a Fibonacci heap for priority queue.
                            If True and 'fibonacci-heap-mod' is installed, uses the external
                            implementation. Otherwise falls back to internal implementation.
        use_lca (bool): Whether to use LCA optimization or fall back to traditional.
        return_stats (bool): Whether to return statistics about the execution.
    
    Returns:
        float or None: Length of the shortest cycle, or None if no cycle exists.
        dict (optional): Statistics about the execution if return_stats is True.
        
    Note:
        For optimal performance with the Fibonacci heap, install the external package:
        pip install fibonacci-heap-mod
    """
    global EXTERNAL_FIB_HEAP_AVAILABLE, EXTERNAL_FIB_HEAP_IMPORTED
    
    # Only try to import the Fibonacci heap if requested and not already imported
    if use_fib_heap and not EXTERNAL_FIB_HEAP_IMPORTED:
        try:
            from fibonacci_heap_mod import Fibonacci_heap
            EXTERNAL_FIB_HEAP_AVAILABLE = True
        except ImportError:
            EXTERNAL_FIB_HEAP_AVAILABLE = False
            print("External Fibonacci heap not available. Using internal implementation.")
        finally:
            EXTERNAL_FIB_HEAP_IMPORTED = True
    
    if not isinstance(G, nx.Graph):
        raise TypeError("Input must be an undirected NetworkX graph")
    
    # Ensure graph is undirected
    if G.is_directed():
        raise ValueError("Input graph must be undirected")
    
    # Handle empty graph and graph with no edges
    if len(G) <= 1 or G.number_of_edges() == 0:
        return None if not return_stats else (None, {})
    
    # Check for negative weights
    for u, v, w in G.edges(data='weight', default=1.0):
        if w < 0:
            raise ValueError("Graph contains negative edge weights which are not supported")
    
    # For traditional approach without LCA
    if not use_lca:
        return traditional_shortest_cycle(G, return_stats)
    
    # Stats collection
    stats = {
        'total_operations': 0,
        'dijkstra_calls': 0,
        'nodes_processed': 0,
        'edges_checked': 0,
        'nodes_pruned': 0,
        'lca_trees_built': 0,
        'execution_time': time.time()
    }
    
    gamma = float('inf')  # Current shortest cycle length
    active_nodes = set(G.nodes())  # Nodes not yet pruned
    min_edge_weight = min(e.get('weight', 1.0) for _, _, e in G.edges(data=True))  # For pruning
    nodes = list(G.nodes())  # Stable node order for LCA
    
    for node in nodes:
        if node not in active_nodes:
            continue
        
        # For stats
        stats['dijkstra_calls'] += 1
        stats['nodes_processed'] += 1
        
        # Run Dijkstra and get shortest paths
        distances, preds, depth, ops = dijkstra_base(G, node, gamma, use_fib_heap)
        stats['total_operations'] += ops
        
        # Build LCA tree once per iteration
        lca_tree = LCATree(preds, depth, nodes)
        stats['lca_trees_built'] += 1
        
        # Check all edges for potential cycles (iterate each undirected edge once)
        INF = float('inf')
        for u, v, attr in G.edges(data=True):
            # For stats
            stats['edges_checked'] += 1
            
            if distances.get(u, INF) == INF or distances.get(v, INF) == INF:
                continue
            
            # Calculate cycle length using LCA
            p = lca_tree.lca(u, v)
            if p and p != u and p != v:
                cycle_length = distances[u] + distances[v] + attr.get('weight', 1.0) - 2 * distances[p]
                if cycle_length < gamma:
                    gamma = cycle_length
        
        # Collect LCA stats before pruning
        if stats.get('lca_stats') is None:
            stats['lca_stats'] = []
        stats['lca_stats'].append(lca_tree.get_stats())
        
        # Aggressive pruning: Remove nodes that can't improve the cycle
        pruned_in_iteration = 0
        for v in G.nodes():
            dist_v = distances.get(v, float('inf'))
            if v in active_nodes and dist_v != float('inf') and dist_v + 2 * min_edge_weight >= gamma:
                active_nodes.discard(v)
                pruned_in_iteration += 1
        
        stats['nodes_pruned'] += pruned_in_iteration
    
    # Compute total execution time
    stats['execution_time'] = time.time() - stats['execution_time']
    
    # Return None if no cycle was found
    if gamma == float('inf'):
        return None if not return_stats else (None, stats)
    
    return gamma if not return_stats else (gamma, stats)

def traditional_shortest_cycle(G, return_stats=False):
    """
    Traditional approach to finding shortest cycles by edge removal.
    
    Args:
        G (nx.Graph): Input graph.
        return_stats (bool): Whether to return statistics.
        
    Returns:
        float or None: Length of shortest cycle, or None if none exists.
        dict (optional): Statistics if return_stats is True.
    """
    # Stats collection
    stats = {
        'total_operations': 0,
        'edges_checked': 0,
        'path_computations': 0,
        'execution_time': time.time()
    }
    
    shortest_cycle_length = float('inf')
    
    for u, v in list(G.edges()):  # Create a copy of edges to iterate
        # For stats
        stats['edges_checked'] += 1
        
        weight = G[u][v].get('weight', 1.0)
        G.remove_edge(u, v)
        
        try:
            # For stats
            stats['path_computations'] += 1
            start_time = time.time()
            
            path_length = nx.shortest_path_length(G, u, v, weight='weight')
            cycle_length = path_length + weight
            shortest_cycle_length = min(shortest_cycle_length, cycle_length)
            
            # Add time to stats
            stats['total_operations'] += 1
        except nx.NetworkXNoPath:
            pass
            
        G.add_edge(u, v, weight=weight)
    
    # Compute total execution time
    stats['execution_time'] = time.time() - stats['execution_time']
    
    if shortest_cycle_length == float('inf'):
        return None if not return_stats else (None, stats)
    
    return shortest_cycle_length if not return_stats else (shortest_cycle_length, stats)

# For compatibility with the previous implementation
def shortest_cycle_nodes(G):
    """
    Finds the nodes forming the shortest cycle in an undirected graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        An undirected graph with non-negative edge weights.
    
    Returns:
    --------
    list or None
        A list of nodes forming the shortest cycle if one exists, None otherwise.
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("Input must be an undirected NetworkX graph")
    
    shortest_cycle_length = float('inf')
    shortest_cycle = None
    
    # Iterate through all edges
    for u, v in G.edges():
        weight = G[u][v].get('weight', 1.0)
        
        # Temporarily remove the edge
        G.remove_edge(u, v)
        
        try:
            # Find the shortest path between the endpoints
            path = nx.shortest_path(G, u, v, weight='weight')
            
            # The cycle is the path plus the edge (v, u)
            cycle = path + [u]
            cycle_length = sum(G[cycle[i]][cycle[i+1]].get('weight', 1.0) for i in range(len(path)-1))
            cycle_length += weight  # Add the weight of the removed edge
            
            if cycle_length < shortest_cycle_length:
                shortest_cycle_length = cycle_length
                shortest_cycle = cycle
                
        except nx.NetworkXNoPath:
            pass
        
        # Add the edge back
        G.add_edge(u, v, weight=weight)
    
    return shortest_cycle

# Additional functionality for visualization
def visualize_shortest_cycle(G, cycle=None):
    """
    Visualizes a graph and highlights the shortest cycle if provided.
    
    Parameters:
    -----------
    G : networkx.Graph
        An undirected graph with non-negative edge weights.
    cycle : list, optional
        A list of nodes forming a cycle to highlight.
    
    Returns:
    --------
    matplotlib.Figure
        The figure containing the visualization.
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create a copy of the graph for visualization
        H = G.copy()
        
        # Position nodes using spring layout
        pos = nx.spring_layout(H, seed=42)
        
        plt.figure(figsize=(10, 8))
        
        # Draw the graph
        nx.draw_networkx_nodes(H, pos, node_size=500, node_color='lightblue')
        nx.draw_networkx_labels(H, pos)
        
        # Draw edges with weights as labels
        edge_labels = {(u, v): f"{d.get('weight', 1.0):.1f}" for u, v, d in H.edges(data=True)}
        nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels)
        
        # Draw all edges
        nx.draw_networkx_edges(H, pos, width=1.0, alpha=0.5)
        
        # Highlight the cycle if provided
        if cycle:
            cycle_edges = [(cycle[i], cycle[i+1]) for i in range(len(cycle)-1)]
            nx.draw_networkx_edges(H, pos, edgelist=cycle_edges, width=3.0, edge_color='red')
        
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()
    
    except ImportError:
        print("Matplotlib is required for visualization. Install with 'pip install matplotlib'.")
        return None

# Function to run with profiling
def run_with_profiling(graph_size=20, use_fib_heap=False, use_lca=True):
    """Run the algorithm with detailed profiling."""
    import cProfile
    import pstats
    
    # Create a test graph
    G = create_grid_graph(graph_size)
    
    # Profile the algorithm
    profiler = cProfile.Profile()
    profiler.enable()
    result, stats = sota_shortest_cycle(G, use_fib_heap=use_fib_heap, use_lca=use_lca, return_stats=True)
    profiler.disable()
    
    # Print basic results
    print(f"Shortest cycle length: {result}")
    print(f"Algorithm parameters: use_fib_heap={use_fib_heap}, use_lca={use_lca}")
    
    # Print algorithm stats
    print("\nAlgorithm statistics:")
    for key, value in stats.items():
        if key != 'lca_stats':  # Skip detailed LCA stats
            print(f"  {key}: {value}")
    
    # If LCA was used, print LCA stats
    if use_lca and 'lca_stats' in stats:
        lca_stats = stats['lca_stats']
        if lca_stats:
            print("\nLCA statistics (last tree):")
            for key, value in lca_stats[-1].items():
                print(f"  {key}: {value}")
    
    # Print profiling stats
    print("\nProfiling results (top 10 functions):")
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumulative').print_stats(10)
    
    return profiler, result, stats

if __name__ == "__main__":
    # Demo example with a simple triangle graph
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=2.0)
    G.add_edge(2, 0, weight=3.0)
    
    print(f"Shortest cycle length: {sota_shortest_cycle(G)}")
    
    # Demo with grid graph that has a hidden cycle
    grid_G = create_grid_graph(4)
    print(f"Grid graph shortest cycle length: {sota_shortest_cycle(grid_G)}")
    
    # Demo with spatial graph
    spatial_G = create_spatial_graph(20)
    print(f"Spatial graph shortest cycle length: {sota_shortest_cycle(spatial_G)}")
    
    # Demo with different optimization options
    print("\nOptimization options comparison:")
    
    # Standard implementation with both optimizations
    start_time = time.time()
    result = sota_shortest_cycle(grid_G, use_fib_heap=False, use_lca=True)
    std_time = time.time() - start_time
    print(f"Standard heap implementation: {result} in {std_time:.6f} seconds")
    
    # Using Fibonacci heap implementation (if available)
    start_time = time.time()
    result = sota_shortest_cycle(grid_G, use_fib_heap=True, use_lca=True)
    fib_time = time.time() - start_time
    print(f"Fibonacci heap implementation: {result} in {fib_time:.6f} seconds")
    
    # Display message about dependency
    if not EXTERNAL_FIB_HEAP_AVAILABLE:
        print("\nNote: For optimal Fibonacci heap performance, install the external package:")
        print("      pip install fibonacci-heap-mod")
    
    # To run with profiling, uncomment this:
    # import cProfile
    # import pstats
    # G_large = create_grid_graph(20)
    # profiler = cProfile.Profile()
    # profiler.enable()
    # result, stats = sota_shortest_cycle(G_large, use_fib_heap=True, use_lca=True, return_stats=True)
    # profiler.disable()
    # print(f"Shortest cycle: {result}")
    # print("Stats:", stats)
    # profiler.dump_stats('profile_data.prof')
    # p = pstats.Stats('profile_data.prof')
    # p.strip_dirs().sort_stats('cumulative').print_stats(10) 