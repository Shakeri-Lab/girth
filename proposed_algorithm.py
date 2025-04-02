import heapq
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Optional Fibonacci heap - will be imported only when requested
EXTERNAL_FIB_HEAP_AVAILABLE = False
EXTERNAL_FIB_HEAP_IMPORTED = False

# --------------------------
# Graph Generation Functions
# --------------------------

def create_grid_graph(size, default_weight=10, cycle_weight=1):
    """Create a grid graph with a hidden low-weight cycle."""
    G = nx.grid_2d_graph(size, size)
    G = nx.convert_node_labels_to_integers(G)
    # Set default weights for all edges
    for u, v in G.edges():
        G[u][v]['weight'] = default_weight
    # Embed a 4-node cycle with lower weight
    hidden_nodes = [size*size - size - 2, size*size - size -1, size*size -1, size*size -2]
    for i in range(4):
        u, v = hidden_nodes[i], hidden_nodes[(i+1)%4]
        G[u][v]['weight'] = cycle_weight
    return G

def create_spatial_graph(n_nodes, radius=0.3, default_weight=10, cycle_weight=1):
    """Create a random geometric graph with a hidden low-weight cycle."""
    G = nx.random_geometric_graph(n_nodes, radius)
    G = nx.convert_node_labels_to_integers(G)
    # Set default weights for all edges
    for u, v in G.edges():
        G[u][v]['weight'] = default_weight
    # Add a 4-node cycle with lower weight
    hidden_nodes = np.random.choice(n_nodes, 4, replace=False)
    for i in range(4):
        u, v = hidden_nodes[i], hidden_nodes[(i+1)%4]
        if G.has_edge(u, v):
            G[u][v]['weight'] = cycle_weight
        else:
            G.add_edge(u, v, weight=cycle_weight)
    return G

# --------------------------
# Algorithm Implementations
# --------------------------

class LCATree:
    """Efficient LCA structure using binary lifting for ancestor jumps."""
    def __init__(self, parents, depth, nodes):
        """
        Initialize LCA tree with parent pointers and depths.
        
        Args:
            parents: Dict of parent nodes from Dijkstra's tree
            depth: Dict of depths from Dijkstra's tree
            nodes: List of all nodes in the graph
        """
        self.nodes = nodes  
        self.log_max_depth = 20  # Supports graphs up to 2^20 nodes
        self.up = np.full((len(nodes), self.log_max_depth), -1, dtype=int)  # Ancestor table
        self.depth = np.array([depth[n] for n in nodes], dtype=int)
        self.node_to_index = {node: idx for idx, node in enumerate(nodes)}  # O(1) lookups

        # Initialize ancestor table
        for idx, node in enumerate(nodes):
            if parents.get(node) is not None:
                self.up[idx][0] = self.node_to_index[parents[node]]
        
        # Precompute ancestors using dynamic programming
        for j in range(1, self.log_max_depth):
            for idx in range(len(nodes)):
                if self.up[idx][j-1] != -1:
                    self.up[idx][j] = self.up[self.up[idx][j-1]][j-1]

    def lca(self, u, v):
        """Find lowest common ancestor using binary lifting."""
        u_idx = self.node_to_index[u]
        v_idx = self.node_to_index[v]

        # Equalize depths using binary jumps
        if self.depth[u_idx] != self.depth[v_idx]:
            if self.depth[u_idx] < self.depth[v_idx]:
                u_idx, v_idx = v_idx, u_idx
            # Lift u to v's depth
            for i in reversed(range(self.log_max_depth)):
                if self.depth[u_idx] - (1 << i) >= self.depth[v_idx]:
                    u_idx = self.up[u_idx][i]

        if u_idx == v_idx:
            return self.nodes[u_idx]

        # Find LCA by lifting both nodes
        for i in reversed(range(self.log_max_depth)):
            if self.up[u_idx][i] != -1 and self.up[u_idx][i] != self.up[v_idx][i]:
                u_idx = self.up[u_idx][i]
                v_idx = self.up[v_idx][i]

        return self.nodes[self.up[u_idx][0]] if self.up[u_idx][0] != -1 else None

# Internal Fibonacci heap implementation
class FibonacciHeap:
    """A simple Fibonacci heap implementation for Dijkstra's algorithm."""
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
    Dijkstra's algorithm with early termination and optional Fibonacci heap.
    
    Returns:
        distances: Shortest paths from start node
        preds: Predecessor pointers for path reconstruction
        depth: Depth of each node in the shortest path tree
        ops: Number of priority queue operations
    """
    global EXTERNAL_FIB_HEAP_AVAILABLE, EXTERNAL_FIB_HEAP_IMPORTED
    
    # Check if we need to import the Fibonacci heap
    if use_fib_heap and not EXTERNAL_FIB_HEAP_IMPORTED:
        try:
            from fibonacci_heap_mod import Fibonacci_heap
            EXTERNAL_FIB_HEAP_AVAILABLE = True
        except ImportError:
            EXTERNAL_FIB_HEAP_AVAILABLE = False
            print("External Fibonacci heap not available. Using internal implementation.")
        finally:
            EXTERNAL_FIB_HEAP_IMPORTED = True
    
    distances = {n: float('inf') for n in G.nodes()}
    depth = {n: 0 for n in G.nodes()}
    distances[start] = 0
    preds = {}
    ops = 0

    if use_fib_heap:
        if EXTERNAL_FIB_HEAP_AVAILABLE:
            # Use the external Fibonacci heap implementation
            from fibonacci_heap_mod import Fibonacci_heap
            heap = Fibonacci_heap()
            nodes_heap = {start: heap.enqueue(start, 0)}
            visited = set()
            
            while nodes_heap:
                try:
                    min_node = heap.dequeue_min()
                    u = min_node.get_value()
                    ops += 1
                    
                    # Remove from tracking dict
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
            visited = set()
            
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
        visited = set()
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

def proposed_algorithm(G, use_fib_heap=False, use_lca=False):
    """Find shortest cycle using optimized Dijkstra with pruning."""
    gamma = float('inf')  # Current shortest cycle length
    total_ops = 0         # Total priority queue operations
    active_nodes = set(G.nodes())  # Nodes not yet pruned
    min_edge_weight = min(e['weight'] for _, _, e in G.edges(data=True))  # For pruning

    nodes = list(G.nodes())  # Stable node order for LCA

    for node in nodes:
        if node not in active_nodes:
            continue

        # Run Dijkstra and get shortest paths
        distances, preds, depth, ops = dijkstra_base(G, node, gamma, use_fib_heap)
        total_ops += ops

        # Build LCA tree once per iteration
        lca_tree = LCATree(preds, depth, nodes) if use_lca else None

        # Track best cycle found in this iteration
        local_cycle_length = float('inf')

        # Check all edges for potential cycles
        for u in G.nodes():
            for v in G.neighbors(u):
                if distances[u] == float('inf') or distances[v] == float('inf') or u == v:
                    continue

                # Calculate cycle length using LCA if available
                if use_lca and lca_tree:
                    p = lca_tree.lca(u, v)
                    if p and p != u and p != v:
                        cycle_length = distances[u] + distances[v] + G[u][v]['weight'] - 2 * distances[p]
                    else:
                        cycle_length = float('inf')  # Invalid cycle
                else:
                    # Fallback to simple cycle check
                    if preds.get(v) != u and preds.get(u) != v:
                        cycle_length = distances[u] + distances[v] + G[u][v]['weight']
                    else:
                        cycle_length = float('inf')

                # Update global and local cycle information
                if cycle_length < gamma:
                    gamma = cycle_length
                if cycle_length < local_cycle_length:
                    local_cycle_length = cycle_length

        # Aggressive pruning: Remove nodes that can't improve the cycle
        for v in G.nodes():
            if distances[v] + 2 * min_edge_weight >= gamma:
                active_nodes.discard(v)

    return gamma, total_ops

# --------------------------
# Benchmarking & Visualization
# --------------------------

def benchmark(graph_type, sizes, n_trials=3):
    """Compare algorithm variants across graph sizes."""
    results = {
        'heapq': {'ops': [], 'time': []},
        'fib_heap': {'ops': [], 'time': []},
        'fib_heap_lca': {'ops': [], 'time': []}
    }

    for size in sizes:
        print(f"Testing {graph_type} size {size}")
        ops = {k: [] for k in results}
        times = {k: [] for k in results}

        for _ in range(n_trials):
            # Generate test graph
            if graph_type == 'grid':
                G = create_grid_graph(size)
            elif graph_type == 'spatial':
                G = create_spatial_graph(size)

            # Test heapq variant
            start = time.time()
            _, o = proposed_algorithm(G, use_fib_heap=False)
            times['heapq'].append(time.time() - start)
            ops['heapq'].append(o)

            # Test Fibonacci heap variant
            start = time.time()
            _, o = proposed_algorithm(G, use_fib_heap=True)
            times['fib_heap'].append(time.time() - start)
            ops['fib_heap'].append(o)

            # Test Fibonacci+LCA variant
            start = time.time()
            _, o = proposed_algorithm(G, use_fib_heap=True, use_lca=True)
            times['fib_heap_lca'].append(time.time() - start)
            ops['fib_heap_lca'].append(o)

        # Aggregate results
        for k in results:
            results[k]['ops'].append(np.mean(ops[k]))
            results[k]['time'].append(np.mean(times[k]))

    return results

def plot_results(results, sizes, title):
    """Visualize benchmark results."""
    plt.figure(figsize=(12, 5))
    
    # Operations plot
    plt.subplot(1, 2, 1)
    for label, data in results.items():
        plt.plot(sizes, data['ops'], marker='o', label=label)
    plt.xlabel('Graph Size')
    plt.ylabel('Priority Queue Operations')
    plt.title(f'{title} - Operations')
    plt.legend()
    
    # Runtime plot
    plt.subplot(1, 2, 2)
    for label, data in results.items():
        plt.plot(sizes, data['time'], marker='o', label=label)
    plt.xlabel('Graph Size')
    plt.ylabel('Execution Time (s)')
    plt.title(f'{title} - Runtime')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# --------------------------
# Main Execution
# --------------------------
if __name__ == '__main__':
    # Configure benchmark parameters
    graph_classes = ['grid', 'spatial']
    sizes = {
        'grid': [5, 7, 9],        # Grid dimensions
        'spatial': [20, 30, 40]   # Node counts
    }

    # Run benchmarks and plot results
    for graph_type in graph_classes:
        results = benchmark(graph_type, sizes[graph_type])
        plot_results(results, sizes[graph_type], f"{graph_type.capitalize()} Graphs") 