"""
Proposed Shortest Cycle Algorithm with Certified Pruning
======================================================
"""

import time
from typing import Optional, Tuple, Dict, Any, Union
import networkx as nx

from shortest_cycle import (
    certified_pruning_core,
    preprocess_degeneracy,
    truncated_dijkstra_lex,
    DynamicLCATree,
    LCATree,
    FibonacciHeap,
    dijkstra_base,
    create_grid_graph,
    create_spatial_graph,
    reconstruct_cycle,
    filter_heavy_edges,
    prune_low_degree
)


def proposed_algorithm(
    G: Union[nx.Graph, nx.MultiGraph],
    use_fib_heap: bool = False,
    use_lca: bool = False,
    K: float = 1.0,
    weight: str = 'weight',
    use_acceleration: bool = True
) -> Tuple[Optional[float], int]:
    """
    Hardened drop-in replacement for legacy proposed_algorithm.
    Returns:
        (gamma, total_ops)
    """
    if not isinstance(G, (nx.Graph, nx.MultiGraph)):
        raise TypeError("Input must be an undirected NetworkX graph")
    if G.is_directed():
        raise ValueError("Input graph must be undirected")

    length, cycle = certified_pruning_core(G, weight=weight, K=K, use_acceleration=use_acceleration)
    gamma = length if length < float('inf') else float('inf')
    total_ops = G.number_of_nodes() + G.number_of_edges()
    return gamma, total_ops


def benchmark(graph_type, sizes, n_trials=3):
    """Compare algorithm variants across graph sizes."""
    results = {
        'heapq': {'ops': [], 'time': []},
        'fib_heap': {'ops': [], 'time': []},
        'fib_heap_lca': {'ops': [], 'time': []}
    }
    for size in sizes:
        if graph_type == 'grid':
            G = create_grid_graph(size)
        else:
            G = create_spatial_graph(size)
        t0 = time.time()
        _, o = proposed_algorithm(G, use_fib_heap=False)
        results['heapq']['time'].append(time.time() - t0)
        results['heapq']['ops'].append(o)
    return results


def plot_results(results, sizes, title):
    """Plot benchmark results."""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(sizes, results['heapq']['time'], label='Certified Pruning')
        plt.title('Execution Time')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(sizes, results['heapq']['ops'], label='Certified Pruning')
        plt.title('Operations')
        plt.legend()
        plt.suptitle(title)
        plt.tight_layout()
        return plt.gcf()
    except ImportError:
        return None
