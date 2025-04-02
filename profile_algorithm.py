#!/usr/bin/env python3
"""
Profiling script for the shortest cycle algorithm.

This script demonstrates different ways to profile and analyze the performance 
of the shortest cycle algorithm with various optimization options.
"""

import time
import cProfile
import pstats
import os
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from io import StringIO
from shortest_cycle import (
    sota_shortest_cycle, 
    traditional_shortest_cycle, 
    create_grid_graph,
    create_spatial_graph,
    run_with_profiling
)

# --------------------------
# Profiling Functions
# --------------------------

def profile_and_save(graph, name, use_fib_heap=False, use_lca=True, output_dir="profiling_results"):
    """
    Profile the algorithm with given settings and save results.
    
    Args:
        graph: NetworkX graph to analyze
        name: Name prefix for output files
        use_fib_heap: Whether to use Fibonacci heap
        use_lca: Whether to use LCA optimization
        output_dir: Directory to save results
    
    Returns:
        tuple: (result, stats, profiler)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run algorithm with timing
    start_time = time.time()
    result, stats = sota_shortest_cycle(
        graph, 
        use_fib_heap=use_fib_heap, 
        use_lca=use_lca, 
        return_stats=True
    )
    execution_time = time.time() - start_time
    
    # Stop profiler
    profiler.disable()
    
    # Generate filename with options
    options = []
    if use_fib_heap:
        options.append("fib")
    if use_lca:
        options.append("lca")
    
    filename_base = f"{name}_{'_'.join(options) if options else 'basic'}"
    prof_file = os.path.join(output_dir, f"{filename_base}.prof")
    stats_file = os.path.join(output_dir, f"{filename_base}_stats.txt")
    
    # Save raw profiling data
    profiler.dump_stats(prof_file)
    
    # Generate and save text report
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    
    with open(stats_file, 'w') as f:
        f.write(f"Shortest cycle length: {result}\n")
        f.write(f"Total execution time: {execution_time:.6f} seconds\n\n")
        f.write("Algorithm options:\n")
        f.write(f"  - Fibonacci heap: {use_fib_heap}\n")
        f.write(f"  - LCA optimization: {use_lca}\n\n")
        f.write("Algorithm statistics:\n")
        for key, value in stats.items():
            if key != 'lca_stats':
                f.write(f"  {key}: {value}\n")
        f.write("\nProfile data:\n")
        f.write(s.getvalue())
    
    print(f"Profiled {name} (fib={use_fib_heap}, lca={use_lca})")
    print(f"  Result: {result}")
    print(f"  Time: {execution_time:.6f} seconds")
    print(f"  Output: {prof_file}")
    
    return result, stats, profiler

def compare_optimization_options(graph_size=20, graph_type="grid"):
    """
    Compare different optimization options on the same graph.
    
    Args:
        graph_size: Size of the graph
        graph_type: Type of graph ('grid' or 'spatial')
    """
    # Create the graph
    if graph_type == "grid":
        graph = create_grid_graph(graph_size)
        graph_name = f"grid_{graph_size}x{graph_size}"
    else:
        graph = create_spatial_graph(graph_size)
        graph_name = f"spatial_{graph_size}"
    
    print(f"\nComparing optimization options on {graph_name}:\n")
    
    # Check if external Fibonacci heap is available
    external_fib_available = False
    try:
        from fibonacci_heap_mod import Fibonacci_heap
        external_fib_available = True
    except ImportError:
        external_fib_available = False
        print("Note: External Fibonacci heap not available. Install with: pip install fibonacci-heap-mod")
    
    # Define combinations to test
    combinations = [
        {"use_fib_heap": False, "use_lca": False, "label": "Basic (no optimizations)"},
        {"use_fib_heap": False, "use_lca": True, "label": "LCA Optimization Only"},
        {"use_fib_heap": True, "use_lca": False, "label": "Internal Fibonacci Heap"},
        {"use_fib_heap": True, "use_lca": True, "label": "Internal FibHeap + LCA"}
    ]
    
    # We don't need to test external/internal separately anymore as
    # the algorithm automatically uses the best available implementation
    
    results = []
    
    # Run profiling for each combination
    for combo in combinations:
        start_time = time.time()
        if combo["use_lca"] == False and combo["use_fib_heap"] == False:
            # Use traditional approach for the basic case
            cycle_length, stats = traditional_shortest_cycle(graph, return_stats=True)
        else:
            cycle_length, stats = sota_shortest_cycle(
                graph, 
                use_fib_heap=combo["use_fib_heap"], 
                use_lca=combo["use_lca"], 
                return_stats=True
            )
        execution_time = time.time() - start_time
        
        results.append({
            "label": combo["label"],
            "time": execution_time,
            "result": cycle_length,
            "stats": stats
        })
        
        print(f"{combo['label']}:")
        print(f"  Result: {cycle_length}")
        print(f"  Time: {execution_time:.6f} seconds")
        if "total_operations" in stats:
            print(f"  Operations: {stats['total_operations']}")
        print()
    
    # Plot comparison
    labels = [r["label"] for r in results]
    times = [r["time"] for r in results]
    
    plt.figure(figsize=(12, 6))
    plt.bar(labels, times, color=['lightblue', 'lightgreen', 'salmon', 'gold'][:len(labels)])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Performance Comparison on {graph_name}')
    plt.tight_layout()
    plt.savefig(f"optimization_comparison_{graph_name}.png", dpi=300)
    
    return results

def analyze_graph_scaling(graph_type="grid", sizes=None, use_fib_heap=True, use_lca=True):
    """
    Analyze how algorithm performance scales with graph size.
    
    Args:
        graph_type: Type of graph ('grid' or 'spatial')
        sizes: List of graph sizes to test
        use_fib_heap: Whether to use Fibonacci heap
        use_lca: Whether to use LCA optimization
    """
    if sizes is None:
        if graph_type == "grid":
            sizes = [5, 10, 15, 20, 25]
        else:
            sizes = [20, 50, 100, 200, 300]
    
    print(f"\nAnalyzing scaling for {graph_type} graphs with sizes {sizes}:")
    print(f"Options: use_fib_heap={use_fib_heap}, use_lca={use_lca}\n")
    
    results = {"sizes": sizes, "times": [], "operations": []}
    
    for size in sizes:
        # Create the graph
        if graph_type == "grid":
            graph = create_grid_graph(size)
        else:
            graph = create_spatial_graph(size)
        
        # Time the algorithm
        start_time = time.time()
        _, stats = sota_shortest_cycle(
            graph, 
            use_fib_heap=use_fib_heap, 
            use_lca=use_lca, 
            return_stats=True
        )
        execution_time = time.time() - start_time
        
        results["times"].append(execution_time)
        results["operations"].append(stats["total_operations"])
        
        print(f"Size {size}:")
        print(f"  Time: {execution_time:.6f} seconds")
        print(f"  Operations: {stats['total_operations']}")
        print(f"  Nodes processed: {stats['nodes_processed']}")
        print(f"  Nodes pruned: {stats['nodes_pruned']}")
        print()
    
    # Plot scaling results
    plt.figure(figsize=(12, 5))
    
    # Time plot
    plt.subplot(1, 2, 1)
    plt.plot(sizes, results["times"], marker='o', color='blue')
    plt.xlabel('Graph Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Time Scaling')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Operations plot
    plt.subplot(1, 2, 2)
    plt.plot(sizes, results["operations"], marker='s', color='red')
    plt.xlabel('Graph Size')
    plt.ylabel('Total Operations')
    plt.title('Computational Complexity Scaling')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"scaling_{graph_type}.png", dpi=300)
    
    return results

def main():
    """Main function to parse arguments and run profiling."""
    parser = argparse.ArgumentParser(description='Profile and analyze shortest cycle algorithm')
    parser.add_argument('--mode', choices=['profile', 'compare', 'scale'], default='profile',
                        help='Profiling mode: single profile, compare options, or analyze scaling')
    parser.add_argument('--graph', choices=['grid', 'spatial'], default='grid',
                        help='Type of graph to analyze')
    parser.add_argument('--size', type=int, default=20,
                        help='Size of the graph')
    parser.add_argument('--use-fib-heap', action='store_true',
                        help='Use Fibonacci heap for priority queue')
    parser.add_argument('--use-lca', action='store_true', default=True,
                        help='Use LCA optimization (default: True)')
    
    args = parser.parse_args()
    
    if args.mode == 'profile':
        # Create graph
        if args.graph == 'grid':
            graph = create_grid_graph(args.size)
            graph_name = f"grid_{args.size}x{args.size}"
        else:
            graph = create_spatial_graph(args.size)
            graph_name = f"spatial_{args.size}"
        
        # Run profiling
        profile_and_save(
            graph, 
            graph_name, 
            use_fib_heap=args.use_fib_heap, 
            use_lca=args.use_lca
        )
        
    elif args.mode == 'compare':
        compare_optimization_options(args.size, args.graph)
        
    elif args.mode == 'scale':
        if args.graph == 'grid':
            sizes = [5, 10, 15, 20, 25]
        else:
            sizes = [20, 50, 100, 200, 300]
        
        analyze_graph_scaling(
            args.graph, 
            sizes, 
            use_fib_heap=args.use_fib_heap, 
            use_lca=args.use_lca
        )

if __name__ == "__main__":
    main() 