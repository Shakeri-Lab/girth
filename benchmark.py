import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pstats
import os
from io import StringIO
from shortest_cycle import sota_shortest_cycle, create_grid_graph, create_spatial_graph

# --------------------------
# Benchmarking & Visualization
# --------------------------

def benchmark_graph_type(graph_generator, sizes, num_trials=3, title="Benchmark Results"):
    """
    Benchmark the shortest cycle algorithm on different graph types and sizes.
    
    Args:
        graph_generator : callable
            A function that takes a size parameter and returns a NetworkX graph.
        sizes : list
            List of graph sizes (nodes) to benchmark.
        num_trials : int, optional
            Number of trials per size to average the results.
        title : str, optional
            Title for the benchmark plot.
    
    Returns:
        tuple: (sizes, times, title) for plotting.
    """
    times = []
    
    for size in sizes:
        print(f"Benchmarking {title} of size {size}...")
        time_sum = 0
        
        for _ in range(num_trials):
            # Generate graph
            G = graph_generator(size)
            
            # Measure execution time
            start_time = time.time()
            sota_shortest_cycle(G)
            end_time = time.time()
            
            time_sum += (end_time - start_time)
        
        # Average time across trials
        avg_time = time_sum / num_trials
        times.append(avg_time)
        print(f"  Average time: {avg_time:.6f} seconds")
    
    return sizes, times, title

def benchmark_vs_traditional(graph_generator, sizes, num_trials=3):
    """
    Compare our optimized algorithm with a traditional approach.
    
    Args:
        graph_generator : callable
            Function to generate test graphs.
        sizes : list
            List of graph sizes to test.
        num_trials : int
            Number of trials per size.
            
    Returns:
        dict: Results with times for each approach.
    """
    results = {
        'optimized': {'time': []},
        'traditional': {'time': []}
    }
    
    for size in sizes:
        print(f"Testing size {size}")
        opt_times = []
        trad_times = []
        
        for _ in range(num_trials):
            G = graph_generator(size)
            
            # Optimized algorithm (our implementation)
            start = time.time()
            sota_shortest_cycle(G)
            opt_times.append(time.time() - start)
            
            # Traditional approach (edge removal + Dijkstra)
            start = time.time()
            traditional_shortest_cycle(G)
            trad_times.append(time.time() - start)
        
        results['optimized']['time'].append(np.mean(opt_times))
        results['traditional']['time'].append(np.mean(trad_times))
    
    return results

def traditional_shortest_cycle(G):
    """
    Traditional approach to finding shortest cycles by edge removal.
    Used for comparison benchmarking.
    
    Args:
        G: NetworkX graph
        
    Returns:
        float or None: Length of shortest cycle
    """
    shortest_cycle_length = float('inf')
    
    for u, v in list(G.edges()):  # Create a copy of edges to iterate
        weight = G[u][v].get('weight', 1.0)
        G.remove_edge(u, v)
        
        try:
            path_length = nx.shortest_path_length(G, u, v, weight='weight')
            cycle_length = path_length + weight
            shortest_cycle_length = min(shortest_cycle_length, cycle_length)
        except nx.NetworkXNoPath:
            pass
            
        G.add_edge(u, v, weight=weight)
    
    if shortest_cycle_length == float('inf'):
        return None
    return shortest_cycle_length

def plot_benchmark_results(results, title):
    """
    Plot benchmark results with log-log scale.
    
    Args:
        results : list
            List of tuples (sizes, times, label).
        title : str
            Plot title.
    """
    plt.figure(figsize=(12, 8))
    
    for sizes, times, label in results:
        plt.loglog(sizes, times, marker='o', label=label)
    
    plt.xlabel('Graph Size (Nodes)')
    plt.ylabel('Execution Time (seconds)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_comparison_results(results, sizes, title):
    """
    Plot comparison between optimized and traditional approaches.
    
    Args:
        results : dict
            Dictionary with benchmark results.
        sizes : list
            Graph sizes that were tested.
        title : str
            Plot title.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(sizes, results['optimized']['time'], marker='o', label='Optimized Algorithm')
    plt.plot(sizes, results['traditional']['time'], marker='s', label='Traditional Approach')
    
    plt.xlabel('Graph Size')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Algorithm Comparison: {title}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.savefig(f"comparison_{title.replace(' ', '_').lower()}.png", dpi=300, bbox_inches='tight')
    plt.show()

def run_all_benchmarks():
    """Run benchmarks on various graph types and plot the results."""
    # Grid graph parameters
    grid_sizes = [4, 5, 7, 9, 11, 13, 15]  # Grid dimensions
    
    # Spatial graph parameters  
    spatial_sizes = [20, 50, 100, 200, 300, 400, 500]  # Node counts
    
    # Results collection for different graph types
    results = []
    
    # Grid graph benchmarks
    print("\nBenchmarking Grid Graphs...")
    result = benchmark_graph_type(
        lambda size: create_grid_graph(size), 
        grid_sizes,
        title="Grid Graphs"
    )
    results.append(result)
    
    # Spatial graph benchmarks
    print("\nBenchmarking Spatial Graphs...")
    result = benchmark_graph_type(
        lambda size: create_spatial_graph(size, radius=0.4), 
        spatial_sizes,
        title="Spatial Graphs"
    )
    results.append(result)
    
    # Plot combined results
    plot_benchmark_results(results, "Shortest Cycle Algorithm Performance")
    
    # Compare with traditional approach on smaller graphs
    print("\nComparing with traditional approach...")
    comparison_grid_sizes = [4, 5, 7, 9]
    comparison_spatial_sizes = [20, 50, 100, 150]
    
    grid_comparison = benchmark_vs_traditional(
        lambda size: create_grid_graph(size),
        comparison_grid_sizes
    )
    plot_comparison_results(grid_comparison, comparison_grid_sizes, "Grid Graphs")
    
    spatial_comparison = benchmark_vs_traditional(
        lambda size: create_spatial_graph(size),
        comparison_spatial_sizes
    )
    plot_comparison_results(spatial_comparison, comparison_spatial_sizes, "Spatial Graphs")
    
    # Print summary
    print("\nBenchmark Summary:")
    for sizes, times, title in results:
        print(f"{title}:")
        for size, t in zip(sizes, times):
            print(f"  Size {size}: {t:.6f} seconds")

def benchmark_algorithm_components():
    """Benchmark different components of the algorithm to identify bottlenecks."""
    # This is an optional function to profile different parts of the algorithm
    
    # Create a medium-sized test graph
    G = create_spatial_graph(200, radius=0.3)
    
    # Time breakdown
    times = {
        'dijkstra': 0,
        'lca_construction': 0,
        'lca_queries': 0,
        'total': 0
    }
    
    # Start timing
    total_start = time.time()
    
    # We'd need to instrument the algorithm to get component times
    # This would typically be done using profiling tools
    
    # End timing
    times['total'] = time.time() - total_start
    
    # Print breakdown
    print("\nAlgorithm Component Analysis:")
    for component, duration in times.items():
        percentage = (duration / times['total']) * 100 if times['total'] > 0 else 0
        print(f"  {component}: {duration:.6f}s ({percentage:.1f}%)")

# --------------------------
# Profiling Functions
# --------------------------

def run_profiling(graph_size=20, output_dir="profiling_results"):
    """
    Run detailed profiling on the algorithm with various settings.
    
    Args:
        graph_size (int): Size of the test graph
        output_dir (str): Directory to save profiling results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running profiling on grid graph of size {graph_size}...")
    G = create_grid_graph(graph_size)
    
    # Run and profile the algorithm
    profiler = cProfile.Profile()
    profiler.enable()
    result = sota_shortest_cycle(G)
    profiler.disable()
    
    # Save raw profiling data
    prof_file = os.path.join(output_dir, f"grid_{graph_size}_profile.prof")
    profiler.dump_stats(prof_file)
    
    # Generate text report
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Print top 30 functions by cumulative time
    
    # Save the report to file
    with open(os.path.join(output_dir, f"grid_{graph_size}_stats.txt"), 'w') as f:
        f.write(s.getvalue())
    
    # Print the report to console
    print("\nProfiling Results Summary:")
    print(f"Shortest cycle length: {result}")
    print("\nTop 10 functions by cumulative time:")
    pstats.Stats(profiler).strip_dirs().sort_stats('cumulative').print_stats(10)
    
    print(f"\nFull profiling data saved to {prof_file}")
    print(f"To analyze further, you can use tools like snakeviz:")
    print(f"    pip install snakeviz")
    print(f"    snakeviz {prof_file}")

def compare_algorithm_variants(graph_size=20, output_dir="profiling_results"):
    """
    Compare different algorithm variants with detailed profiling.
    
    Args:
        graph_size (int): Size of the test graph
        output_dir (str): Directory to save profiling results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Comparing algorithm variants on grid graph of size {graph_size}...")
    G = create_grid_graph(graph_size)
    
    # Define variant descriptions for output
    variants = {
        'traditional': "Traditional approach (edge removal + shortest path)",
        'optimized': "Optimized approach with LCA and pruning",
    }
    
    # Initialize results
    results = {}
    
    # Profile traditional approach
    profiler = cProfile.Profile()
    profiler.enable()
    trad_result = traditional_shortest_cycle(G)
    profiler.disable()
    
    prof_file = os.path.join(output_dir, f"grid_{graph_size}_traditional.prof")
    profiler.dump_stats(prof_file)
    
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    with open(os.path.join(output_dir, f"grid_{graph_size}_traditional_stats.txt"), 'w') as f:
        f.write(s.getvalue())
    
    results['traditional'] = {
        'result': trad_result,
        'prof_file': prof_file
    }
    
    # Profile optimized approach
    profiler = cProfile.Profile()
    profiler.enable()
    opt_result = sota_shortest_cycle(G)
    profiler.disable()
    
    prof_file = os.path.join(output_dir, f"grid_{graph_size}_optimized.prof")
    profiler.dump_stats(prof_file)
    
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    with open(os.path.join(output_dir, f"grid_{graph_size}_optimized_stats.txt"), 'w') as f:
        f.write(s.getvalue())
    
    results['optimized'] = {
        'result': opt_result,
        'prof_file': prof_file
    }
    
    # Print comparison summary
    print("\nAlgorithm Variant Comparison Summary:")
    print(f"Graph: Grid graph with {graph_size}x{graph_size} nodes")
    print(f"Shortest cycle length: {opt_result}")
    
    for variant, data in results.items():
        print(f"\n{variants[variant]}:")
        print(f"  Result: {data['result']}")
        print(f"  Profile data: {data['prof_file']}")
        print("  Top 5 functions by cumulative time:")
        pstats.Stats(data['prof_file']).strip_dirs().sort_stats('cumulative').print_stats(5)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Shortest Cycle Algorithm Benchmarks and Profiling')
    parser.add_argument('--run', choices=['benchmarks', 'profile', 'compare'], default='benchmarks',
                       help='What to run: regular benchmarks, detailed profiling, or algorithm comparison')
    parser.add_argument('--size', type=int, default=20,
                       help='Graph size for profiling (default: 20)')
    
    args = parser.parse_args()
    
    if args.run == 'benchmarks':
        print("Starting benchmarks...")
        run_all_benchmarks()
        print("\nBenchmarking complete!")
    elif args.run == 'profile':
        print("Starting detailed profiling...")
        run_profiling(graph_size=args.size)
        print("\nProfiling complete!")
    elif args.run == 'compare':
        print("Starting algorithm comparison...")
        compare_algorithm_variants(graph_size=args.size)
        print("\nComparison complete!") 