# Shortest Cycle Algorithm

This project implements an efficient algorithm for finding the shortest cycle in an undirected graph with non-negative edge weights. The algorithm leverages advanced optimization techniques including Lowest Common Ancestor (LCA) trees with binary lifting, priority queues, aggressive pruning, and smart caching.

## Installation

To use this algorithm, you need Python 3.6 or higher installed on your system. The project relies on the following dependencies:

- **`networkx`**: For graph creation, manipulation, and built-in algorithms.
- **`numpy`**: For efficient numerical computations.
- **`matplotlib`**: For visualizing benchmark results (optional).
- **`cProfile`** and **`pstats`**: For performance profiling (included in standard library).

Install these dependencies using pip:

```bash
pip install networkx numpy matplotlib
```

For optimal performance with the Fibonacci heap, you can install the optional external implementation:

```bash
pip install fibonacci-heap-mod
```

This package is optional and the algorithm will work fine without it. If this package is not installed, the algorithm will automatically fall back to an internal implementation.

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/shortest-cycle-algorithm.git
cd shortest-cycle-algorithm
```

## Usage

The primary function, `sota_shortest_cycle`, accepts a `NetworkX` graph as input and returns the length of the shortest cycle. Below is a simple example to get you started.

### Example

```python
import networkx as nx
from shortest_cycle import sota_shortest_cycle

# Create a sample undirected graph
G = nx.Graph()
G.add_edge(0, 1, weight=1)
G.add_edge(1, 2, weight=2)
G.add_edge(2, 0, weight=3)

# Compute the shortest cycle length
cycle_length = sota_shortest_cycle(G)
print(f"The shortest cycle length is: {cycle_length}")
```

**Output:**
```
The shortest cycle length is: 6.0
```

This example creates a triangle graph with edges `(0, 1)`, `(1, 2)`, and `(2, 0)` and finds the shortest cycle, which is the perimeter of the triangle (1 + 2 + 3 = 6).

### Advanced Usage

The project also provides additional utility functions:

- **`shortest_cycle_nodes(G)`**: Returns the list of nodes that form the shortest cycle.
- **`visualize_shortest_cycle(G, cycle)`**: Visualizes the graph and highlights a given cycle.
- **`create_grid_graph(size, default_weight, cycle_weight)`**: Creates a grid graph with a hidden low-weight cycle for testing.
- **`create_spatial_graph(n_nodes, radius, default_weight, cycle_weight)`**: Creates a random geometric graph with a hidden low-weight cycle.

Example with visualization:

```python
import networkx as nx
from shortest_cycle import sota_shortest_cycle, shortest_cycle_nodes, visualize_shortest_cycle

# Create a sample graph
G = nx.Graph()
G.add_edge(0, 1, weight=1)
G.add_edge(1, 2, weight=2)
G.add_edge(2, 3, weight=1)
G.add_edge(3, 0, weight=1)
G.add_edge(0, 2, weight=2.5)

# Get the shortest cycle length
cycle_length = sota_shortest_cycle(G)
print(f"The shortest cycle length is: {cycle_length}")

# Get the nodes in the shortest cycle
cycle_nodes = shortest_cycle_nodes(G)
print(f"The shortest cycle consists of nodes: {cycle_nodes}")

# Visualize the graph with the shortest cycle highlighted
visualize_shortest_cycle(G, cycle_nodes)
```

### Optimization Options

The algorithm supports several optimization options that can be enabled or disabled:

```python
# Default configuration (using standard heap with LCA optimization)
cycle_length = sota_shortest_cycle(G)

# Enable Fibonacci heap (if available)
cycle_length = sota_shortest_cycle(G, use_fib_heap=True, use_lca=True)

# Use LCA optimization with standard binary heap
cycle_length = sota_shortest_cycle(G, use_fib_heap=False, use_lca=True)

# Use traditional approach without LCA optimization
cycle_length = sota_shortest_cycle(G, use_lca=False)
```

The `use_fib_heap` parameter enables the Fibonacci heap implementation for priority queue operations, which can offer better theoretical performance for large graphs. There are two implementations available:

1. **Internal Implementation**: Used by default or if the external package is not installed.
2. **External Implementation**: Used if the `fibonacci-heap-mod` package is installed and `use_fib_heap=True`.

To use the external implementation, first install the package:

```bash
pip install fibonacci-heap-mod
```

Then set `use_fib_heap=True` when calling the function. The algorithm will automatically detect if the package is available and use it; otherwise, it will fall back to the internal implementation.

## Profiling and Analysis

The project includes tools for profiling and analyzing the algorithm's performance. This is particularly useful for understanding bottlenecks, comparing optimization strategies, and analyzing complexity scaling with graph size.

### Basic Profiling

To profile the algorithm's performance:

```python
import cProfile
import pstats
from shortest_cycle import sota_shortest_cycle, create_grid_graph

# Create a test graph
G = create_grid_graph(20, default_weight=10, cycle_weight=1)

# Profile the algorithm with Fibonacci heap optimization (if available)
profiler = cProfile.Profile()
profiler.enable()
result = sota_shortest_cycle(G, use_fib_heap=True, use_lca=True)
profiler.disable()

# Print the profiling results
profiler.dump_stats('profile_data.prof')
p = pstats.Stats('profile_data.prof')
p.strip_dirs().sort_stats('cumulative').print_stats(10)
```

### Using the Profiling Script

A dedicated profiling script is provided for more detailed analysis:

```bash
# Basic profiling on a grid graph of size 20x20
python profile_algorithm.py --mode profile --graph grid --size 20

# Compare different optimization strategies
python profile_algorithm.py --mode compare --graph grid --size 15

# Analyze scaling with graph size
python profile_algorithm.py --mode scale --graph spatial --use-fib-heap
```

### Profiling Options

The profiling script supports the following options:

- **Profiling modes**:
  - `profile`: Single profiling run with detailed stats
  - `compare`: Compare different optimization combinations
  - `scale`: Analyze how performance scales with graph size

- **Graph types**:
  - `grid`: Regular grid graphs
  - `spatial`: Random geometric graphs

- **Optimization flags**:
  - `--use-fib-heap`: Use Fibonacci heap implementation
  - `--use-lca`: Use LCA optimization (default: True)

### Visualizing Results

The profiling script generates visualizations to help understand performance:

1. **Optimization comparison**: Bar chart comparing execution times for different optimization combinations
2. **Scaling analysis**: Line plots showing how execution time and operation count scale with graph size

### Advanced Analysis

For more advanced analysis, the profiling script saves raw profiling data in `.prof` files that can be analyzed with external tools:

```bash
# Use snakeviz for interactive visualization
pip install snakeviz
snakeviz profiling_results/grid_20x20_lca_fib.prof
```

## Algorithm Explanation

The algorithm uses several advanced techniques to efficiently find the shortest cycle:

### 1. Lowest Common Ancestor (LCA) with Binary Lifting

Instead of the traditional approach of removing each edge and running Dijkstra's algorithm, this implementation:

1. Runs Dijkstra's algorithm from each node, generating a shortest-path tree.
2. Constructs an LCA data structure with binary lifting, allowing O(log n) queries to find the lowest common ancestor of any two nodes.
3. For each edge (u, v), uses the LCA to determine the cycle formed with the shortest-path tree in O(log n) time.

### 2. Early Termination and Pruning

- **Dijkstra Early Termination**: Stops Dijkstra's algorithm once all relevant nodes are processed, using the current shortest cycle length (gamma) as a bound.
- **Node Pruning**: After processing a node, prunes nodes that cannot possibly contribute to a shorter cycle, reducing work in subsequent iterations.

### 3. Efficient Data Structures

- **Binary Heap**: Implements Dijkstra's algorithm with a binary heap for efficient priority queue operations.
- **Fibonacci Heap Option**: Optional Fibonacci heap implementation for potentially improved performance.
- **LCA Caching**: Caches LCA query results to avoid redundant computations.
- **Ancestor Table**: Uses dynamic programming to precompute ancestors for fast LCA queries.

## Performance and Complexity

- **Time Complexity**: O(V·(E + V log V)) in the worst case, but with early termination and pruning, the practical performance is often much better.
- **Space Complexity**: O(V log V) for the LCA data structure and O(V) for Dijkstra's algorithm.

The algorithm excels on sparse graphs, where aggressive pruning significantly reduces the number of nodes processed in each iteration.

## Testing

A comprehensive test suite is included to verify the algorithm's correctness across various scenarios. To run the tests, install `pytest` and execute:

```bash
pip install pytest
pytest test_shortest_cycle.py
```

The tests cover:
- Standard graph structures (triangles, squares, complete graphs)
- Graphs with hidden low-weight cycles
- Edge cases (empty graphs, disconnected graphs)
- Comparison against traditional approaches for correctness

## Project Structure

```
shortest-cycle-algorithm/
├── shortest_cycle.py     # Main algorithm implementation
├── test_shortest_cycle.py # Test suite for verification
├── benchmark.py          # Performance benchmarking script
├── profile_algorithm.py  # Detailed profiling tool
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Limitations

- **Non-Negative Weights**: The algorithm assumes edge weights are non-negative, as Dijkstra's algorithm does not support negative weights.
- **Undirected Graphs**: Designed specifically for undirected graphs; directed graphs require a different approach.
- **Memory Usage**: The LCA data structure has slightly higher memory requirements than the traditional approach.

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a feature or bugfix branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes with descriptive messages:
   ```bash
   git commit -m "Add feature X to improve Y"
   ```
4. Push your branch and open a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

Ensure your code passes all tests and follows the project's style guidelines.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this algorithm in your work, please cite the original manuscript (update with actual details when available):

> [Author Name]. (Year). Finding a minimum weight cycle in undirected graphs using LCA and pruning techniques. *[Journal/Conference Name]*, *[Volume(Issue)]*, *[Pages]*. 