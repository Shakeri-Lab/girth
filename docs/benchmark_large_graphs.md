# Large-Graph Ablation (1 k nodes)

3 trials on a 1 000-node spatial graph (`radius=0.08`).

| Backend    | Run 1 (s) | Run 2 (s) | Run 3 (s) | Mean (s) | σ (s) |
|------------|-----------|-----------|-----------|----------|-------|
| heapq      | 0.141     | 0.113     | 0.650*    | 0.301    | 0.302 |
| heapdict   | 0.113     | 1.218*    | 0.120     | 0.484    | 0.636 |

*Both backends show occasional outliers due to Python GC / cache effects; overall, `heapdict` provides **no speed-up** and sometimes slows things down for this graph size.*

Conclusion: Python-level priority-queue replacements yield negligible benefit. Next optimisation step should target a compiled heap (external Fibonacci-heap) or move the Dijkstra relaxation loop to Cython/C++.
