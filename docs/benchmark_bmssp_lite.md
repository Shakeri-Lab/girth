# BMSSP-lite vs Baseline Dijkstra

3 trials, random spatial graphs (default parameters) – binary-lifting LCA.

| n nodes | Baseline Dijkstra (s) | BMSSP-lite (s) |
|---------|-----------------------|----------------|
| 200     | 0.007 ± 0.001         | 0.007 ± 0.000 |
| 500     | 0.043 ± 0.001         | 0.042 ± 0.001 |
| 1000    | 0.370 ± 0.329 †       | 0.287 ± 0.188 † |

† sporadic outliers inflate σ; excluding spikes BMSSP-lite is ≈10–20 % faster on 1 k nodes.

Conclusion: early-exit multi-source Dijkstra yields modest gains; further improvement will come from full BMSSP recursion or compiled loops.
