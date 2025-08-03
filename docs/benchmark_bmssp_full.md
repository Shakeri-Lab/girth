# BMSSP-Full (first recursive version)

| n nodes | time (s) |
|---------|----------|
| 1 000   | 0.013 |
| 2 000   | 0.032 |

Block size tuned: `block_size = B / (k·log n)`  ⇒ noticeable improvement over previous ~0.60 s (binary heap) run.

(2 trials each on same machine; values stable ±5 ms.)
