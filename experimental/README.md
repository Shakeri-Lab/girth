# experimental/ — quarantined back-ends

The code in this directory is **not correct** and is not used by any result in the paper
*Minimum Weight Cycles via Composite Distance*. It is kept for reference and possible
repair, not for use. For a validated implementation see `../reference/`.

## Why these were quarantined

Differential testing against the exact edge-removal oracle
`gamma* = min over e=(u,v) of [w(e) + d_{G-e}(u,v)]`
on 400 random graphs with `4 <= n <= 9` (the oracle itself cross-checked against
exhaustive cycle enumeration, 0 disagreements):

| back-end | disagreements with the oracle |
|---|---|
| BMSSP-full (`bmssp_full.py`, `block_pq.py`, `bmssp_core.*`) | **91.5%** (366/400) |
| Euler-tour LCA (`lca_euler.py`) | **69.5%** (278/400), plus 104 raised exceptions |

Two findings made the quarantine urgent rather than cosmetic.

1. **BMSSP-full was the silent default.** `hybrid_mwc_length()` selected
   `BMSSPFullStrategy()` whenever `use_bmssp_lite` was false — which is the default — so
   every caller that did not explicitly opt out got the 91.5%-wrong path. In particular
   `loop_modulus/utils_shortest.py` calls `hybrid_mwc_length(...)` without setting
   `use_bmssp_lite`, so the Loop Modulus constraint generator was running on it.

2. **It returned objects that were not cycles.** With `return_edges=True`, BMSSP-full
   returned an edge set that is not a simple cycle of `G` in **94 of 94** cases checked.
   Example, on an 8-edge graph:

   ```
   G edges (u,v,w): (0,6,3) (2,3,4) (2,5,4) (2,6,3) (3,4,2) (3,5,5) (4,6,7) (5,6,9)
   returned edge set: [(2,5), (2,6), (3,4), (3,5), (4,6)]
   reported length  : 7.0        true girth: 13.0
   ```

   Those edge sets became rows of the Loop Modulus quadratic program without validation,
   so an infeasible constraint could enter the optimisation.

A minimal reproduction of the underlying distance bug: on a 6-cycle with unit weights,
BMSSP-full computes distances from source 1 as `{1:0, 2:0, 3:1, 4:inf, 5:1, 6:0}` where
the correct answer is `{1:0, 2:1, 3:2, 4:3, 5:2, 6:1}`.

## What changed in the caller

`hybrid_mwc.py` no longer selects BMSSP-full implicitly. With this directory absent from
`sys.path`, the guarded import leaves `BMSSPFullStrategy = None` and `HybridMWC` falls back
to its baseline Dijkstra strategy. Requesting `use_euler_lca=True` now raises an
`ImportError` that explains why rather than failing obscurely.

## Running anything in here

These modules import from the repository root (`from hybrid_mwc import SSSPStrategy`), so
they need the parent directory on `sys.path`:

```bash
PYTHONPATH=.. python -c "import bmssp_full"
```

If you repair one of them, validate it against `../reference/mwc.py::mwc_oracle` before
promoting it back — `../reference/AUDIT_legacy_vs_oracle.txt` is the harness that produced
the table above.
