# Reference implementation — minimum weight cycle via composite distance

This directory is the **canonical, theorem-matching implementation** for the paper
*Minimum Weight Cycles via Composite Distance: Safe Truncation, Cycle-Transversal Roots,
and a Tight Approximation Frontier*. It follows Algorithm 1 of the manuscript line by line.

It is deliberately kept **separate from the files in the repository root**
(`shortest_cycle.py`, `proposed_algorithm.py`, `hybrid_mwc.py`, `bmssp_*.py`), which are
earlier/experimental code. Those files do **not** implement the algorithm as analysed in
the paper, and several of them are incorrect — see `AUDIT_legacy_vs_oracle.txt` and the
summary below. No result in the paper is produced by them.

## Contents

| file | role |
|---|---|
| `mwc.py` | core, dependency-free (stdlib only). `mwc_oracle`, `mwc`, `mwc_transversal`. |
| `gen.py` | graph generators used by the tests (networkx only for the graph atlas). |
| `test_mwc.py` | 25 named regressions + differential tests. |
| `transversal_study.py` | reproduces the root-count / settled-vertex reduction table. |
| `alg1_exact_rational.py` | independent Algorithm-1 simulator in exact rational arithmetic; takes `gamma0` so it can model line 2. |
| `tightness_witness.py` | the 14-vertex tightness witness, SEED-RESISTANT (see below), checked in both implementations. |
| `sharpness_witness.py` | the witnesses for Proposition (Sharpness), parts (i) and (ii). |
| `RESULTS.md` | measured validation campaign. |
| `AUDIT_legacy_vs_oracle.txt` | differential audit of the legacy code paths. |

### Experiments campaign (2026-08)

| file | role |
|---|---|
| `adaptive.py` | `mwc_adaptive` — picks the root set at run time from `mu >= m-n+1`, decided in O(1); also the A0–A7 ablation ladder. |
| `campaign.py` | driver: `timing`, `ablation`, `theta`, `frontier`, `argmin`, `real`. Every subcommand cross-checks exactness and records provenance. |
| `make_tables.py` | turns the campaign JSON into the paper's LaTeX tables, stamped with source file, commit, job id and CPU. |
| `plot_argmin.py` | the argmin figure, emitted at its final typeset width so LaTeX does not rescale the labels. |
| `probe_blocks.py` | when does the block decomposition pay? (measured: not on any graph tried). |
| `probe_stats_bias.py` | does `collect_stats=True` flatter the low-root configurations? (measured: by under 10%). |
| `campaign_array_v2.slurm`, `real_array.slurm` | SLURM job arrays; one task per experiment, one task per real network. |
| `realnets/` | the ten real networks, uniform `u v w` edge lists plus per-network JSON recording source, weight semantics and preprocessing; `fetch_all.py` re-acquires them. |

Reproduce the measured tables with, e.g.

```
python campaign.py --out results --repeats 5 timing    --sizes 400 900 1600
python campaign.py --out results --repeats 5 ablation  --sizes 400 900 1600
python campaign.py --out results --repeats 3 frontier  --sizes 200 400 800
python campaign.py --out results --repeats 5 real --data realnets --only sydney-road --tag sydney-road
python make_tables.py --results results --out <manuscript>/tables
```

## Running

```bash
python -m pytest test_mwc.py -q      # 25 passed
python transversal_study.py          # root/settled reduction table
python tightness_witness.py          # ratio -> kappa(alpha,beta); asserts gamma_0 = 1
python sharpness_witness.py          # radius 1/2 optimal; component-meeting root set fails
```

Only `test_mwc.py` (graph atlas) and `gen.py` need `networkx`; `mwc.py` itself has no
third-party dependencies and operates on a plain adjacency dict `{u: {v: w}}`.

## What this implementation does that the legacy code did not

1. **Searches the active induced subgraph** `H = G[A]` in approximate mode. The legacy code
   gated only *root selection* by the active set while still running Dijkstra on the full
   graph, so discarded vertices kept relaxing edges.
2. **Skips only predecessor-tree edges.** The legacy guard `if p and p != u and p != v`
   both (a) silently skipped every fundamental cycle whose LCA is the vertex labelled `0`,
   because `0` is falsy in Python, and (b) wrongly rejected valid ancestor-to-non-child
   chords. Here the test is `p is not None` plus an explicit tree-edge check.
3. **Implements the `(alpha, beta)` discard rule literally**, and asserts that no deletion
   occurs when `alpha <= beta`. The legacy rule was `dist(v) + 2*w_min >= gamma`, which is
   a different rule and is unsafe: with component A a cycle of length 10 and component B a
   vertex `x` joined to a unit triangle by weight-8 edges, it deletes the whole triangle
   because `8 + 2*1 >= 10`, and the length-3 cycle is never found.
4. **Returns the cycle, not just a length**, reconstructs it, checks it is a simple cycle of
   `G`, and independently re-sums its weight before letting it move `gamma`. A length-only
   interface makes an entire class of defect invisible — one legacy backend returned an edge
   set that was not a cycle at all in 94 of 94 cases tested, and those edge sets were being
   written into the Loop Modulus QP constraint matrix unvalidated.
5. **Sizes the binary-lifting table from `|Q|`** rather than hard-coding height 20.
6. **Uses a two-phase root search** — truncated Dijkstra at a frozen radius `gamma_in/2`,
   then a scan of the non-tree edges induced on the settled set — so that `Gamma` is well
   defined and the discard test provably sees the post-search bound.

## The approximation constant

```python
kappa(alpha, beta) = max(1, 1 / (1 - 2*min(alpha, 0.5) + 2*beta))
```

The `min(alpha, 0.5)` is not cosmetic. The record-setting LCA must itself have been settled,
so every unsettled vertex is at distance `>= d_to_cycle`, not merely `>= Gamma/2`; that gives
the extra inequality `2*beta*Gamma <= gamma*`, capping the trigger parameter at `1/2`. The
unclipped `1/(1 - 2*alpha + 2*beta)` is a valid but loose bound — at `(0.7, 0.25)` it claims
10 where the truth is 2 — and it is outright false for `alpha < beta`, where it would assert
a returned cycle shorter than the minimum.

## Validation status

* Exact mode vs. the edge-removal oracle: **0 mismatches** over 1 725 random instances across
  12 families plus the complete graph atlas on <= 7 vertices (3 747 weighted instances x 3 runs).
* Approximate mode: **0 violations** of `gamma* <= ghat <= kappa*gamma*` in > 640 000 runs.
* Tightness: the 14-vertex witness drives `ghat/gamma*` to `kappa` at every admissible
  `(alpha, beta)`, confirmed by two independent implementations, **with the seed of line 2
  enabled** (`gamma_0 = 1`, asserted per configuration).
* Cycle certification (reconstruct + re-sum) never fired a mismatch in the whole campaign.

The knife edge is now removed rather than merely noted: `D` sits `eps/8` strictly inside the
deletion threshold instead of exactly on it. The old placement was decided by an exact real
equality that floating point need not reproduce — at `(alpha, beta) = (0.5, 0.1)` the
threshold evaluates to 0.39749999999999996 against `delta(v0) = 0.3975`, `v0` survives, and
the ratio collapses to 1.


## A trap worth knowing about: the witnesses must survive the seed

Line 2 of Algorithm 1 sets `gamma_0` to the smallest fundamental cycle of a
spanning forest, before any root search. If the minimum weight cycle `c*` is a
whole component, or is pendant (attached at one vertex), then `c*` minus any one
edge is a path, so every spanning forest contains all but one edge of `c*` and
the omitted edge closes `c*` itself. Then `gamma_0 = gamma*` exactly and the
algorithm returns the optimum before the discard rule can do anything -- such a
graph proves nothing about the approximation ratio.

The earlier `tightness_13vertex.py` had exactly this shape. Measured on it at
(alpha, beta, eps) = (0.3, 0, 1e-2): `gamma_0 = gamma* = 0.410000`, ratio 1.000
against kappa = 2.5.

`tightness_witness.py` and `sharpness_witness.py` therefore attach `c*` to a hub
at two antipodal vertices by heavy edges and control the vertex presentation
order, so the BFS forest enters `c*` from both sides and leaves two of its edges
non-tree. Both scripts ASSERT `gamma_0 == 1` and fail loudly otherwise; keep that
assertion if you modify them. Note also that no witness can defeat every spanning
forest -- for any cycle there is some tree making it fundamental -- which is why
the forest rule is fixed, exactly as the root order is.
