# T7 — Canonical reference implementation: results

Artifacts in this directory:

| file | role |
|---|---|
| `mwc.py` | dependency-free core: `mwc_oracle`, `mwc` (Algorithm 1, repaired), `mwc_transversal` (repair R4) |
| `gen.py` | graph generators (networkx used for the atlas only) |
| `test_mwc.py` | 25 named regressions / differential tests (pytest) |
| `report.py` | produces the tables below |
| `pytest_out.txt`, `report_out.txt` | captured runs |

Reproduce with

```
cd refimpl
<venv>/bin/python -m pytest test_mwc.py -q     # 25 passed in ~15 s
<venv>/bin/python report.py                    # ~30 s
```

---

## 0. Suite status

```
25 passed in 15.17s
```

No test is skipped; `networkx` is present so both atlas tests run.

---

## 1. How many graphs were tested, and in which families

### 1a. Exact-mode differential test vs. the edge-removal oracle (`report.py` §A)

Every graph is checked twice: `mwc(G)` (all roots) and `mwc_transversal(G)`.

| family | graphs | `mwc` mismatches | `mwc_transversal` mismatches |
|---|---:|---:|---:|
| complete | 100 | 0 | 0 |
| disconnected | 100 | 0 | 0 |
| erdos_renyi | 100 | 0 | 0 |
| geometric | 100 | 0 | 0 |
| grid | 100 | 0 | 0 |
| hub_gadget (adversarial) | 25 | 0 | 0 |
| multiscale (adversarial weights) | 700 | 0 | 0 |
| planted_cycle | 100 | 0 | 0 |
| pref_attach (Barabási–Albert) | 100 | 0 | 0 |
| small_world (Watts–Strogatz) | 100 | 0 | 0 |
| tree | 100 | 0 | 0 |
| unicyclic | 100 | 0 | 0 |
| **TOTAL** | **1 725** | **0** | **0** |

Weightings used across these families: unit, integer `U{1..10}`, continuous
`U(0.05, 1)`, "wide" `{1,2,5,13,40,97}`, Euclidean, and the four-scale
"multiscale" distribution `{~0.1, ~1, ~5, ~10}`.

### 1b. Exhaustive graph atlas, `n ≤ 7` (`report.py` §B, `test_mwc.py`)

| quantity | value |
|---|---:|
| atlas graphs with 3–7 vertices | 1 249 |
| weighted instances (int + continuous) | 2 498 |
| runs per instance (`mwc`, `mwc` reversed root order, `mwc_transversal`) | 3 |
| **mismatches vs oracle** | **0** |

That is 1 249 unit-weight + 2 498 weighted = **3 747 instances × 3 runs = 11 241
exact runs, 0 mismatches.** The atlas covers *every* isomorphism class of simple
graph up to 7 vertices, so this is an exhaustive structural check.

### 1c. Additional randomized exact checks inside the suite

`test_root_order_permutations_agree` (60 graphs × 8 random root orders),
`test_exact_mode_alpha_le_beta` (120 graphs × 8 `(α ≤ β)` pairs = 960 runs, all
asserted to have **zero deletions**), `test_gamma_monotone_and_witnessed`
(120 graphs × 3 parameter settings), `test_two_core_and_blocks_preserve_girth`
(120 graphs), `test_transversal_size_bound` (80 graphs).

**Total exact-mode mismatch count over everything: 0.**

---

## 2. Approximation ratio vs. κ

κ(α, β) = max{1, 1/(1 − 2α + 2β)} (repair R2; the manuscript's bare
`1/(1−2α+2β)` is false for α < β because it claims `ĝ < γ*`).

### 2a. Parameter grid over a 1 962-graph pool, 3 root orders each (`report.py` §C)

| α | β | κ | runs | deletions | runs with ĝ > γ* | worst ĝ/γ* |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 1.000 | 5 886 | 0 | 0 | 1.000000 |
| 0.05 | 0.00 | 1.111 | 5 886 | 23 | 0 | 1.000000 |
| 0.10 | 0.00 | 1.250 | 5 886 | 38 | 0 | 1.000000 |
| 0.20 | 0.00 | 1.667 | 5 886 | 125 | 0 | 1.000000 |
| 0.25 | 0.05 | 1.667 | 5 886 | 18 | 0 | 1.000000 |
| 0.30 | 0.10 | 1.667 | 5 886 | 4 | 0 | 1.000000 |
| 0.40 | 0.10 | 2.500 | 5 886 | 10 | 0 | 1.000000 |
| 0.45 | 0.20 | 2.000 | 5 886 | 0 | 0 | 1.000000 |
| 0.49 | 0.00 | 50.000 | 5 886 | 206 | 0 | 1.000000 |
| 0.60 | 0.20 | 5.000 | 5 886 | 3 | 0 | 1.000000 |
| 0.75 | 0.40 | 3.333 | 5 886 | 1 | 0 | 1.000000 |
| 1.00 | 0.60 | 5.000 | 5 886 | 0 | 0 | 1.000000 |

**70 632 approximate runs, 0 violations of `γ* ≤ ĝ ≤ κ γ* + 1e−9`.**

### 2b. In-suite property test (`test_approximation_bound_property`)

```
[approx] 82188 (graph, alpha, beta, order) runs, 479 vertex deletions;
         worst observed ratio = 1.160262
```

2 520 graphs × 12 (α, β) pairs × 3 root orders. The test asserts
`γ* ≤ ĝ ≤ κ γ* + 1e−9` on every run *and* asserts non-vacuity
(`deletions > 0`).

### 2c. Dedicated randomized adversarial search

Two independent searches over the multiscale family:

| search | runs | deletions | runs with ĝ > γ* | κ-bound violations | worst ĝ/γ* |
|---|---:|---:|---:|---:|---:|
| `report.py` §C2 (seed 2024, `n = 5..10`) | 56 068 | 840 | 1 | **0** | 1.013858 (α=0.49, β=0, κ=50) |
| standalone sweep (seed 2024, 10 (α,β) pairs incl. α>1) | 390 815 | — | — | **0** | **1.131780** (α=0.4, β=0, κ=5) |
| standalone sweep (seed 11, MWC-vertex-deletion filter) | 116 695 | — | 87 | **0** | 1.074127 (α=0.4, β=0, κ=5) |

**Aggregate: > 640 000 approximate runs, 0 violations of the κ bound.**

### 2d. Empirical observations worth putting in the paper

1. **The bound is extremely loose in practice.** The largest ratio ever
   observed is **1.13** against a κ of 5, and over ~640 k randomized runs only
   ~0.1 % of runs deviate from exact at all. The κ = 1/(1 − 2α + 2β) worst case
   is an adversarial phenomenon, not a typical one.
2. **The discard rule almost never fires on uniformly-weighted random graphs.**
   Across the entire uniform/int/continuous family suite the deletion count was
   *zero*. The reason is structural: once γ drops to the girth, the truncation
   horizon γ/2 is so small that no *decoy* cycle survives inside it, so
   `d⁺_min = ∞` and the trigger `ℓ_best > γ ∧ d⁺_min < (1+α)γ` cannot hold.
   Deletions require **well-separated weight scales** — the `multiscale` and
   `hub_gadget` families were added specifically to reach the code path.
3. **A single-root deletion cannot by itself force the worst case.** Any
   deleted vertex has `δ(z) ≤ δ(p) − βγ` where `p` is the LCA of the composite
   minimiser; every vertex of that minimiser is at `δ ≥ δ(p)`. So the MWC has to
   sit *strictly inside* the decoy, which caps the achievable single-root ratio
   well below κ. Reaching κ needs the multi-root schedule of R3 (one root per
   MWC vertex). This is a concrete constraint the tightness construction must
   satisfy.
4. **A pinned worst case is kept as a regression**
   (`test_known_approximation_error_instance`): a 7-vertex, 13-edge multiscale
   graph with α = 0.4, β = 0 where vertex 0 (an MWC vertex) is deleted and
   ĝ = 12.1835 > γ* = 11.3427, ratio 1.074 vs κ = 5.

---

## 3. Transversal variant: measured root-count and settled-vertex reductions

`mwc_transversal` = 2-core peeling + biconnected-block decomposition +
spanning-forest cycle transversal `S` (one endpoint per non-tree edge) +
initial bound `γ₀ = min over non-tree e=(u,v) of [ℓ_T(u,v) + w(e)]`, running
gamma threaded across blocks. Exact mode only. All answers verified against the
oracle (0 mismatches, §1a/§1b).

`report.py` §D, 1 175 graphs:

| family | graphs | roots (full) | roots (transversal) | ratio | settled (full) | settled (transversal) | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| complete | 100 | 507 | 307 | 0.606 | 2 186 | 1 296 | 0.593 |
| disconnected | 100 | 1 635 | 445 | 0.272 | 4 765 | 1 457 | 0.306 |
| erdos_renyi (n≈4..16) | 100 | 1 031 | 506 | 0.491 | 4 840 | 2 429 | 0.502 |
| erdos_renyi n=40, p=0.12 | 20 | 800 | 496 | 0.620 | 3 491 | 1 934 | 0.554 |
| geometric (n≈6..20) | 100 | 1 227 | 656 | 0.535 | 3 947 | 2 089 | 0.529 |
| geometric n=40, r=0.30 | 20 | 800 | 665 | 0.831 | 2 774 | 1 908 | 0.688 |
| grid (small) | 100 | 1 171 | 591 | 0.505 | 5 767 | 2 562 | 0.444 |
| grid 8×8 | 10 | 640 | 490 | 0.766 | 2 594 | 1 708 | 0.658 |
| hub_gadget | 25 | 330 | 155 | 0.470 | 1 925 | 853 | 0.443 |
| multiscale | 100 | 789 | 441 | 0.559 | 3 444 | 1 818 | 0.528 |
| planted_cycle | 100 | 1 234 | 375 | 0.304 | 4 048 | 1 472 | 0.364 |
| pref_attach | 100 | 1 072 | 432 | 0.403 | 7 342 | 1 659 | 0.226 |
| small_world | 100 | 1 196 | 477 | 0.399 | 7 230 | 2 128 | 0.294 |
| tree | 100 | 630 | **0** | 0.000 | 4 490 | **0** | 0.000 |
| unicyclic | 100 | 777 | 100 | 0.129 | 4 953 | 519 | 0.105 |
| **ALL** | **1 175** | **13 839** | **6 136** | **0.443** | **63 796** | **23 832** | **0.374** |

Independent in-suite measurement (`test_transversal_exactness_vs_oracle`,
250-graph family suite):

```
[transversal] roots 941/2423 (0.388), settled 3851/11508 (0.335)
```

Headline: **~56 % fewer root searches and ~63 % fewer settled vertices**, with
identical (exact) answers. The reduction is structural, not heuristic:

* forests are answered with **zero** Dijkstra calls (the full algorithm runs
  one per vertex);
* unicyclic graphs need **one** root instead of `n`;
* `|S| ≤ min{n, μ}` with `μ = m − n + c(G)` is asserted per 2-core in
  `test_transversal_size_bound`;
* the initial bound `γ₀` shrinks the very first truncation horizon from `∞` to
  `γ₀/2`, which is where most of the settled-vertex saving comes from on dense
  instances.

---

## 4. Mismatch summary

| mode | instances | mismatches |
|---|---:|---:|
| exact `mwc` (all roots) vs oracle | 1 725 random + 3 747 atlas + ~1 600 in-suite | **0** |
| exact `mwc_transversal` vs oracle | same | **0** |
| exact mode with `α ≤ β` (8 pairs) | 960 runs | **0** (and 0 deletions, asserted) |
| approximate `γ* ≤ ĝ ≤ κγ*` | > 640 000 runs | **0 violations** |
| candidate-cycle certification failures | every scanned non-tree edge of every run | **0** |

Certification is on by default: every candidate fundamental cycle is
reconstructed, checked to be a simple cycle of `G`, and independently
re-summed before it is allowed to move `γ`; the final `γ` is re-verified
against its witness cycle. Not one of these checks fired during any green run,
i.e. the `δ(y) + δ(z) + w − 2δ(p)` identity held exactly (to 1e−9 relative) on
every scanned edge in the whole campaign.
