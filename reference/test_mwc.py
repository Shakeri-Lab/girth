"""Differential test harness for the canonical MWC reference implementation.

Run with::

    <venv>/bin/python -m pytest test_mwc.py -q

Every named regression corresponds to a defect that was found either in the
manuscript's Algorithm 1 or in the shipped ``shortest_cycle.py``.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

import gen
from mwc import (
    INF,
    CertificationError,
    LCAStructure,
    biconnected_components,
    check_graph,
    cycle_weight,
    is_simple_cycle,
    kappa_of,
    mwc,
    mwc_oracle,
    mwc_transversal,
    two_core,
    _dijkstra_avoid_edge,
    _index_map,
    _truncated_dijkstra,
)

TOL = 1e-9


def approx_eq(a, b, tol=TOL):
    if a == b:
        return True
    if math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def plain_dijkstra(adj, src):
    dist, _ = _dijkstra_avoid_edge(adj, src, _index_map(adj))
    return dist


# ===========================================================================
# 1.  "fully settled but not fundamental"  (manuscript defect #1 / Lemma 3)
# ===========================================================================

def square_with_far_root():
    """Unit square a,b,c,d; root x joined to a and c with weight 10."""
    return gen.from_edges(
        ["x", "a", "b", "c", "d"],
        [("a", "b", 1), ("b", "c", 1), ("c", "d", 1), ("d", "a", 1),
         ("x", "a", 10), ("x", "c", 10)],
    )


def test_fully_settled_but_not_fundamental_square():
    G = square_with_far_root()
    check_graph(G)
    g_star, c_star = mwc_oracle(G)
    assert approx_eq(g_star, 4.0)

    # The whole square is settled from x, yet it is NOT a fundamental cycle of
    # the shortest-path tree rooted at x (which contains both xa and xc).
    delta, pred, Q = _truncated_dijkstra(G, "x", set(G), INF, _index_map(G))
    assert set(Q) == set(G)                       # everything settled
    assert pred["a"] == "x" and pred["c"] == "x"  # square not fundamental

    # Detected composite score from x is 22, the TRUE d^+(x) is 10 + 4 = 14.
    from_x = mwc(G, roots={"x"})
    assert approx_eq(from_x.length, 22.0), from_x.length
    true_d_plus_x = min(plain_dijkstra(G, "x")[v] for v in "abcd") + 4.0
    assert approx_eq(true_d_plus_x, 14.0)
    assert from_x.length > true_d_plus_x   # <- the manuscript's defect #1

    # ... but the full algorithm still returns 4 globally.
    for order in itertools.permutations(["x", "a", "b", "c", "d"]):
        res = mwc(G, root_order=list(order))
        assert approx_eq(res.length, 4.0), (order, res.length)
        assert is_simple_cycle(G, res.cycle)
        assert approx_eq(cycle_weight(G, res.cycle), 4.0)
        assert res.mode == "exact" and res.certified


# ===========================================================================
# 2.  LCA equal to the vertex labelled 0  (truthiness bug: `if p and ...`)
# ===========================================================================

def test_lca_is_vertex_zero():
    # triangle 0-1-2; from root 0 the closing edge (1,2) has LCA exactly 0.
    G = gen.from_edges(3, [(0, 1, 1), (1, 2, 1), (2, 0, 1)])
    delta, pred, Q = _truncated_dijkstra(G, 0, set(G), INF, _index_map(G))
    lca = LCAStructure(Q, pred, 0)
    assert lca.lca(1, 2) == 0            # falsy label!
    assert bool(lca.lca(1, 2)) is False  # exactly what `if p` throws away
    res = mwc(G, root_order=[0, 1, 2])
    assert approx_eq(res.length, 3.0)
    assert approx_eq(mwc_oracle(G)[0], 3.0)

    # A bigger instance where vertex 0 is the LCA of an interior pair.
    H = gen.from_edges(
        7,
        [(0, 1, 1), (1, 2, 1), (0, 3, 1), (3, 4, 1), (2, 4, 1),
         (0, 5, 20), (5, 6, 20), (6, 0, 20)],
    )
    delta, pred, Q = _truncated_dijkstra(H, 0, set(H), INF, _index_map(H))
    lca = LCAStructure(Q, pred, 0)
    assert lca.lca(2, 4) == 0
    assert approx_eq(mwc(H, root_order=list(range(7))).length, mwc_oracle(H)[0])
    assert approx_eq(mwc_oracle(H)[0], 5.0)


# ===========================================================================
# 3.  non-parent ancestor/descendant non-tree edge
#     (`if p and p != u and p != v` drops these)
# ===========================================================================

def test_ancestor_descendant_non_tree_edge():
    # 0-1-2-3 path, plus the heavy chord 0-3: LCA(0,3) = 0 = an endpoint and
    # 0 is NOT pred(3).
    G = gen.from_edges(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, 10)])
    delta, pred, Q = _truncated_dijkstra(G, 0, set(G), INF, _index_map(G))
    assert pred[3] == 2                    # not 0 -> genuine non-tree edge
    lca = LCAStructure(Q, pred, 0)
    assert lca.lca(0, 3) == 0
    res = mwc(G, root_order=[0, 1, 2, 3])
    assert approx_eq(res.length, 13.0)
    assert approx_eq(mwc_oracle(G)[0], 13.0)


def test_interior_lca_not_parent_of_either_endpoint():
    # r - p ; p - y1 - y ; p - z1 - z ; chord (y,z).
    names = ["r", "p", "y1", "y", "z1", "z"]
    G = gen.from_edges(
        names,
        [("r", "p", 1), ("p", "y1", 1), ("y1", "y", 1),
         ("p", "z1", 1), ("z1", "z", 1), ("y", "z", 7)],
    )
    delta, pred, Q = _truncated_dijkstra(G, "r", set(G), INF, _index_map(G))
    lca = LCAStructure(Q, pred, "r")
    p = lca.lca("y", "z")
    assert p == "p"
    assert p not in ("y", "z")
    assert pred["y"] != p and pred["z"] != p   # not a parent of either
    res = mwc(G, root_order=names)
    assert approx_eq(res.length, 11.0)         # 1+1+1+1+7
    assert approx_eq(mwc_oracle(G)[0], 11.0)


# ===========================================================================
# 4.  disconnected graphs, MWC in a later component
# ===========================================================================

def test_disconnected_mwc_in_later_component():
    rng = random.Random(7)
    big = gen.from_edges(6, [(i, (i + 1) % 6, 3.0) for i in range(6)])   # l = 18
    small = gen.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])   # l = 3
    tree = gen.random_tree(5, rng, "int")
    G = gen.disconnected([big, tree, small])
    g_star, _ = mwc_oracle(G)
    assert approx_eq(g_star, 3.0)
    res = mwc(G)
    assert approx_eq(res.length, 3.0)
    assert is_simple_cycle(G, res.cycle)
    # reversed order too
    res2 = mwc(G, root_order=list(reversed(list(G))))
    assert approx_eq(res2.length, 3.0)
    # transversal variant
    assert approx_eq(mwc_transversal(G).length, 3.0)


def test_acyclic_and_empty():
    rng = random.Random(1)
    for n in range(1, 10):
        T = gen.random_tree(n, rng, "int")
        assert mwc_oracle(T)[0] == INF
        r = mwc(T)
        assert r.length == INF and r.cycle is None
        assert mwc_transversal(T).length == INF
    empty = {}
    assert mwc_oracle(empty)[0] == INF
    assert mwc(empty).length == INF
    assert mwc_transversal(empty).length == INF
    iso = gen.from_edges(4, [])
    assert mwc(iso).length == INF


# ===========================================================================
# 5.  ties (equal distances, equal weights)
# ===========================================================================

def test_equal_distance_and_equal_weight_ties():
    # K4 with all-equal weights: every Dijkstra tie is live.
    for n in (4, 5, 6):
        rng = random.Random(n)
        G = gen.complete(n, rng, "unit")
        assert approx_eq(mwc(G).length, 3.0)
        assert approx_eq(mwc_oracle(G)[0], 3.0)

    # complete bipartite K_{3,3}, unit weights: girth 4, massive ties.
    edges = [(("L", i), ("R", j)) for i in range(3) for j in range(3)]
    K33 = gen.from_edges([("L", i) for i in range(3)] + [("R", j) for j in range(3)],
                         [(u, v, 1.0) for u, v in edges])
    assert approx_eq(mwc(K33).length, 4.0)
    assert approx_eq(mwc_oracle(K33)[0], 4.0)

    # Two equally-long shortest paths to the same vertex + an equal-weight
    # closing edge: cycle 0-1-3-2-0 and 0-1-4-2-0 both have weight 4.
    G2 = gen.from_edges(5, [(0, 1, 1), (0, 2, 1), (1, 3, 1), (2, 3, 1),
                            (1, 4, 1), (2, 4, 1)])
    assert approx_eq(mwc(G2).length, 4.0)
    for order in itertools.permutations(range(5)):
        assert approx_eq(mwc(G2, root_order=list(order)).length, 4.0)

    # ties under the truncation horizon: even cycle, both halves equal
    C6 = gen.from_edges(6, [(i, (i + 1) % 6, 1.0) for i in range(6)])
    assert approx_eq(mwc(C6).length, 6.0)
    assert approx_eq(mwc_oracle(C6)[0], 6.0)


# ===========================================================================
# 6.  several root-order permutations give the same exact answer
# ===========================================================================

def test_root_order_permutations_agree():
    rng = random.Random(20240607)
    for trial in range(60):
        n = rng.randint(4, 8)
        G = gen.erdos_renyi(n, rng.uniform(0.3, 0.8), rng,
                            rng.choice(["unit", "int", "cont"]))
        g_star, _ = mwc_oracle(G)
        base = list(G)
        seen = set()
        for _ in range(8):
            perm = base[:]
            rng.shuffle(perm)
            res = mwc(G, root_order=perm)
            seen.add(round(res.length, 9) if res.length < INF else INF)
            assert approx_eq(res.length, g_star), (perm, res.length, g_star)
            if res.cycle is not None:
                assert is_simple_cycle(G, res.cycle)
        assert len(seen) == 1


# ===========================================================================
# 7.  the unsafe-pruning regression (legacy `dist + 2*w_min >= gamma` rule)
# ===========================================================================

def unsafe_pruning_graph():
    """Component A: a 10-cycle (total weight 10).
    Component B: root x joined to a,b,c by weight-8 edges; abc a unit triangle.
    The true MWC is the unit triangle, gamma* = 3.
    """
    edges = [(f"A{i}", f"A{(i + 1) % 10}", 1.0) for i in range(10)]
    edges += [("x", "a", 8.0), ("x", "b", 8.0), ("x", "c", 8.0)]
    edges += [("a", "b", 1.0), ("b", "c", 1.0), ("c", "a", 1.0)]
    nodes = [f"A{i}" for i in range(10)] + ["x", "a", "b", "c"]
    return gen.from_edges(nodes, edges)


def test_unsafe_pruning_regression():
    G = unsafe_pruning_graph()
    order = [f"A{i}" for i in range(10)] + ["x", "a", "b", "c"]

    g_star, c_star = mwc_oracle(G)
    assert approx_eq(g_star, 3.0)
    assert set(c_star) == {"a", "b", "c"}

    # the repaired EXACT algorithm gets it right, in this and any order
    res = mwc(G, root_order=order)
    assert approx_eq(res.length, 3.0), res.length
    assert set(res.cycle) == {"a", "b", "c"}
    assert res.stats["total_deletions"] == 0
    rev = mwc(G, root_order=list(reversed(order)))
    assert approx_eq(rev.length, 3.0)
    assert approx_eq(mwc_transversal(G).length, 3.0)

    # ---- documentation of the LEGACY rule (NOT implemented in mwc.py) -----
    # shortest_cycle.py, line 441:
    #     if v in active_nodes and dist_v != inf and dist_v + 2*min_edge_weight >= gamma:
    #         active_nodes.discard(v)
    # After the first component has been processed gamma = 10 and w_min = 1.
    gamma_after_A = mwc(G, root_order=order, roots={"A0"}).length
    assert approx_eq(gamma_after_A, 10.0)
    w_min = min(w for u in G for w in G[u].values())
    assert approx_eq(w_min, 1.0)
    dist_x = plain_dijkstra(G, "x")
    doomed = {
        v for v in G
        if dist_x.get(v, INF) < INF and dist_x[v] + 2 * w_min >= gamma_after_A
    }
    # the legacy rule wipes out exactly the MWC ...
    assert doomed == {"a", "b", "c"}
    assert doomed == set(c_star)
    # ... while keeping x, so no later root can ever see the triangle.
    assert "x" not in doomed
    # the surviving graph has girth 10, i.e. the legacy answer is 10 != 3
    survivors = {v: {u: w for u, w in G[v].items() if u not in doomed}
                 for v in G if v not in doomed}
    assert approx_eq(mwc_oracle(survivors)[0], 10.0)


# ===========================================================================
# 8.  exact mode for several alpha <= beta pairs
# ===========================================================================

EXACT_PAIRS = [(0.0, 0.0), (0.0, 0.1), (0.1, 0.1), (0.2, 0.2), (0.05, 0.5),
               (0.3, 0.3), (0.0, 1.0), (0.49, 0.49)]


def test_exact_mode_alpha_le_beta():
    rng = random.Random(4242)
    graphs = gen.family_suite(rng, count=120)
    for name, G in graphs:
        g_star, _ = mwc_oracle(G)
        for a, b in EXACT_PAIRS:
            res = mwc(G, alpha=a, beta=b)
            assert res.mode == "exact"
            assert res.certified
            assert approx_eq(res.kappa, 1.0)
            assert res.stats["total_deletions"] == 0, (name, a, b)
            assert approx_eq(res.length, g_star), (name, a, b, res.length, g_star)
            if res.cycle is not None:
                assert is_simple_cycle(G, res.cycle)
                assert approx_eq(cycle_weight(G, res.cycle), res.length)


# ===========================================================================
# 9.  approximation property   gamma* <= ghat <= kappa*gamma* + 1e-9
# ===========================================================================

APPROX_PAIRS = [
    (0.0, 0.0), (0.05, 0.0), (0.1, 0.0), (0.2, 0.0), (0.25, 0.05),
    (0.3, 0.1), (0.4, 0.1), (0.45, 0.2), (0.49, 0.0), (0.6, 0.2),
    (0.75, 0.4), (1.0, 0.6),
]


def test_approximation_bound_property():
    rng = random.Random(20250725)
    graphs = gen.family_suite(rng, count=600)
    graphs += [("planted", gen.planted_cycle(rng.randint(8, 14), rng.randint(3, 6), rng))
               for _ in range(120)]
    graphs += [("multiscale", gen.multiscale(rng.randint(5, 10), rng))
               for _ in range(1800)]
    assert len(graphs) >= 2000
    checked = 0
    worst = 0.0
    deletions = 0
    for name, G in graphs:
        g_star, _ = mwc_oracle(G)
        order = list(G)
        shuffled = order[:]
        rng.shuffle(shuffled)
        for a, b in APPROX_PAIRS:
            k = kappa_of(a, b)
            for perm in (order, list(reversed(order)), shuffled):
                res = mwc(G, alpha=a, beta=b, root_order=perm)
                assert approx_eq(res.kappa, k)
                deletions += res.stats["total_deletions"]
                if g_star == INF:
                    assert res.length == INF
                    continue
                assert res.length < INF
                assert is_simple_cycle(G, res.cycle)
                assert approx_eq(cycle_weight(G, res.cycle), res.length)
                assert res.length >= g_star - 1e-9, (name, a, b)
                assert res.length <= k * g_star + 1e-9, (
                    name, a, b, res.length, g_star, k)
                worst = max(worst, res.length / g_star)
                checked += 1
    assert checked > 2000
    # non-vacuity: the discard rule must actually have fired somewhere
    assert deletions > 0, "approximation test degenerated to exact mode"
    print(f"\n[approx] {checked} (graph, alpha, beta, order) runs, "
          f"{deletions} vertex deletions; worst observed ratio = {worst:.6f}")


def test_discard_rule_actually_fires():
    """Non-vacuity guard: the parameterized sweep must be reachable."""
    G = gen.hub_gadget()
    order = [("A", i) for i in range(5)] + [("B", "x"), ("B", "h")]
    order += [("B", i) for i in range(5)]
    g_star, _ = mwc_oracle(G)
    assert approx_eq(g_star, 5.0)
    fired = 0
    for a, b in APPROX_PAIRS:
        res = mwc(G, alpha=a, beta=b, root_order=order)
        assert res.length >= g_star - 1e-9
        assert res.length <= kappa_of(a, b) * g_star + 1e-9
        if res.stats["total_deletions"]:
            fired += 1
            causes = [pr["deletion_cause"] for pr in res.stats["per_root"]
                      if pr["deletion_cause"] is not None]
            assert causes
            for c in causes:
                assert c["cycle"] is not None
                assert is_simple_cycle(G, c["cycle"])
                assert c["cycle_length"] > c["gamma"] - 1e-12
                assert c["threshold"] is not None
            # the hub is what gets removed
            deleted = set()
            for pr in res.stats["per_root"]:
                deleted |= set(pr["deleted"])
            assert deleted == {("B", "h")}, deleted
    assert fired >= 3, fired


def test_known_approximation_error_instance():
    """A concrete instance where the discard rule really costs accuracy.

    Found by randomized search over the multiscale family.  An MWC vertex
    (vertex 0) is deleted, so ghat = 12.1835 > gamma* = 11.3427, while
    kappa(0.4, 0.0) = 5.  Guards against the code silently degenerating to
    exact mode.
    """
    edges = [(0, 1, 4.5731), (0, 2, 0.9449), (0, 4, 4.591), (0, 5, 4.7299),
             (1, 2, 9.9534), (1, 4, 9.7135), (1, 5, 4.5372), (1, 6, 9.6639),
             (2, 5, 10.4497), (3, 4, 0.9568), (3, 5, 1.065), (4, 5, 10.1617),
             (4, 6, 4.7395)]
    G = gen.from_edges(7, edges)
    order = [1, 6, 2, 3, 4, 0, 5]
    g_star, c_star = mwc_oracle(G)
    assert approx_eq(g_star, 11.3427)
    assert set(c_star) == {0, 3, 4, 5}

    exact = mwc(G, root_order=order)
    assert approx_eq(exact.length, g_star)
    assert exact.stats["total_deletions"] == 0

    res = mwc(G, alpha=0.4, beta=0.0, root_order=order)
    assert res.mode == "approximate"
    assert not res.certified
    assert approx_eq(res.kappa, 5.0)
    assert res.stats["total_deletions"] == 1
    deleted = set()
    for pr in res.stats["per_root"]:
        deleted |= set(pr["deleted"])
    assert deleted == {0}
    assert deleted & set(c_star)                     # an MWC vertex was lost
    assert approx_eq(res.length, 12.1835)
    assert res.length > g_star                       # a genuine error
    assert res.length <= res.kappa * g_star + 1e-9   # ... within the bound
    assert is_simple_cycle(G, res.cycle)
    assert approx_eq(cycle_weight(G, res.cycle), res.length)


# ===========================================================================
# 10.  monotonicity of gamma + every finite gamma is a real cycle weight
# ===========================================================================

def test_gamma_monotone_and_witnessed():
    rng = random.Random(99)
    graphs = gen.family_suite(rng, count=120)
    for name, G in graphs:
        for a, b in [(0.0, 0.0), (0.3, 0.0), (0.5, 0.2)]:
            res = mwc(G, alpha=a, beta=b)
            trace = res.stats["gamma_trace"]
            for i in range(1, len(trace)):
                assert trace[i] <= trace[i - 1] + 1e-12, (name, a, b, trace)
            for g in trace:
                if g == INF:
                    continue
                # every finite value in the trace must be realised by a cycle
                assert g >= mwc_oracle(G)[0] - 1e-9
            if res.length < INF:
                assert is_simple_cycle(G, res.cycle)
                assert approx_eq(cycle_weight(G, res.cycle), res.length)
            else:
                assert res.cycle is None


def test_finite_gamma_always_has_a_witness_cycle():
    rng = random.Random(31337)
    for _ in range(150):
        n = rng.randint(3, 12)
        G = gen.erdos_renyi(n, rng.uniform(0.1, 0.9), rng,
                            rng.choice(["unit", "int", "cont", "wide"]))
        a, b = rng.choice(APPROX_PAIRS)
        res = mwc(G, alpha=a, beta=b)
        if res.length < INF:
            assert is_simple_cycle(G, res.cycle)
            assert approx_eq(cycle_weight(G, res.cycle), res.length)


# ===========================================================================
# 11.  transversal variant: exactness vs the oracle + the promised reductions
# ===========================================================================

def test_transversal_exactness_vs_oracle():
    rng = random.Random(555)
    graphs = gen.family_suite(rng, count=250)
    tot_roots_full = tot_roots_tv = 0
    tot_settled_full = tot_settled_tv = 0
    for name, G in graphs:
        g_star, _ = mwc_oracle(G)
        res = mwc_transversal(G)
        assert approx_eq(res.length, g_star), (name, res.length, g_star)
        assert res.mode == "exact"
        if res.length < INF:
            assert is_simple_cycle(G, res.cycle)
            assert approx_eq(cycle_weight(G, res.cycle), res.length)
        else:
            assert res.cycle is None
        # and with the structural reductions individually disabled
        for kw in ({"use_2core": False}, {"use_blocks": False},
                   {"use_2core": False, "use_blocks": False}):
            r2 = mwc_transversal(G, **kw)
            assert approx_eq(r2.length, g_star), (name, kw, r2.length)
        full = mwc(G)
        tot_roots_full += full.stats["roots_run"]
        tot_settled_full += full.stats["total_settled"]
        tot_roots_tv += res.stats["roots_run"]
        tot_settled_tv += res.stats["total_settled"]
    assert tot_roots_tv <= tot_roots_full
    print(f"\n[transversal] roots {tot_roots_tv}/{tot_roots_full} "
          f"({tot_roots_tv / max(1, tot_roots_full):.3f}), settled "
          f"{tot_settled_tv}/{tot_settled_full} "
          f"({tot_settled_tv / max(1, tot_settled_full):.3f})")


def test_transversal_size_bound():
    """|S| <= min{n, mu} with mu = m - n + c(G) on each 2-core block."""
    rng = random.Random(808)
    for _ in range(80):
        n = rng.randint(4, 14)
        G = gen.erdos_renyi(n, rng.uniform(0.15, 0.7), rng, "int")
        res = mwc_transversal(G)
        core = two_core(G)
        m = sum(len(d) for d in core.values()) // 2
        nn = len(core)
        # number of components of the 2-core
        seen, comps = set(), 0
        for s in core:
            if s in seen:
                continue
            comps += 1
            stack = [s]
            seen.add(s)
            while stack:
                u = stack.pop()
                for v in core[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
        mu = m - nn + comps
        assert res.stats["transversal_size"] <= max(0, min(nn, mu)) or nn == 0


def test_two_core_and_blocks_preserve_girth():
    rng = random.Random(1234)
    for _ in range(120):
        n = rng.randint(3, 14)
        G = gen.erdos_renyi(n, rng.uniform(0.1, 0.6), rng, "int")
        g_star, _ = mwc_oracle(G)
        core = two_core(G)
        assert approx_eq(mwc_oracle(core)[0] if core else INF, g_star)
        best = INF
        for edges in biconnected_components(G):
            blk = {}
            for u, v, w in edges:
                blk.setdefault(u, {})[v] = w
                blk.setdefault(v, {})[u] = w
            check_graph(blk)
            best = min(best, mwc_oracle(blk)[0])
        assert approx_eq(best, g_star)


# ===========================================================================
# 12.  exhaustive differential test over the graph atlas (n <= 7)
# ===========================================================================

@pytest.mark.skipif(not gen.HAVE_NX, reason="networkx required for the atlas")
def test_graph_atlas_exhaustive_unit_weights():
    n_tested = 0
    for name, g in gen.atlas_graphs(max_nodes=7):
        G = gen.from_nx(g, wf=lambda: 1.0)
        g_star, c_star = mwc_oracle(G)
        res = mwc(G)
        assert approx_eq(res.length, g_star), (name, res.length, g_star)
        assert approx_eq(mwc_transversal(G).length, g_star), name
        if res.length < INF:
            assert is_simple_cycle(G, res.cycle)
            assert approx_eq(cycle_weight(G, res.cycle), res.length)
        n_tested += 1
    assert n_tested >= 1000
    print(f"\n[atlas/unit] {n_tested} graphs")


@pytest.mark.skipif(not gen.HAVE_NX, reason="networkx required for the atlas")
def test_graph_atlas_exhaustive_random_weights():
    rng = random.Random(2718281)
    n_tested = 0
    for name, g in gen.atlas_graphs(max_nodes=7):
        for kind in ("int", "cont"):
            G = gen.from_nx(g, wf=gen.weight_fn(kind, rng))
            g_star, _ = mwc_oracle(G)
            res = mwc(G)
            assert approx_eq(res.length, g_star), (name, kind, res.length, g_star)
            rev = mwc(G, root_order=list(reversed(list(G))))
            assert approx_eq(rev.length, g_star), (name, kind)
            assert approx_eq(mwc_transversal(G).length, g_star), (name, kind)
            if res.length < INF:
                assert is_simple_cycle(G, res.cycle)
                assert approx_eq(cycle_weight(G, res.cycle), res.length)
            n_tested += 1
    assert n_tested >= 2000
    print(f"\n[atlas/weighted] {n_tested} (graph, weighting) pairs")


# ===========================================================================
# extra guards
# ===========================================================================

def test_lca_table_height_is_not_constant():
    """A deep path forces log2(|Q|) lifting levels; a constant height fails."""
    n = 300
    G = gen.from_edges(n, [(i, i + 1, 1.0) for i in range(n - 1)] + [(0, n - 1, 1.0)])
    delta, pred, Q = _truncated_dijkstra(G, 1, set(G), INF, _index_map(G))
    lca = LCAStructure(Q, pred, 1)
    assert lca.log >= (len(Q)).bit_length()
    assert lca.lca(n - 2, 2) == 2 or lca.lca(n - 2, 2) == 1
    res = mwc(G, root_order=list(range(n)))
    assert approx_eq(res.length, float(n)), res.length


def test_certifier_plumbing_fires(monkeypatch):
    """The certifier must really compare against an independent re-summation."""
    import mwc as mwc_mod

    G = gen.from_edges(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)])
    assert approx_eq(mwc(G).length, 4.0)

    real = mwc_mod.cycle_weight
    monkeypatch.setattr(mwc_mod, "cycle_weight", lambda a, c: real(a, c) + 1.0)
    with pytest.raises(CertificationError):
        mwc(G, root_order=[0, 1, 2, 3])


def test_initial_bound_is_validated():
    G = gen.from_edges(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)])
    with pytest.raises(CertificationError):
        mwc(G, gamma0=2.0, cycle0=(0, 1, 2, 3))       # real weight is 4
    with pytest.raises(ValueError):
        mwc(G, gamma0=2.0, cycle0=(0, 2))             # not a cycle at all
    ok = mwc(G, gamma0=4.0, cycle0=(0, 1, 2, 3))
    assert approx_eq(ok.length, 4.0)


def test_alpha_beta_validation():
    G = gen.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
    with pytest.raises(ValueError):
        mwc(G, alpha=1.0, beta=0.0)      # 1 - 2 + 0 = -1 <= 0
    with pytest.raises(ValueError):
        mwc(G, alpha=0.5, beta=0.0)      # 1 - 1 + 0 = 0  <= 0
    with pytest.raises(ValueError):
        mwc(G, alpha=-0.1)
    with pytest.raises(ValueError):
        mwc(G, root_order=[0, 1])        # not a permutation
    assert approx_eq(kappa_of(0.0, 0.0), 1.0)
    assert approx_eq(kappa_of(0.0, 0.5), 1.0)       # max{1, 1/2} = 1  (repair R2)
    assert approx_eq(kappa_of(0.25, 0.0), 2.0)


def test_graph_validation():
    with pytest.raises(ValueError):
        check_graph({0: {0: 1.0}})                       # self loop
    with pytest.raises(ValueError):
        check_graph({0: {1: 1.0}, 1: {0: 2.0}})          # asymmetric weight
    with pytest.raises(ValueError):
        check_graph({0: {1: 1.0}})                       # dangling
    with pytest.raises(ValueError):
        check_graph({0: {1: 0.0}, 1: {0: 0.0}})          # zero weight
    check_graph({0: {1: 0.0}, 1: {0: 0.0}}, allow_zero=True)
