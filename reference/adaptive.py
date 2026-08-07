"""Adaptive root-set selection, and the A0-A7 ablation ladder.

Motivation
----------
Table 5 of the manuscript shows the cycle-transversal reduction is *conditional*:
it wins by up to 71x when mu/n is small and loses 13-35% of the speed when
mu/n approaches 1, because the 2-core peel, the block decomposition and the
seed are paid for without buying a root reduction.  The manuscript's own
recommendation ("compute mu first --- it is O(n+m) --- and use transversal
roots only when mu/n is small") is stated but never measured.  This module
turns that recommendation into an algorithm and makes it measurable.

The observation that makes it cheap
-----------------------------------
The decision quantity mu = m - n + c(G) and the fundamental-cycle seed gamma_0
come from the SAME spanning forest (`_transversal_for`).  So a single O(n+m)
pass yields all three of: the switch statistic mu, a valid initial bound
gamma_0, and the transversal S itself.  The seed is therefore free on BOTH
branches -- including the all-roots branch, which in the manuscript's
Table 5 configuration ("allroots") runs unseeded.  The adaptive algorithm is
consequently never worse than an unseeded all-roots run by more than the one
forest pass, and is strictly better whenever the seed activates the
radius-gamma/2 truncation earlier.

Exactness
---------
Every variant here is exact.  A0-A7 all run with alpha = beta = 0, so the
discard rule provably never fires (Theorem, No discarding when alpha <= beta).
The only variation is *which* accelerations are switched on.  A0 additionally
sets radius_factor = inf, disabling truncation -- still exact, just slower.
Root sets are always cycle transversals or all of V, so exactness follows from
Theorem (Exactness with cycle-transversal roots) in every case.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mwc import (
    INF,
    MWCResult,
    _subgraph_from_edges,
    _transversal_for,
    biconnected_components,
    mwc,
    mwc_transversal,
    spanning_forest,
    two_core,
)

# Fitted from `campaign.py theta` (job 18104376_2, Xeon Gold 6248, n = 800 and
# 1600, 3 instances per point).  Median transversal-vs-allroots speedup stays
# above 1 through mu/n = 0.7 (1.11x at n=1600) and crosses below 1 at
# mu/n = 0.8-1.0, so the switch is set at 0.8 rather than at the midpoint 0.5
# guessed before the sweep -- 0.5 gave away the 1.1-1.3x still available in
# 0.5 <= mu/n <= 0.7.
THETA_DEFAULT = 0.8


def cyclomatic_number(adj: Dict[Any, Dict[Any, float]]) -> int:
    """mu(G) = m - n + c(G), in O(n+m), as the count of non-tree edges."""
    _, _, non_tree, _ = spanning_forest(adj)
    return len(non_tree)


def greedy_vc_transversal(adj: Dict[Any, Dict[Any, float]]) -> Tuple[set, float, Optional[Tuple]]:
    """Greedy max-degree vertex cover of the non-tree edges (variant A6).

    Any vertex cover of E \\ E(T_0) is a cycle transversal, since every simple
    cycle contains a non-tree edge.  Measured at 26% fewer roots than the
    one-endpoint rule in the manuscript's transversal study.  Returns the same
    triple shape as `_transversal_for` minus mu.
    """
    parent, roots, non_tree, order = spanning_forest(adj)
    if not non_tree:
        return set(), INF, None
    # gamma_0 and the realising cycle come from the shared forest pass.
    _, gamma0, cyc0, _ = _transversal_for(adj)

    # O(m + mu log mu): incident-edge lists plus a lazy max-heap on degree.
    # A naive rescan of the surviving edge set per pick is O(mu^2) and, measured
    # at n=1600, made this variant 10x SLOWER than the one-endpoint rule on the
    # grid family -- i.e. it would have benchmarked the implementation rather
    # than the cover.
    edges = [(u, v) if str(u) <= str(v) else (v, u) for u, v, _ in non_tree]
    incident: Dict[Any, List[int]] = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        incident[u].append(i)
        incident[v].append(i)
    alive = [True] * len(edges)
    deg = {x: len(ix) for x, ix in incident.items()}
    heap = [(-d, str(x), x) for x, d in deg.items()]
    heapq.heapify(heap)

    S: set = set()
    covered = 0
    while covered < len(edges) and heap:
        negd, _, x = heapq.heappop(heap)
        if -negd != deg.get(x, 0):        # stale entry
            continue
        if deg[x] == 0:
            continue
        S.add(x)
        for i in incident[x]:
            if not alive[i]:
                continue
            alive[i] = False
            covered += 1
            u, v = edges[i]
            for y in (u, v):
                deg[y] -= 1
                if y != x:
                    heapq.heappush(heap, (-deg[y], str(y), y))
        deg[x] = 0
    return S, gamma0, cyc0


def _run_on_blocks(adj, *, root_fn, gamma0, cycle0, use_2core, use_blocks,
                   certify, collect_stats, radius_factor=0.5):
    """Shared driver: optional 2-core peel, optional block split, then one
    root search pass per block, threading the global gamma through as the
    truncation bound (Proposition, Core and block reduction)."""
    work = two_core(adj) if use_2core else dict(adj)
    if use_blocks:
        blocks = [_subgraph_from_edges(e) for e in biconnected_components(work)]
        blocks = [b for b in blocks if sum(len(d) for d in b.values()) // 2 >= len(b)]
    else:
        blocks = [work] if work else []

    gamma = float(gamma0)
    best = cycle0
    agg = {"roots_run": 0, "total_settled": 0, "blocks": len(blocks),
           "n_2core": len(work), "m_2core": sum(len(d) for d in work.values()) // 2}
    for b in blocks:
        if len(b) < 3:
            continue
        roots = root_fn(b) if root_fn is not None else None
        if roots is not None and not roots:
            continue
        # gamma is a bound certified on another block / the whole graph, so the
        # realising cycle need not live in b: pass it as an external bound
        # (cycle0=None), exactly as mwc_transversal does across blocks.
        res = mwc(b, roots=roots, gamma0=gamma, cycle0=None,
                  certify=certify, collect_stats=collect_stats,
                  radius_factor=radius_factor,
                  roots_are_transversal=roots is not None)
        if res.length < gamma:
            gamma, best = res.length, res.cycle
        agg["roots_run"] += res.stats.get("roots_run", 0)
        agg["total_settled"] += res.stats.get("total_settled", 0)
    return gamma, best, agg


def mwc_adaptive(
    adj: Dict[Any, Dict[Any, float]],
    *,
    theta: float = THETA_DEFAULT,
    seed_allroots: bool = False,
    certify: bool = True,
    collect_stats: bool = True,
) -> MWCResult:
    """Exact MWC with the root set chosen by the measured mu/n statistic.

    The decision must cost NOTHING, or the switch eats the speedup it exists
    to protect.  Two earlier versions failed this test on the theta sweep: one
    took mu from `_transversal_for` (which also builds the forest distance
    structure and evaluates gamma_0 over every non-tree edge) and gave away
    44% of the available speedup at mu/n = 0.02 -- 8.2x instead of 14.5x; a
    second took mu from a bare spanning forest and still gave away 36%.  The
    reason is that when mu/n is small the whole run is O(n+m)-dominated, so
    ANY extra O(n+m) pass is a constant fraction of the total.

    The fix uses the cyclomatic identity rather than a traversal.  Since
    mu = m - n + c(G) and c(G) >= 1,

        mu >= mu_lb := m - n + 1,

    and mu_lb is available in O(1) from the sizes alone.  If mu_lb > theta*n
    the graph is certainly above the switch point and the all-roots branch is
    taken without ever walking the graph; otherwise the transversal branch is
    taken, and it computes the forest it needs anyway.  So the decision adds
    no traversal on either side.  On disconnected inputs the bound is
    conservative (c(G) > 1 makes the true mu larger), which can route a graph
    to the transversal branch when the exact statistic would not have; that
    branch is exact regardless, so the cost is at most the transversal
    overhead on a graph near the boundary, never a wrong answer.

    `seed_allroots` controls whether the dense branch also pays for gamma_0.
    It defaults to FALSE on measured evidence: obtaining gamma_0 means running
    `_transversal_for`, which builds the forest distance structure and
    evaluates a tree distance for EVERY non-tree edge -- work proportional to
    mu, i.e. largest exactly on the branch that needs it least.  At n = 1600
    seeding the dense branch cost 15% on the grid family (0.0267s vs 0.0233s)
    and 7% on small world, against a best case of 2% on dense ER.  With it off,
    the dense branch is byte-for-byte the published all-roots configuration, so
    the switch is exactly "pick the better of the two configurations of
    Table 5, decided in O(1)" and adds nothing to either branch.
    """
    n = len(adj)
    m = sum(len(d) for d in adj.values()) // 2
    mu_lb = m - n + 1                              # O(1); exact when connected
    took_transversal = mu_lb <= theta * n

    # NB: do NOT short-circuit on m < n as "must be a forest".  That inference
    # needs connectivity: a triangle plus isolated vertices has m < n and girth
    # 3.  Acyclicity is mu = 0, not m < n, and mu is what we are deliberately
    # not computing.  m < n gives mu_lb <= 0 <= theta*n, so such graphs take the
    # transversal branch, which detects a forest correctly on its own.

    if took_transversal:
        # use_blocks=False on measured evidence.  The block reduction is
        # sound (Proposition, Core and block reduction) but was a pessimization
        # on all ten graphs probed -- five synthetic families and five real
        # road/power networks -- because the 2-core is dominated by ONE giant
        # biconnected component in every case (largest block = 91-100% of the
        # 2-core), so Hopcroft-Tarjan is paid for and splits nothing.  Cost
        # ranged from 10% (grid) to 34% (sydney-road).  The 2-core peel is
        # kept: it is cheap and removes 92% of the near-tree family.
        res = mwc_transversal(adj, use_blocks=False, certify=certify,
                              collect_stats=collect_stats)
        stats = dict(res.stats)
        stats.update(branch="transversal")
        length, cycle = res.length, res.cycle
    elif seed_allroots:
        _, gamma0, cyc0, _ = _transversal_for(adj)
        res = mwc(adj, gamma0=gamma0, cycle0=cyc0, certify=certify,
                  collect_stats=collect_stats)
        stats = dict(res.stats)
        stats.update(branch="allroots_seeded", gamma0=gamma0)
        length, cycle = res.length, res.cycle
    else:
        res = mwc(adj, certify=certify, collect_stats=collect_stats)
        stats = dict(res.stats)
        stats.update(branch="allroots_unseeded")
        length, cycle = res.length, res.cycle

    stats.update(n=n, m=m, mu_lb=mu_lb, mu_lb_over_n=mu_lb / n if n else 0.0,
                 theta=theta)
    return MWCResult(length=length, cycle=cycle, certified=certify, mode="exact",
                     kappa=1.0, stats=stats)


# --------------------------------------------------------------------------
# The ablation ladder.  Each rung adds exactly one acceleration to the one
# above it, so the delta between consecutive rows is attributable to that
# single mechanism.  A1 and A5 reproduce Table 5's "allroots" and
# "transversal" columns respectively.
# --------------------------------------------------------------------------
ABLATION_LABELS = {
    "A0": "all roots, no truncation (radius = inf), no seed",
    "A1": "+ radius-gamma/2 truncation                      [Table 5 allroots]",
    "A2": "+ fundamental-cycle seed gamma_0",
    "A3": "+ 2-core peeling",
    "A4": "+ biconnected block decomposition",
    "A5": "+ transversal roots (one endpoint per non-tree edge) [Table 5 transversal]",
    "A6": "+ greedy vertex-cover transversal (smaller S)",
    "A7": "+ adaptive mu/n switch                            [this work]",
}


def run_ablation(adj, variant: str, *, theta: float = THETA_DEFAULT,
                 certify: bool = False, collect_stats: bool = True):
    """Run one rung of the ladder.  Returns (length, cycle, stats)."""
    if variant == "A0":
        r = mwc(adj, certify=certify, collect_stats=collect_stats,
                radius_factor=math.inf)
        return r.length, r.cycle, dict(r.stats)
    if variant == "A1":
        r = mwc(adj, certify=certify, collect_stats=collect_stats)
        return r.length, r.cycle, dict(r.stats)
    if variant == "A2":
        _, g0, c0, _ = _transversal_for(adj)
        r = mwc(adj, gamma0=g0, cycle0=c0, certify=certify,
                collect_stats=collect_stats)
        return r.length, r.cycle, dict(r.stats)
    if variant == "A3":
        _, g0, c0, _ = _transversal_for(adj)
        g, c, agg = _run_on_blocks(adj, root_fn=None, gamma0=g0, cycle0=c0,
                                   use_2core=True, use_blocks=False,
                                   certify=certify, collect_stats=collect_stats)
        return g, c, agg
    if variant == "A4":
        _, g0, c0, _ = _transversal_for(adj)
        g, c, agg = _run_on_blocks(adj, root_fn=None, gamma0=g0, cycle0=c0,
                                   use_2core=True, use_blocks=True,
                                   certify=certify, collect_stats=collect_stats)
        return g, c, agg
    if variant == "A5":
        r = mwc_transversal(adj, certify=certify, collect_stats=collect_stats)
        return r.length, r.cycle, dict(r.stats)
    if variant == "A6":
        _, g0, c0, _ = _transversal_for(adj)
        g, c, agg = _run_on_blocks(
            adj, root_fn=lambda b: greedy_vc_transversal(b)[0],
            gamma0=g0, cycle0=c0, use_2core=True, use_blocks=True,
            certify=certify, collect_stats=collect_stats)
        return g, c, agg
    if variant == "A7":
        r = mwc_adaptive(adj, theta=theta, certify=certify,
                         collect_stats=collect_stats)
        return r.length, r.cycle, dict(r.stats)
    raise ValueError(f"unknown ablation variant {variant!r}")
