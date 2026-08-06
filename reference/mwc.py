"""Canonical reference implementation of the composite-distance MWC algorithm.

This module is the artifact the revised manuscript points to.  It contains

  * ``mwc_oracle``      -- an exact, independent minimum-weight-cycle oracle
                           based on the edge-removal formula
                           ``gamma* = min_{e=(u,v)} [ w(e) + d_{G-e}(u,v) ]``.
  * ``mwc``             -- Algorithm 1 of the manuscript, implemented
                           *literally*, with the minimal set of repairs that
                           make it implementable and sound (documented below).
  * ``mwc_transversal`` -- the cycle-transversal root reduction (repair R4),
                           exact mode only.

The core is dependency-free: it operates on a plain adjacency dictionary

    adj = {u: {v: w, ...}, ...}

which must be symmetric (``adj[u][v] == adj[v][u]``), loop-free
(``u not in adj[u]``) and positively weighted (``w > 0``; ``w >= 0`` is
tolerated but flagged, see :func:`check_graph`).

--------------------------------------------------------------------------
Where Algorithm 1 *as literally written* is not implementable / not sound
--------------------------------------------------------------------------
Every item below is a manuscript bug; the code implements the repaired
version and the docstring records the deviation.

A1. ``for all x in V_active`` iterates over a set that the loop body mutates.
    Repaired: a FIXED root order (a list), with roots skipped when they are
    no longer in ``V_active`` at the time their turn comes.

A2. The inner Dijkstra relaxes only edges to ``z in V_active`` but the
    ``argmin`` in line 10 ranges over all ``v notin Q``.  Repaired: the whole
    search runs on the active induced subgraph ``H = G[V_active]``; the root
    itself must be active.  (Without this, ``delta`` mixes ``d_G`` and
    ``d_H`` and the approximation analysis -- which needs ``d_H`` -- breaks.)

A3. The truncation test ``delta(v) < gamma/2`` uses the *current* gamma,
    which the same loop shrinks online (line 21).  Shrinking the horizon
    online invalidates the outer-loop invariant proof, which compares
    ``2 d(x,v)`` against the *final* gamma.  Repaired: two-phase design.
    Phase 1 runs the truncated Dijkstra with the FIXED radius
    ``gamma_in / 2``, where ``gamma_in`` is the value of gamma when the root
    search starts.  Phase 2 does all cycle detection.  ``gamma`` therefore
    never changes during phase 1.

A4. Cycle detection in lines 12--24 is interleaved with relaxation and only
    tests ``z != pred(y)``.  Because the scan happens while ``y`` is being
    settled, ``pred(z)`` cannot equal ``y`` -- but that is an accident of the
    online formulation, and the manuscript never says so.  Repaired: after
    phase 1 we scan EVERY edge of ``H[Q]`` once and skip an edge only when it
    is a predecessor-tree edge in either direction
    (``pred[u] == v or pred[v] == u``).  This is strictly more thorough than
    the online scan and is what Lemma "R1 case 2" actually needs.

A5. ``if p`` / ``if z != pred(y)`` style truthiness tests are wrong for the
    vertex labelled ``0`` and for a ``None`` predecessor.  (The shipped code
    in ``shortest_cycle.py`` literally writes ``if p and p != u and p != v``,
    which silently drops every fundamental cycle whose LCA is vertex 0 *and*
    every ancestor-descendant non-tree edge.)  Repaired: all predecessor
    tests use ``is not None``; LCA equal to an endpoint is a legal case.

A6. The manuscript treats ``LCA_{pred*}`` as an oracle.  Repaired: an
    explicit static binary-lifting table over the predecessor tree restricted
    to ``Q``, with table height ``max(1, |Q|.bit_length())`` -- *not* a
    constant.  (A constant height silently returns a wrong ancestor on deep
    trees.)

A7. Line 21 sets ``gamma <- min(gamma, l_c)`` from an arithmetic expression
    only.  Nothing checks that the expression corresponds to a real simple
    cycle.  Repaired: every candidate is reconstructed, validated (distinct
    vertices, >= 3 of them, every consecutive pair an edge of ``G``) and its
    weight is recomputed by independent summation before ``gamma`` may move.
    A mismatch raises :class:`CertificationError`.

A8. ``d(x,c) = delta(p)`` (eq. 8) is only the distance to the *detected*
    fundamental cycle, and only inside ``H``.  Kept as written, but the
    returned ``d_to_cycle`` is documented as ``d_H``.

A9. ``gamma = infinity`` must be handled: the horizon ``gamma/2`` is then
    ``+inf`` (full Dijkstra), and the discard trigger ``l_best > gamma``
    is False, so no discarding can happen on the first root.  Kept, but the
    code is explicit about it.

A10. The discard sweep says "``z in V_active \\ {x}`` with ``z`` not yet
    processed".  "Processed" must mean "already used as an outer-loop root";
    ``x`` itself has just been processed, so the ``\\ {x}`` is redundant but
    harmless.  Implemented as written.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

INF = float("inf")

__all__ = [
    "INF",
    "CertificationError",
    "MWCResult",
    "check_graph",
    "mwc",
    "mwc_oracle",
    "mwc_transversal",
    "cycle_weight",
    "is_simple_cycle",
    "kappa_of",
    "two_core",
    "biconnected_components",
    "spanning_forest",
]


class CertificationError(AssertionError):
    """Raised when a candidate cycle fails independent re-validation."""


# ---------------------------------------------------------------------------
# graph utilities
# ---------------------------------------------------------------------------


def check_graph(adj: Dict[Any, Dict[Any, float]], *, allow_zero: bool = False) -> None:
    """Validate that *adj* is a simple, symmetric, positively weighted graph."""
    for u, nbrs in adj.items():
        if u in nbrs:
            raise ValueError(f"self-loop at {u!r}")
        for v, w in nbrs.items():
            if v not in adj:
                raise ValueError(f"edge {u!r}-{v!r} points at unknown vertex")
            if u not in adj[v]:
                raise ValueError(f"asymmetric adjacency: {u!r}-{v!r}")
            if adj[v][u] != w:
                raise ValueError(f"asymmetric weight on {u!r}-{v!r}")
            if w < 0:
                raise ValueError(f"negative weight on {u!r}-{v!r}")
            if w == 0 and not allow_zero:
                raise ValueError(f"zero weight on {u!r}-{v!r} (pass allow_zero=True)")


def edges_of(adj: Dict[Any, Dict[Any, float]], index: Dict[Any, int]):
    """Yield each undirected edge exactly once as ``(u, v, w)``."""
    for u, nbrs in adj.items():
        iu = index[u]
        for v, w in nbrs.items():
            if iu < index[v]:
                yield u, v, w


def _index_map(adj: Dict[Any, Dict[Any, float]]) -> Dict[Any, int]:
    return {v: i for i, v in enumerate(adj)}


def is_simple_cycle(adj: Dict[Any, Dict[Any, float]], cycle: Sequence[Any]) -> bool:
    """True iff *cycle* is a closed simple cycle of *adj* (>= 3 vertices)."""
    if cycle is None:
        return False
    k = len(cycle)
    if k < 3:
        return False
    if len(set(cycle)) != k:
        return False
    for i in range(k):
        u, v = cycle[i], cycle[(i + 1) % k]
        if v not in adj.get(u, {}):
            return False
    return True


def cycle_weight(adj: Dict[Any, Dict[Any, float]], cycle: Sequence[Any]) -> float:
    """Independent re-summation of a cycle's weight."""
    k = len(cycle)
    return math.fsum(adj[cycle[i]][cycle[(i + 1) % k]] for i in range(k))


def kappa_of(alpha: float, beta: float) -> float:
    """Worst-case approximation ratio of the (alpha, beta) discard rule.

        kappa(alpha, beta) = max{1, 1 / (1 - 2*min(alpha, 1/2) + 2*beta)}

    The min(alpha, 1/2) is not cosmetic.  The record-setting LCA must itself have been
    settled, so every unsettled vertex is at distance >= d_to_cycle, not merely
    >= Gamma/2.  That yields the extra inequality 2*beta*Gamma <= gamma*, which caps the
    trigger parameter at 1/2.  The unclipped 1/(1 - 2*alpha + 2*beta) is a valid but
    loose bound: at (alpha, beta) = (0.7, 0.25) it gives 10 where the truth is 2.

    Admissibility is alpha < beta + 1/2; at the boundary the error is unbounded.
    """
    if alpha >= beta + 0.5:
        raise ValueError("require alpha < beta + 1/2")
    denom = 1.0 - 2.0 * min(alpha, 0.5) + 2.0 * beta
    if denom <= 0.0:                       # unreachable given the guard above
        return math.inf
    return max(1.0, 1.0 / denom)


def _close(a: float, b: float, tol: float) -> bool:
    if a == b:
        return True
    if math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ---------------------------------------------------------------------------
# exact oracle
# ---------------------------------------------------------------------------


def _dijkstra_avoid_edge(
    adj, src, index, banned_u=None, banned_v=None, allowed=None
):
    """Plain Dijkstra from *src*, optionally forbidding one undirected edge."""
    dist = {src: 0.0}
    pred: Dict[Any, Any] = {}
    done = set()
    heap = [(0.0, index[src], src)]
    while heap:
        d, _, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v, w in adj[u].items():
            if allowed is not None and v not in allowed:
                continue
            if banned_u is not None and (
                (u == banned_u and v == banned_v) or (u == banned_v and v == banned_u)
            ):
                continue
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, index[v], v))
    return dist, pred


def mwc_oracle(adj: Dict[Any, Dict[Any, float]]) -> Tuple[float, Optional[Tuple]]:
    """Exact minimum weight cycle by the edge-removal formula.

    ``gamma* = min over e=(u,v) of [ w(e) + d_{G-e}(u,v) ]``.

    For a simple, undirected, positively weighted graph the shortest ``u--v``
    path in ``G - e`` is simple and avoids ``e``, so appending ``e`` yields a
    simple cycle of length >= 3; conversely for any cycle ``C`` and any
    ``e in C`` the formula returns at most ``l(C)``.  Hence the minimum is the
    girth.  Works component-wise for free (unreachable => ``inf``).

    Returns ``(inf, None)`` for acyclic graphs (forests) and for the empty
    graph.
    """
    check_graph(adj, allow_zero=True)
    index = _index_map(adj)
    best = INF
    best_cycle: Optional[Tuple] = None
    for u, v, w in edges_of(adj, index):
        dist, pred = _dijkstra_avoid_edge(adj, u, index, banned_u=u, banned_v=v)
        dv = dist.get(v, INF)
        if dv == INF:
            continue
        cand = w + dv
        if cand < best:
            # reconstruct the u..v path in G - e
            path = [v]
            cur = v
            while cur != u:
                cur = pred[cur]
                path.append(cur)
            path.reverse()  # u ... v
            cyc = tuple(path)
            if not is_simple_cycle(adj, cyc):
                raise CertificationError(f"oracle produced a non-simple cycle {cyc!r}")
            rec = cycle_weight(adj, cyc)
            if not _close(rec, cand, 1e-9):
                raise CertificationError(
                    f"oracle weight mismatch: formula {cand!r} vs recomputed {rec!r}"
                )
            best = rec
            best_cycle = cyc
    return best, best_cycle


# ---------------------------------------------------------------------------
# static LCA (binary lifting) over the predecessor tree restricted to Q
# ---------------------------------------------------------------------------


class LCAStructure:
    """Binary-lifting LCA over a rooted forest given as a ``pred`` dict.

    Repair A6: the table height is ``max(1, |Q|.bit_length())`` -- derived
    from the tree size, never a constant.
    """

    __slots__ = ("pos", "nodes", "depth", "up", "log", "queries")

    def __init__(self, nodes: Sequence[Any], pred: Dict[Any, Any], root: Any):
        self.nodes = list(nodes)
        self.pos = {v: i for i, v in enumerate(self.nodes)}
        n = len(self.nodes)
        self.queries = 0
        self.log = max(1, n.bit_length())
        par = [0] * n
        for v, i in self.pos.items():
            if v == root:
                par[i] = i
                continue
            p = pred.get(v)
            if p is None:
                # v is a root of its own tree fragment (should not happen for
                # the settled set Q of a connected search, but keep it total)
                par[i] = i
            else:
                if p not in self.pos:
                    raise CertificationError(
                        f"predecessor {p!r} of {v!r} is outside the settled set"
                    )
                par[i] = self.pos[p]
        # iterative depth computation
        depth = [-1] * n
        for i in range(n):
            if depth[i] >= 0:
                continue
            stack = []
            j = i
            while depth[j] < 0 and par[j] != j:
                stack.append(j)
                j = par[j]
            base = depth[j] if depth[j] >= 0 else 0
            depth[j] = base
            while stack:
                j2 = stack.pop()
                base += 1
                depth[j2] = base
        self.depth = depth
        up = [par]
        for k in range(1, self.log):
            prev = up[k - 1]
            up.append([prev[prev[i]] for i in range(n)])
        self.up = up

    def lca(self, u: Any, v: Any) -> Any:
        self.queries += 1
        iu, iv = self.pos[u], self.pos[v]
        du, dv = self.depth[iu], self.depth[iv]
        if du < dv:
            iu, iv = iv, iu
            du, dv = dv, du
        diff = du - dv
        k = 0
        while diff:
            if diff & 1:
                iu = self.up[k][iu]
            diff >>= 1
            k += 1
        if iu == iv:
            return self.nodes[iu]
        for k in range(self.log - 1, -1, -1):
            if self.up[k][iu] != self.up[k][iv]:
                iu = self.up[k][iu]
                iv = self.up[k][iv]
        return self.nodes[self.up[0][iu]]


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MWCResult:
    length: float
    cycle: Optional[Tuple]
    certified: bool
    mode: str            # 'exact' | 'approximate'
    kappa: float
    stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in ("exact", "approximate"):
            raise ValueError("mode must be 'exact' or 'approximate'")


def _truncated_dijkstra(adj, x, active, radius, index):
    """Phase 1: Dijkstra on ``H = G[active]`` truncated at ``radius``.

    Faithful to line 9 of Algorithm 1: the while loop continues only while
    some ``v notin Q`` has ``delta(v) < radius`` (STRICT ``<``).  ``radius``
    is fixed for the whole search (repair A3).

    Returns ``(delta, pred, Q_list)`` where ``delta`` also holds the tentative
    labels of the unsettled frontier (Algorithm 1's discard sweep reads
    those).
    """
    delta = {x: 0.0}
    pred: Dict[Any, Any] = {}
    settled = set()
    order: List[Any] = []
    heap = [(0.0, index[x], x)]
    while heap:
        d, _, y = heapq.heappop(heap)
        if y in settled:
            continue
        if not (d < radius):          # line 9 fails -> stop the while loop
            break
        settled.add(y)
        order.append(y)
        delta[y] = d
        for z, w in adj[y].items():
            if z not in active:
                continue
            if z in settled:
                continue
            nd = d + w
            if nd < delta.get(z, INF):
                delta[z] = nd
                pred[z] = y
                heapq.heappush(heap, (nd, index[z], z))
    return delta, pred, order


def _fundamental_cycle(y, z, p, pred):
    """Vertex sequence of the fundamental cycle closed by the edge (y, z)."""
    up_y = [y]
    cur = y
    while cur != p:
        cur = pred[cur]
        up_y.append(cur)
    up_z = [z]
    cur = z
    while cur != p:
        cur = pred[cur]
        up_z.append(cur)
    # up_y = y..p, up_z = z..p ; cycle = p..y , z..(child of p)
    up_y.reverse()                     # p .. y
    return up_y + up_z[:-1]            # p..y then z..child(p)


def mwc(
    adj: Dict[Any, Dict[Any, float]],
    *,
    alpha: float = 0.0,
    beta: float = 0.0,
    root_order: Optional[Sequence[Any]] = None,
    roots: Optional[Iterable[Any]] = None,
    certify: bool = True,
    collect_stats: bool = True,
    gamma0: float = INF,
    cycle0: Optional[Sequence[Any]] = None,
    roots_are_transversal: bool = False,
    allow_zero_weights: bool = False,
    radius_factor: float = 0.5,
    tol: float = 1e-9,
) -> MWCResult:
    """Algorithm 1 of the manuscript, repaired (see module docstring).

    Parameters
    ----------
    alpha, beta
        Discard-rule parameters, ``alpha, beta >= 0`` and
        ``1 - 2*alpha + 2*beta > 0``.  ``alpha <= beta`` is exact: the code
        *asserts* that no vertex is ever deleted in that regime.
    radius_factor
        Phase-1 truncation radius as a multiple of ``gamma_in``.  The default
        ``0.5`` is Algorithm 1 as analysed; it is the ONLY value for which
        exactness is proven.  Proposition (Sharpness)(i) shows every factor
        ``< 1/2`` can miss the MWC, and any factor ``> 1/2`` is still exact but
        settles a superset.  Exposed solely so the ablation study can measure
        what the truncation buys (``radius_factor=inf`` disables it); leave it
        alone in production.
    root_order
        Fixed outer-loop order; must be a permutation of ``V``.  Defaults to
        insertion order of ``adj``.
    roots
        Optional restriction of the roots actually run (subset of ``V``).
        The order is still taken from ``root_order``.
    certify
        Reconstruct + validate + independently re-weight every candidate
        fundamental cycle before it is allowed to move ``gamma``.
    gamma0, cycle0
        Optional valid initial upper bound and the cycle realising it.
    """
    check_graph(adj, allow_zero=allow_zero_weights)
    if alpha < 0 or beta < 0:
        raise ValueError("alpha, beta must be >= 0")
    kappa = kappa_of(alpha, beta)
    mode = "exact" if alpha <= beta else "approximate"

    index = _index_map(adj)
    nodes = list(adj)

    if root_order is None:
        order = list(nodes)
    else:
        order = list(root_order)
        if len(order) != len(nodes) or set(order) != set(nodes):
            raise ValueError("root_order must be a permutation of V")
    if roots is not None:
        root_set = set(roots)
        if not root_set <= set(nodes):
            raise ValueError("roots must be a subset of V")
        order = [x for x in order if x in root_set]

    external_bound = False
    if gamma0 < INF:
        if cycle0 is None:
            # gamma0 is a valid cycle length certified elsewhere (e.g. another
            # biconnected block); no local witness is required.
            external_bound = True
        else:
            if not is_simple_cycle(adj, cycle0):
                raise ValueError("cycle0 is not a simple cycle of adj")
            rec0 = cycle_weight(adj, cycle0)
            if not _close(rec0, gamma0, tol):
                raise CertificationError("cycle0 weight does not match gamma0")

    active = set(nodes)
    processed = set()
    gamma = float(gamma0)
    best_cycle: Optional[Tuple] = tuple(cycle0) if cycle0 is not None else None

    per_root: List[Dict[str, Any]] = []
    gamma_trace: List[float] = [gamma]
    total_deletions = 0
    total_settled = 0
    total_scans = 0
    total_lca = 0
    roots_skipped = 0

    for x in order:
        if x not in active:                       # repair A1
            roots_skipped += 1
            continue
        processed.add(x)                          # x counts as processed now
        gamma_in = gamma
        radius = INF if gamma_in == INF else radius_factor * gamma_in

        # ---- phase 1: truncated Dijkstra on H = G[active] ------------------
        delta, pred, Q = _truncated_dijkstra(adj, x, active, radius, index)
        Qset = set(Q)

        # ---- phase 2: static LCA + full scan of H[Q] ------------------------
        lca = LCAStructure(Q, pred, x)
        d_plus_min = INF
        d_to_cycle = INF
        l_best = INF
        best_local_cycle: Optional[Tuple] = None
        scans = 0

        for y in Q:
            iy = index[y]
            for z, w in adj[y].items():
                if z not in Qset:
                    continue
                if index[z] < iy:                 # each edge once
                    continue
                py = pred.get(y)
                pz = pred.get(z)
                if (py is not None and py == z) or (pz is not None and pz == y):
                    continue                       # predecessor-tree edge
                scans += 1
                p = lca.lca(y, z)
                l_c = delta[y] + delta[z] + w - 2.0 * delta[p]
                d_xc = delta[p]                    # eq. (8), inside H
                d_plus = d_xc + l_c

                cyc = None
                if certify:
                    cyc = tuple(_fundamental_cycle(y, z, p, pred))
                    if not is_simple_cycle(adj, cyc):
                        raise CertificationError(
                            f"reconstructed fundamental cycle is not simple: "
                            f"root={x!r} edge=({y!r},{z!r}) lca={p!r} cyc={cyc!r}"
                        )
                    rec = cycle_weight(adj, cyc)
                    if not _close(rec, l_c, tol):
                        raise CertificationError(
                            f"cycle-length formula mismatch: root={x!r} "
                            f"edge=({y!r},{z!r}) lca={p!r} formula={l_c!r} "
                            f"recomputed={rec!r} cyc={cyc!r}"
                        )
                    l_c = rec                      # use the certified value

                if l_c < gamma:                    # line 21
                    gamma = l_c
                    if cyc is None:
                        cyc = tuple(_fundamental_cycle(y, z, p, pred))
                    best_cycle = cyc
                if d_plus < d_plus_min:            # lines 22-23
                    d_plus_min = d_plus
                    d_to_cycle = d_xc
                    l_best = l_c
                    best_local_cycle = cyc

        # ---- discard sweep (lines 25-28) -----------------------------------
        deletions: List[Any] = []
        threshold = None
        triggered = (
            d_plus_min < INF and l_best > gamma and d_plus_min < (1.0 + alpha) * gamma
        )
        if triggered:
            threshold = d_to_cycle - beta * gamma
            for z in list(active):
                if z == x or z in processed:       # "not yet processed"
                    continue
                if delta.get(z, INF) <= threshold:
                    active.discard(z)
                    deletions.append(z)
        if alpha <= beta and deletions:
            raise AssertionError(
                "alpha <= beta must never delete a vertex; deleted "
                f"{deletions!r} from root {x!r} (threshold={threshold!r})"
            )

        total_deletions += len(deletions)
        total_settled += len(Q)
        total_scans += scans
        total_lca += lca.queries
        gamma_trace.append(gamma)
        if collect_stats:
            per_root.append(
                {
                    "root": x,
                    "gamma_in": gamma_in,
                    "gamma_out": gamma,
                    "radius": radius,
                    "Q_size": len(Q),
                    "edge_scans": scans,
                    "lca_queries": lca.queries,
                    "deletions": len(deletions),
                    "deleted": list(deletions),
                    "deletion_cause": (
                        {
                            "root": x,
                            "cycle": best_local_cycle,
                            "cycle_length": l_best,
                            "d_to_cycle": d_to_cycle,
                            "d_plus_min": d_plus_min,
                            "gamma": gamma,
                            "threshold": threshold,
                        }
                        if deletions
                        else None
                    ),
                }
            )

    if best_cycle is not None:
        if not is_simple_cycle(adj, best_cycle):
            raise CertificationError(f"final cycle is not simple: {best_cycle!r}")
        rec = cycle_weight(adj, best_cycle)
        if not _close(rec, gamma, tol):
            raise CertificationError(
                f"final gamma {gamma!r} != recomputed cycle weight {rec!r}"
            )
    elif gamma < INF and not external_bound:
        raise CertificationError("finite gamma without a witness cycle")

    full_coverage = roots is None
    certified = bool(certify and mode == "exact" and (full_coverage or roots_are_transversal))

    stats: Dict[str, Any] = {}
    if collect_stats:
        stats = {
            "n": len(nodes),
            "m": sum(len(d) for d in adj.values()) // 2,
            "roots_run": len(per_root),
            "roots_skipped": roots_skipped,
            "roots_offered": len(order),
            "total_settled": total_settled,
            "total_edge_scans": total_scans,
            "total_lca_queries": total_lca,
            "total_deletions": total_deletions,
            "gamma_trace": gamma_trace,
            "per_root": per_root,
            "alpha": alpha,
            "beta": beta,
        }

    return MWCResult(
        length=gamma,
        cycle=best_cycle,
        certified=certified,
        mode=mode,
        kappa=kappa,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# repair R4: cycle-transversal root reduction
# ---------------------------------------------------------------------------


def two_core(adj: Dict[Any, Dict[Any, float]]) -> Dict[Any, Dict[Any, float]]:
    """Iteratively peel vertices of degree <= 1 (they lie on no cycle)."""
    deg = {v: len(nb) for v, nb in adj.items()}
    alive = {v for v in adj}
    stack = [v for v in adj if deg[v] <= 1]
    while stack:
        v = stack.pop()
        if v not in alive:
            continue
        if deg[v] > 1:
            continue
        alive.discard(v)
        for u in adj[v]:
            if u in alive:
                deg[u] -= 1
                if deg[u] <= 1:
                    stack.append(u)
    return {v: {u: w for u, w in adj[v].items() if u in alive} for v in adj if v in alive}


def biconnected_components(adj: Dict[Any, Dict[Any, float]]) -> List[List[Tuple]]:
    """Iterative Hopcroft--Tarjan; returns a list of edge lists ``(u, v, w)``."""
    index = _index_map(adj)
    num: Dict[Any, int] = {}
    low: Dict[Any, int] = {}
    parent: Dict[Any, Any] = {}
    counter = 0
    estack: List[Tuple] = []
    comps: List[List[Tuple]] = []

    for start in adj:
        if start in num:
            continue
        num[start] = low[start] = counter
        counter += 1
        parent[start] = None
        it_stack = [(start, iter(sorted(adj[start], key=lambda t: index[t])))]
        while it_stack:
            u, it = it_stack[-1]
            advanced = False
            for v in it:
                if v == parent[u]:
                    continue
                if v in num:
                    if num[v] < num[u]:
                        estack.append((u, v, adj[u][v]))
                        low[u] = min(low[u], num[v])
                else:
                    estack.append((u, v, adj[u][v]))
                    parent[v] = u
                    num[v] = low[v] = counter
                    counter += 1
                    it_stack.append((v, iter(sorted(adj[v], key=lambda t: index[t]))))
                    advanced = True
                    break
            if advanced:
                continue
            it_stack.pop()
            if it_stack:
                p = it_stack[-1][0]
                low[p] = min(low[p], low[u])
                if low[u] >= num[p]:
                    comp = []
                    while estack:
                        e = estack.pop()
                        comp.append(e)
                        if e[0] == p and e[1] == u:
                            break
                    if comp:
                        comps.append(comp)
    if estack:
        comps.append(list(estack))
    return [c for c in comps if c]


def _subgraph_from_edges(edges: Sequence[Tuple]) -> Dict[Any, Dict[Any, float]]:
    g: Dict[Any, Dict[Any, float]] = {}
    for u, v, w in edges:
        g.setdefault(u, {})[v] = w
        g.setdefault(v, {})[u] = w
    return g


def spanning_forest(adj: Dict[Any, Dict[Any, float]]):
    """BFS spanning forest.

    Returns ``(tree_parent, tree_roots, non_tree_edges, order)`` where
    ``order`` is a BFS order of all vertices (roots first in each tree).
    """
    index = _index_map(adj)
    parent: Dict[Any, Any] = {}
    seen = set()
    roots: List[Any] = []
    order: List[Any] = []
    non_tree: List[Tuple] = []
    tree_edges = set()
    for s in adj:
        if s in seen:
            continue
        roots.append(s)
        seen.add(s)
        order.append(s)
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in sorted(adj[u], key=lambda t: index[t]):
                if v not in seen:
                    seen.add(v)
                    parent[v] = u
                    tree_edges.add(frozenset((u, v)))
                    order.append(v)
                    queue.append(v)
    for u, v, w in edges_of(adj, index):
        if frozenset((u, v)) not in tree_edges:
            non_tree.append((u, v, w))
    return parent, roots, non_tree, order


def _forest_distance_structure(adj, parent, order):
    """Weighted depth + binary lifting over a BFS forest (``order`` is BFS)."""
    index = {v: i for i, v in enumerate(order)}
    n = len(order)
    log = max(1, n.bit_length())
    par = [0] * n
    depth = [0] * n
    wdepth = [0.0] * n
    for v in order:
        i = index[v]
        p = parent.get(v)
        if p is None:
            par[i] = i
        else:
            j = index[p]
            par[i] = j
            depth[i] = depth[j] + 1
            wdepth[i] = wdepth[j] + adj[v][p]
    up = [par]
    for k in range(1, log):
        prev = up[k - 1]
        up.append([prev[prev[i]] for i in range(n)])

    def lca(u, v):
        iu, iv = index[u], index[v]
        if depth[iu] < depth[iv]:
            iu, iv = iv, iu
        diff = depth[iu] - depth[iv]
        k = 0
        while diff:
            if diff & 1:
                iu = up[k][iu]
            diff >>= 1
            k += 1
        if iu == iv:
            return order[iu]
        for k in range(log - 1, -1, -1):
            if up[k][iu] != up[k][iv]:
                iu, iv = up[k][iu], up[k][iv]
        return order[up[0][iu]]

    def tree_dist(u, v):
        p = lca(u, v)
        return wdepth[index[u]] + wdepth[index[v]] - 2.0 * wdepth[index[p]], p

    def path_to(u, p):
        out = [u]
        cur = u
        while cur != p:
            cur = parent[cur]
            out.append(cur)
        return out

    return tree_dist, path_to


def _transversal_for(adj):
    """Spanning-forest cycle transversal + initial bound ``gamma_0``.

    ``S`` = one endpoint of each non-tree edge, so ``|S| <= min{n, mu}`` with
    ``mu = m - n + c(G)``.  ``gamma_0 = min over non-tree e=(u,v) of
    [ l_T(u,v) + w(e) ]`` is the length of a real (fundamental) simple cycle,
    hence a valid upper bound on ``gamma*``.
    """
    parent, roots, non_tree, order = spanning_forest(adj)
    if not non_tree:
        return set(), INF, None, 0
    tree_dist, path_to = _forest_distance_structure(adj, parent, order)
    S = set()
    gamma0 = INF
    cyc0 = None
    for u, v, w in non_tree:
        S.add(u)
        d, p = tree_dist(u, v)
        cand = d + w
        if cand < gamma0:
            gamma0 = cand
            pu = path_to(u, p)          # u .. p
            pv = path_to(v, p)          # v .. p
            pu.reverse()                # p .. u
            cyc0 = tuple(pu + pv[:-1])  # p..u then v..child(p)
    return S, gamma0, cyc0, len(non_tree)


def mwc_transversal(
    adj: Dict[Any, Dict[Any, float]],
    *,
    root_order: Optional[Sequence[Any]] = None,
    use_2core: bool = True,
    use_blocks: bool = True,
    certify: bool = True,
    collect_stats: bool = True,
    allow_zero_weights: bool = False,
    tol: float = 1e-9,
) -> MWCResult:
    """Exact MWC with the roots restricted to a spanning-forest cycle
    transversal, optional 2-core peeling and biconnected-block decomposition.

    Exact mode only (``alpha = beta = 0``): the discard rule is disabled, so
    no vertex is ever removed and every transversal root really is processed.
    Since a spanning-forest transversal meets every cycle, the outer-loop
    invariant ``gamma <= l(C)`` for every cycle ``C`` touching a processed
    root gives ``gamma = gamma*``.
    """
    check_graph(adj, allow_zero=allow_zero_weights)
    n0 = len(adj)
    m0 = sum(len(d) for d in adj.values()) // 2

    work = two_core(adj) if use_2core else dict(adj)
    if use_blocks:
        blocks = [
            _subgraph_from_edges(e) for e in biconnected_components(work)
        ]
        blocks = [b for b in blocks if sum(len(d) for d in b.values()) // 2 >= len(b)]
    else:
        blocks = [work] if work else []

    gamma = INF
    best_cycle = None
    agg = {
        "n": n0,
        "m": m0,
        "n_2core": len(work),
        "m_2core": sum(len(d) for d in work.values()) // 2,
        "blocks": len(blocks),
        "roots_run": 0,
        "roots_offered": 0,
        "transversal_size": 0,
        "total_settled": 0,
        "total_edge_scans": 0,
        "total_lca_queries": 0,
        "total_deletions": 0,
        "per_block": [],
    }

    for blk in blocks:
        S, g0, c0, n_non_tree = _transversal_for(blk)
        if not S:
            continue
        # Thread the running bound across blocks: it is always a real cycle
        # length of G, hence a valid upper bound on gamma* for every block.
        # When it comes from a different block we pass it as an EXTERNAL bound
        # (no local witness cycle).
        if gamma < g0:
            g_use, c_use = gamma, None
        else:
            g_use, c_use = g0, c0
        order_b = None
        if root_order is not None:
            order_b = [v for v in root_order if v in blk]
            order_b += [v for v in blk if v not in set(order_b)]
        res = mwc(
            blk,
            alpha=0.0,
            beta=0.0,
            root_order=order_b,
            roots=S,
            certify=certify,
            collect_stats=collect_stats,
            gamma0=g_use,
            cycle0=c_use,
            roots_are_transversal=True,
            allow_zero_weights=allow_zero_weights,
            tol=tol,
        )
        if res.length < gamma:
            gamma = res.length
            best_cycle = res.cycle
        agg["transversal_size"] += len(S)
        if collect_stats and res.stats:
            agg["roots_run"] += res.stats["roots_run"]
            agg["roots_offered"] += res.stats["roots_offered"]
            agg["total_settled"] += res.stats["total_settled"]
            agg["total_edge_scans"] += res.stats["total_edge_scans"]
            agg["total_lca_queries"] += res.stats["total_lca_queries"]
            agg["total_deletions"] += res.stats["total_deletions"]
            agg["per_block"].append(
                {
                    "n": len(blk),
                    "m": sum(len(d) for d in blk.values()) // 2,
                    "transversal": len(S),
                    "non_tree_edges": n_non_tree,
                    "gamma0": g0,
                    "gamma": res.length,
                    "roots_run": res.stats["roots_run"],
                    "settled": res.stats["total_settled"],
                }
            )

    if best_cycle is not None:
        if not is_simple_cycle(adj, best_cycle):
            raise CertificationError(f"transversal cycle is not simple: {best_cycle!r}")
        rec = cycle_weight(adj, best_cycle)
        if not _close(rec, gamma, tol):
            raise CertificationError("transversal gamma / cycle weight mismatch")
    elif gamma < INF:
        raise CertificationError("finite gamma without witness cycle")

    return MWCResult(
        length=gamma,
        cycle=best_cycle,
        certified=bool(certify),
        mode="exact",
        kappa=1.0,
        stats=agg if collect_stats else {},
    )
