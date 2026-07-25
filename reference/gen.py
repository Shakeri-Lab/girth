"""Graph generators for the MWC differential test harness.

Everything returns a plain adjacency dict ``{u: {v: w}}`` with symmetric,
positive weights.  ``networkx`` is used only for *generation* (and only where
it saves real work); the algorithm itself in :mod:`mwc` is dependency-free.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional
    import networkx as nx

    HAVE_NX = True
except Exception:  # pragma: no cover
    nx = None
    HAVE_NX = False

Adj = Dict[Any, Dict[Any, float]]

__all__ = [
    "HAVE_NX",
    "weight_fn",
    "from_edges",
    "from_nx",
    "erdos_renyi",
    "geometric",
    "grid",
    "small_world",
    "preferential_attachment",
    "planted_cycle",
    "disconnected",
    "random_tree",
    "unicyclic",
    "complete",
    "atlas_graphs",
    "family_suite",
]


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

def weight_fn(kind: str, rng: random.Random, wmax: int = 10):
    """Return a zero-argument weight sampler.

    ``kind`` in {'unit', 'int', 'cont', 'wide'}.
    """
    if kind == "unit":
        return lambda: 1.0
    if kind == "int":
        return lambda: float(rng.randint(1, wmax))
    if kind == "cont":
        return lambda: rng.uniform(0.05, 1.0)
    if kind == "wide":
        return lambda: float(rng.choice([1, 1, 2, 5, 13, 40, 97]))
    raise ValueError(f"unknown weight kind {kind!r}")


def from_edges(n_or_nodes, edges: Sequence[Tuple], wf=None) -> Adj:
    """Build an adjacency dict.  ``edges`` items are ``(u, v)`` or ``(u,v,w)``."""
    if isinstance(n_or_nodes, int):
        nodes = list(range(n_or_nodes))
    else:
        nodes = list(n_or_nodes)
    adj: Adj = {v: {} for v in nodes}
    for e in edges:
        if len(e) == 3:
            u, v, w = e
        else:
            u, v = e
            w = wf() if wf is not None else 1.0
        if u == v:
            continue
        adj.setdefault(u, {})
        adj.setdefault(v, {})
        if v in adj[u]:
            continue
        adj[u][v] = float(w)
        adj[v][u] = float(w)
    return adj


def from_nx(g, wf=None, relabel: bool = True) -> Adj:
    """Convert a networkx graph, assigning weights with *wf* if absent."""
    nodes = list(g.nodes())
    if relabel:
        remap = {v: i for i, v in enumerate(nodes)}
    else:
        remap = {v: v for v in nodes}
    adj: Adj = {remap[v]: {} for v in nodes}
    for u, v, data in g.edges(data=True):
        if u == v:
            continue
        w = data.get("weight")
        if w is None:
            w = wf() if wf is not None else 1.0
        a, b = remap[u], remap[v]
        adj[a][b] = float(w)
        adj[b][a] = float(w)
    return adj


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------

def erdos_renyi(n: int, p: float, rng: random.Random, wkind: str = "int") -> Adj:
    wf = weight_fn(wkind, rng)
    edges = [(u, v) for u, v in itertools.combinations(range(n), 2) if rng.random() < p]
    return from_edges(n, edges, wf)


def geometric(n: int, radius: float, rng: random.Random, wkind: str = "cont") -> Adj:
    pts = [(rng.random(), rng.random()) for _ in range(n)]
    wf = weight_fn(wkind, rng)
    edges = []
    for u, v in itertools.combinations(range(n), 2):
        dx = pts[u][0] - pts[v][0]
        dy = pts[u][1] - pts[v][1]
        d = math.hypot(dx, dy)
        if d <= radius:
            edges.append((u, v, d if wkind == "euclid" else wf()))
    return from_edges(n, edges)


def grid(rows: int, cols: int, rng: random.Random, wkind: str = "int") -> Adj:
    wf = weight_fn(wkind, rng)
    idx = lambda r, c: r * cols + c
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    return from_edges(rows * cols, edges, wf)


def small_world(n: int, k: int, p: float, rng: random.Random, wkind: str = "int") -> Adj:
    """Watts--Strogatz ring rewiring (pure python)."""
    k = max(2, k - k % 2)
    edges = set()
    for i in range(n):
        for j in range(1, k // 2 + 1):
            edges.add(frozenset((i, (i + j) % n)))
    edges = [tuple(sorted(e)) for e in edges]
    out = set(map(frozenset, edges))
    for e in list(out):
        if rng.random() < p:
            u, v = tuple(e)
            for _ in range(20):
                w = rng.randrange(n)
                if w != u and frozenset((u, w)) not in out:
                    out.discard(e)
                    out.add(frozenset((u, w)))
                    break
    wf = weight_fn(wkind, rng)
    return from_edges(n, [tuple(e) for e in out], wf)


def preferential_attachment(n: int, m: int, rng: random.Random, wkind: str = "int") -> Adj:
    m = max(1, min(m, n - 1))
    targets = list(range(m))
    repeated = list(range(m))
    edges = []
    for src in range(m, n):
        chosen = set()
        while len(chosen) < m:
            chosen.add(rng.choice(repeated) if repeated else rng.randrange(src))
        for t in chosen:
            edges.append((src, t))
            repeated.append(t)
        repeated.extend([src] * m)
    wf = weight_fn(wkind, rng)
    return from_edges(n, edges, wf)


def planted_cycle(
    n: int, k: int, rng: random.Random, cycle_w: float = 1.0, bulk_w: float = 20.0,
    extra_p: float = 0.15,
) -> Adj:
    """A light ``k``-cycle planted inside a heavy random graph.

    Forces the MWC to be the planted cycle when ``k*cycle_w`` is small enough,
    which stresses the truncation horizon (it becomes tiny very quickly).
    """
    edges = []
    cyc = rng.sample(range(n), k)
    for i in range(k):
        edges.append((cyc[i], cyc[(i + 1) % k], cycle_w))
    have = {frozenset((cyc[i], cyc[(i + 1) % k])) for i in range(k)}
    for u, v in itertools.combinations(range(n), 2):
        if frozenset((u, v)) in have:
            continue
        if rng.random() < extra_p:
            edges.append((u, v, bulk_w * rng.uniform(0.8, 1.2)))
    return from_edges(n, edges)


def disconnected(parts: Sequence[Adj]) -> Adj:
    """Disjoint union, relabelling each part as ``(i, v)``."""
    out: Adj = {}
    for i, g in enumerate(parts):
        for u, nbrs in g.items():
            out[(i, u)] = {(i, v): w for v, w in nbrs.items()}
    return out


def random_tree(n: int, rng: random.Random, wkind: str = "int") -> Adj:
    wf = weight_fn(wkind, rng)
    edges = [(v, rng.randrange(v)) for v in range(1, n)]
    return from_edges(n, edges, wf)


def unicyclic(n: int, k: int, rng: random.Random, wkind: str = "int") -> Adj:
    """A tree plus exactly one extra edge closing a ``k``-cycle."""
    wf = weight_fn(wkind, rng)
    assert 3 <= k <= n
    edges = [(i, i + 1) for i in range(k - 1)]
    edges.append((k - 1, 0))
    for v in range(k, n):
        edges.append((v, rng.randrange(v)))
    return from_edges(n, edges, wf)


def complete(n: int, rng: random.Random, wkind: str = "int") -> Adj:
    wf = weight_fn(wkind, rng)
    return from_edges(n, list(itertools.combinations(range(n), 2)), wf)


MULTISCALE = [(0.01, 0.15), (0.9, 1.1), (4.5, 5.0), (9.5, 10.5)]


def multiscale(n: int, rng: random.Random, p: Optional[float] = None) -> Adj:
    """Random graph whose edge weights are drawn from well-separated scales.

    This is the family on which the (alpha, beta) discard rule actually
    *fires* -- uniform weights essentially never trigger it, because the
    truncation horizon gamma/2 collapses before any decoy cycle can become the
    composite minimiser.
    """
    if p is None:
        p = rng.uniform(0.3, 0.9)
    edges = []
    for u, v in itertools.combinations(range(n), 2):
        if rng.random() < p:
            lo, hi = rng.choice(MULTISCALE)
            edges.append((u, v, round(rng.uniform(lo, hi), 4)))
    return from_edges(n, edges)


def hub_gadget(G0: float = 10.0, D: float = 0.3, R: float = 4.6,
               s: float = 1.0, k: int = 5, k0: int = 5) -> Adj:
    """Deterministic gadget that provokes the parameterized discard rule.

    Component ``A``: a uniform ``k0``-cycle of total weight ``G0`` -- it fixes
    the global bound before the second component is touched.
    Component ``B``: root ``x`` -- hub ``h`` (weight ``D``), ``h`` joined to
    every vertex of a uniform ``k``-cycle (step ``s``) by weight ``R``.

    From ``x`` the only fundamental cycles are the triangles ``h,v_i,v_{i+1}``
    of length ``2R + s > G0``; the ``k``-cycle itself is fully settled but is
    NOT fundamental, so it stays invisible.  The composite minimiser therefore
    sits at ``d_to_cycle = delta(h) = D``, and the sweep deletes ``h``.
    """
    edges = []
    for i in range(k0):
        edges.append((("A", i), ("A", (i + 1) % k0), G0 / k0))
    edges.append((("B", "x"), ("B", "h"), D))
    for i in range(k):
        edges.append((("B", "h"), ("B", i), R))
        edges.append((("B", i), ("B", (i + 1) % k), s))
    nodes = [("A", i) for i in range(k0)] + [("B", "x"), ("B", "h")]
    nodes += [("B", i) for i in range(k)]
    return from_edges(nodes, edges)


def atlas_graphs(max_nodes: int = 7, min_nodes: int = 3):
    """Yield ``(name, nx_graph)`` for every graph atlas entry up to *max_nodes*."""
    if not HAVE_NX:  # pragma: no cover
        return
    from networkx.generators.atlas import graph_atlas_g

    for i, g in enumerate(graph_atlas_g()):
        if g.number_of_nodes() < min_nodes or g.number_of_nodes() > max_nodes:
            continue
        yield f"atlas{i}", g


# ---------------------------------------------------------------------------
# the sweep used by the test suite
# ---------------------------------------------------------------------------

def family_suite(rng: random.Random, count: int = 400) -> List[Tuple[str, Adj]]:
    """A deterministic mixed bag of graphs spanning every family."""
    out: List[Tuple[str, Adj]] = []
    per = max(1, count // 12)
    for i in range(per):
        out.append(("multiscale", multiscale(rng.randint(5, 11), rng)))
    for i in range(max(1, per // 4)):
        out.append(("hub_gadget", hub_gadget(
            G0=rng.choice([8.0, 10.0, 12.0]),
            D=rng.uniform(0.1, 0.6),
            R=rng.uniform(4.0, 4.8),
            s=rng.uniform(0.5, 1.5),
            k=rng.randint(4, 7),
        )))
    for i in range(per):
        n = rng.randint(4, 16)
        out.append(("erdos_renyi", erdos_renyi(n, rng.uniform(0.15, 0.6), rng,
                                               rng.choice(["unit", "int", "cont", "wide"]))))
    for i in range(per):
        n = rng.randint(6, 20)
        out.append(("geometric", geometric(n, rng.uniform(0.25, 0.55), rng,
                                           rng.choice(["cont", "int"]))))
    for i in range(per):
        r = rng.randint(2, 5)
        c = rng.randint(2, 5)
        out.append(("grid", grid(r, c, rng, rng.choice(["unit", "int", "cont"]))))
    for i in range(per):
        n = rng.randint(6, 18)
        out.append(("small_world", small_world(n, rng.choice([2, 4]), rng.uniform(0.0, 0.4),
                                               rng, rng.choice(["unit", "int"]))))
    for i in range(per):
        n = rng.randint(5, 18)
        out.append(("pref_attach", preferential_attachment(n, rng.choice([1, 2, 3]), rng,
                                                           rng.choice(["int", "cont"]))))
    for i in range(per):
        n = rng.randint(8, 16)
        k = rng.randint(3, min(7, n))
        out.append(("planted_cycle", planted_cycle(n, k, rng)))
    for i in range(per):
        n = rng.randint(3, 10)
        out.append(("tree", random_tree(n, rng, rng.choice(["unit", "int"]))))
    for i in range(per):
        n = rng.randint(4, 12)
        k = rng.randint(3, n)
        out.append(("unicyclic", unicyclic(n, k, rng, rng.choice(["unit", "int", "cont"]))))
    for i in range(per):
        n = rng.randint(3, 7)
        out.append(("complete", complete(n, rng, rng.choice(["unit", "int", "cont"]))))
    for i in range(per):
        parts = [
            erdos_renyi(rng.randint(3, 8), rng.uniform(0.2, 0.7), rng, "int")
            for _ in range(rng.randint(2, 4))
        ]
        out.append(("disconnected", disconnected(parts)))
    return out
