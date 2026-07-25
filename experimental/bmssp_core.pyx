# cython: language_level=3
"""Cython-accelerated core relaxation routine for BMSSP.

This file is a minimal drop-in replacement for the Python edge-relaxation loop
inside `girth.bmssp_full.BMSSPFullStrategy._bmssp`.  It keeps the original data
structures (plain Python `dict`s, NetworkX `Graph`, and the existing
`BlockBasedPQ`) so no higher-level code needs to change.  Even with Python
objects, executing the innermost loop in C reduces interpreter overhead and
provides a 2-3× speed-up on ≥2 k-node graphs (profiling numbers in README).

Notes
-----
1.  We do *not* depend on NumPy – pure Python and NetworkX only.
2.  The routine is written as a `def` (not `cpdef`) so it can be called
    seamlessly from regular Python.  If the compiled extension is missing the
    import will fail gracefully and BMSSP falls back to the original pure-
    Python loop.
"""

from libc.math cimport INFINITY

# ------------------------------------------------------------
# Public API – callable from Python
# ------------------------------------------------------------

def relax_batch(nodes, dict dist, dict pred, dict depth,
                 object G, double B_i, double B, object pq):
    """Relax outgoing edges for all *nodes*.

    Parameters
    ----------
    nodes : iterable of hashable
        Vertex ids whose outgoing edges should be relaxed.
    dist, pred, depth : dict
        Data structures updated in-place.  Same semantics as in
        `BMSSPFullStrategy`.
    G : networkx.Graph
        The graph (undirected / directed) providing adjacency and weights.
    B_i : float
        Local bound (distance threshold for this batch).
    B : float
        Global bound – distances above this value are disregarded.
    pq : BlockBasedPQ
        Priority queue used by BMSSP recursion.
    """

    cdef double du, nd, weight
    cdef object u, v, nbrs, data

    for u in nodes:
        du = <double>dist.get(u, INFINITY)
        if du > B_i:
            continue

        nbrs = G[u]               # adjacency mapping for *u*
        # Equivalent to: for v, data in G[u].items():
        for v, data in nbrs.items():
            weight = data.get("weight", 1.0)
            nd = du + weight
            if nd <= B and nd < dist.get(v, INFINITY):
                dist[v] = nd
                pred[v] = u
                depth[v] = depth[u] + 1
                pq.insert(nd, v)
