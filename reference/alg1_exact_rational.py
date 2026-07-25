"""Faithful simulator of Algorithm 1 (MWC via composite-distance minimization).

Graph: dict node -> dict nbr -> weight (simple, undirected, positive weights).
Root order: explicit list of all vertices.

Follows main.tex lines 205-248 literally:
  - inner while guard: exists v notin Q with delta(v) < gamma/2  (gamma is LIVE,
    it can drop mid-search, shrinking the horizon)
  - y = argmin_{v notin Q} delta(v)
  - scan neighbours z of y with z in V_active
  - z notin Q -> relax;  z in Q and z != pred(y) -> LCA / cycle detect
  - gamma <- min(gamma, l_c) immediately on every detection
  - after the search: if d+_min < inf and l_best > gamma and d+_min < (1+alpha)gamma
    then discard every not-yet-processed active z != x with delta(z) <= d_to_cycle - beta*gamma
"""
import heapq
from itertools import count

INF = float('inf')


def lca(pred, root, a, b):
    """LCA of a,b in the pred-forest rooted at `root`."""
    anc = []
    u = a
    seen = set()
    while u is not None:
        anc.append(u)
        seen.add(u)
        if u == root:
            break
        u = pred.get(u)
    u = b
    while u is not None:
        if u in seen:
            return u
        if u == root:
            break
        u = pred.get(u)
    return root


def run_alg1(adj, order, alpha, beta, trace=False, tol=1e-12):
    gamma = INF
    active = set(adj.keys())
    processed = set()
    log = []
    for x in order:
        if x not in active:
            continue
        processed.add(x)
        delta = {v: INF for v in adj}
        pred = {}
        delta[x] = 0.0
        Q = set()
        dpmin = INF
        d_to_cycle = INF
        l_best = INF
        heap = [(0.0, next_tie := 0, x)]
        tie = count(1)
        events = []
        while True:
            # pop the min-delta unsettled vertex
            y = None
            while heap:
                dy, _, cand = heapq.heappop(heap)
                if cand in Q:
                    continue
                if dy > delta[cand] + tol:
                    continue
                y = cand
                break
            if y is None:
                break
            if not (delta[y] < gamma / 2.0 - tol):     # inner while guard
                break
            Q.add(y)
            events.append((y, delta[y]))
            for z, w in sorted(adj[y].items(), key=lambda t: str(t[0])):
                if z not in active:
                    continue
                if z not in Q:
                    if delta[y] + w < delta[z] - tol:
                        delta[z] = delta[y] + w
                        pred[z] = y
                        heapq.heappush(heap, (delta[z], next(tie), z))
                elif z != pred.get(y):
                    p = lca(pred, x, y, z)
                    lc = delta[y] + delta[z] + w - 2 * delta[p]
                    dxc = delta[p]
                    dp = dxc + lc
                    if lc < gamma:
                        gamma = lc
                    if dp < dpmin - tol:
                        dpmin = dp
                        d_to_cycle = dxc
                        l_best = lc
        fired = False
        removed = []
        if dpmin < INF and l_best > gamma + tol and dpmin < (1 + alpha) * gamma - tol:
            fired = True
            thr = d_to_cycle - beta * gamma
            for z in list(active):
                if z == x or z in processed:
                    continue
                if delta[z] <= thr + tol:
                    active.discard(z)
                    removed.append(z)
        log.append(dict(root=x, gamma_after=gamma, dpmin=dpmin, d_to_cycle=d_to_cycle,
                        l_best=l_best, fired=fired, removed=sorted(map(str, removed)),
                        removed_raw=list(removed), delta=dict(delta), Q=set(Q),
                        active=set(active),
                        settled=[(str(a), round(b, 6)) for a, b in events]))
        if trace:
            print(f"root={x} |Q|={len(Q)} gamma={gamma:.6f} dpmin={dpmin:.6f} "
                  f"r={d_to_cycle:.6f} lbest={l_best:.6f} fired={fired} rm={removed}")
    return gamma, log


# ---------- ground truth ----------
def true_girth(adj):
    """Exact minimum weight cycle by Dijkstra-from-every-vertex + non-tree edge check
    (no truncation, no discarding). Uses the standard 'shortest cycle through each
    edge' formulation, exact for positive weights."""
    import networkx as nx
    G = nx.Graph()
    for u in adj:
        G.add_node(u)
        for v, w in adj[u].items():
            G.add_edge(u, v, weight=w)
    best = INF
    for (u, v, w) in list(G.edges(data='weight')):
        G.remove_edge(u, v)
        try:
            d = nx.shortest_path_length(G, u, v, weight='weight')
            best = min(best, d + w)
        except nx.NetworkXNoPath:
            pass
        G.add_edge(u, v, weight=w)
    return best


def path(adj, a, b, total, nseg, tag):
    """Add a simple path a -> b of total weight `total` with nseg segments,
    through fresh vertices named f'{tag}#i'."""
    assert nseg >= 1
    seg = total / nseg
    prev = a
    for i in range(1, nseg):
        m = f"{tag}#{i}"
        add(adj, prev, m, seg)
        prev = m
    add(adj, prev, b, seg)


def add(adj, u, v, w):
    adj.setdefault(u, {})[v] = w
    adj.setdefault(v, {})[u] = w
