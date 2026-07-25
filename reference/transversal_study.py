"""
R4: cycle-transversal root reduction for the composite-distance MWC algorithm.

Reference implementation + adversarial numerical validation.

Contents
--------
  oracle_mwc_edge          exact MWC via  min_e [ d_{G-e}(u,v) + w(e) ]
  oracle_mwc_enum          exact MWC via full simple-cycle enumeration (small graphs)
  dfs_forest               spanning forest, tree/non-tree edge split          O(n+m)
  transversal_endpoint     S = one chosen endpoint per non-tree edge          O(n+m)
  transversal_greedy_vc    S = greedy vertex cover of the non-tree edge set   O(n+m log n)
  gamma0_bound             gamma_0 = min_{e non-tree} [ l_T(u,v) + w(e) ]     O(n+m) w/ LCA
  two_core                 2-core peeling                                     O(n+m)
  alg1                     faithful Algorithm 1 (exact mode) with a user-supplied root list
  run_pipeline             2-core -> blocks -> per-block transversal -> gamma_0 -> alg1

Usage:  python transversal.py            (runs the full validation suite)
"""

import heapq
import math
import random
import sys
from collections import defaultdict, deque

import networkx as nx

INF = float("inf")
TOL = 1e-9


# --------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------
def _dijkstra(G, s, banned=None):
    dist = {s: 0.0}
    pq = [(0.0, s)]
    done = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        for v, data in G[u].items():
            if banned is not None and ((u, v) == banned or (v, u) == banned):
                continue
            nd = d + data["weight"]
            if nd < dist.get(v, INF) - TOL:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def oracle_mwc_edge(G):
    """Exact minimum weight cycle: min over edges e=(u,v) of d_{G-e}(u,v)+w(e)."""
    best = INF
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        dist = _dijkstra(G, u, banned=(u, v))
        d = dist.get(v, INF)
        if d + data["weight"] < best:
            best = d + data["weight"]
    return best


def oracle_mwc_enum(G):
    """Exact MWC by enumerating all simple cycles (only for tiny graphs)."""
    best = INF
    for cyc in nx.simple_cycles(G):
        k = len(cyc)
        if k < 3:
            continue
        L = sum(G[cyc[i]][cyc[(i + 1) % k]]["weight"] for i in range(k))
        best = min(best, L)
    return best


# --------------------------------------------------------------------------
# Spanning forest, transversal, gamma_0
# --------------------------------------------------------------------------
def dfs_forest(G, order=None):
    """Iterative DFS spanning forest.  Returns (parent, tree_edges, nontree_edges).

    O(n+m).  Works on disconnected graphs (one tree per component).
    """
    parent = {}
    visited = set()
    tree_edges = set()
    nodes = list(G.nodes()) if order is None else list(order)
    for r in nodes:
        if r in visited:
            continue
        visited.add(r)
        parent[r] = None
        stack = [(r, iter(G[r]))]
        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    tree_edges.add(frozenset((u, v)))
                    stack.append((v, iter(G[v])))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
    nontree = [(u, v) for u, v in G.edges() if frozenset((u, v)) not in tree_edges]
    return parent, tree_edges, nontree


def transversal_endpoint(G, order=None, pick=0):
    """S = { one chosen endpoint of each non-tree edge }.  |S| <= min(n, mu)."""
    _, _, nontree = dfs_forest(G, order)
    S = []
    seen = set()
    for e in nontree:
        x = e[pick]
        if x not in seen:
            seen.add(x)
            S.append(x)
    return S


def transversal_greedy_vc(G, order=None):
    """S = greedy (max-degree-first) vertex cover of the non-tree edge set.

    Any vertex cover of (V, E\\T) is a cycle transversal, since every cycle
    contains a non-tree edge.  Never larger than transversal_endpoint's output
    in practice; still |S| <= mu.
    """
    _, _, nontree = dfs_forest(G, order)
    remaining = set(map(lambda e: (min(e, key=str), max(e, key=str)), nontree))
    deg = defaultdict(int)
    for u, v in remaining:
        deg[u] += 1
        deg[v] += 1
    S = []
    while remaining:
        x = max(deg, key=lambda z: (deg[z], str(z)))
        if deg[x] == 0:
            break
        S.append(x)
        gone = [e for e in remaining if x in e]
        for e in gone:
            remaining.discard(e)
            deg[e[0]] -= 1
            deg[e[1]] -= 1
        deg[x] = 0
    return S


def _depths_and_lca_prep(G, parent, tree_edges):
    """Weighted depth (root distance in T) and hop-depth, for LCA by walking up."""
    wdepth, hdepth = {}, {}
    children = defaultdict(list)
    roots = []
    for v, p in parent.items():
        if p is None:
            roots.append(v)
        else:
            children[p].append(v)
    for r in roots:
        wdepth[r] = 0.0
        hdepth[r] = 0
        stack = [r]
        while stack:
            u = stack.pop()
            for c in children[u]:
                wdepth[c] = wdepth[u] + G[u][c]["weight"]
                hdepth[c] = hdepth[u] + 1
                stack.append(c)
    return wdepth, hdepth


def _tree_lca(u, v, parent, hdepth):
    while hdepth[u] > hdepth[v]:
        u = parent[u]
    while hdepth[v] > hdepth[u]:
        v = parent[v]
    while u != v:
        u = parent[u]
        v = parent[v]
    return u


def gamma0_bound(G, order=None):
    """gamma_0 = min over non-tree e=(u,v) of [ l_T(u,v) + w(e) ].

    Each term is the length of the fundamental cycle of e, hence a genuine
    simple-cycle length; so gamma* <= gamma_0 < INF whenever G has a cycle.
    Returns INF for forests.
    """
    parent, tree_edges, nontree = dfs_forest(G, order)
    wdepth, hdepth = _depths_and_lca_prep(G, parent, tree_edges)
    best = INF
    for u, v in nontree:
        p = _tree_lca(u, v, parent, hdepth)
        L = wdepth[u] + wdepth[v] - 2 * wdepth[p] + G[u][v]["weight"]
        best = min(best, L)
    return best


def two_core(G):
    """Vertex set of the 2-core (iteratively delete vertices of degree <= 1)."""
    deg = dict(G.degree())
    alive = set(G.nodes())
    dq = deque([v for v in alive if deg[v] <= 1])
    while dq:
        v = dq.popleft()
        if v not in alive or deg[v] > 1:
            continue
        alive.discard(v)
        for u in G[v]:
            if u in alive:
                deg[u] -= 1
                if deg[u] <= 1:
                    dq.append(u)
    return alive


# --------------------------------------------------------------------------
# Algorithm 1 (exact mode: alpha <= beta, no discarding), faithful transcription
# --------------------------------------------------------------------------
def alg1(G, roots, gamma_init=INF, tie=None):
    """Run Algorithm 1's outer loop over the given root list, exact mode.

    Returns (gamma, stats) where stats has the number of settled vertices and
    the number of scanned incidences summed over all roots.

    `tie` is an optional dict node->float used only to break priority ties,
    so the adversarial tie behaviour can be randomised.
    """
    gamma = gamma_init
    settled_total = 0
    scanned_total = 0
    argmin_ops = 0
    tie = tie or {}

    for x in roots:
        if x not in G:
            continue
        delta = {x: 0.0}
        pred = {}
        hdep = {x: 0}
        Q = set()
        pq = [(0.0, tie.get(x, 0.0), x)]
        while True:
            # while-condition of Line 9, re-evaluated with the CURRENT gamma
            y = None
            while pq:
                d, _, cand = pq[0]
                if cand in Q or d > delta[cand] + TOL:
                    heapq.heappop(pq)
                    continue
                if d < gamma / 2.0:          # STRICT, as in the pseudocode
                    heapq.heappop(pq)
                    y = cand
                    argmin_ops += 1
                break
            if y is None:
                break
            Q.add(y)
            settled_total += 1
            if y != x:
                hdep[y] = hdep[pred[y]] + 1
            for z in G[y]:
                scanned_total += 1
                w = G[y][z]["weight"]
                if z not in Q:
                    if delta[y] + w < delta.get(z, INF) - TOL:
                        delta[z] = delta[y] + w
                        pred[z] = y
                        heapq.heappush(pq, (delta[z], tie.get(z, 0.0), z))
                else:
                    if pred.get(y, None) != z:
                        p = _tree_lca(y, z, pred_fn(pred, x), hdep)
                        lc = delta[y] + delta[z] + w - 2 * delta[p]
                        if lc < gamma:
                            gamma = lc
    return gamma, {
        "settled": settled_total,
        "scanned": scanned_total,
        "argmin": argmin_ops,
        "roots": len([r for r in roots if r in G]),
    }


class pred_fn(dict):
    """pred with pred[root] = None so _tree_lca can walk up uniformly."""

    def __init__(self, pred, root):
        super().__init__(pred)
        self[root] = None


# --------------------------------------------------------------------------
# Full pipeline: 2-core -> blocks -> per-block transversal + global gamma_0
# --------------------------------------------------------------------------
def run_pipeline(G, use_core=True, use_blocks=True, use_gamma0=True,
                 transversal=transversal_endpoint, tie=None, shuffle=False,
                 rng=None):
    H = G
    if use_core:
        core = two_core(G)
        H = G.subgraph(core).copy()
    if H.number_of_edges() == 0:
        return INF, {"settled": 0, "scanned": 0, "argmin": 0, "roots": 0, "S": 0}

    gamma = gamma0_bound(H) if use_gamma0 else INF

    pieces = []
    if use_blocks:
        for comp in nx.connected_components(H):
            sub = H.subgraph(comp)
            for blk in nx.biconnected_components(sub):
                B = sub.subgraph(blk)
                # a block of a SIMPLE graph is cyclic iff |V(B)|>=3
                if B.number_of_nodes() >= 3:
                    pieces.append(B.copy())
    else:
        pieces = [H]

    total = {"settled": 0, "scanned": 0, "argmin": 0, "roots": 0, "S": 0}
    for B in pieces:
        S = transversal(B)
        if shuffle and rng is not None:
            rng.shuffle(S)
        total["S"] += len(S)
        gamma, st = alg1(B, S, gamma_init=gamma, tie=tie)
        for k in ("settled", "scanned", "argmin", "roots"):
            total[k] += st[k]
    return gamma, total


# --------------------------------------------------------------------------
# Random graph families
# --------------------------------------------------------------------------
def _weightify(G, rng, mode):
    for u, v in G.edges():
        if mode == "unit":
            G[u][v]["weight"] = 1.0
        elif mode == "smallint":
            G[u][v]["weight"] = float(rng.randint(1, 3))
        elif mode == "int":
            G[u][v]["weight"] = float(rng.randint(1, 20))
        else:
            G[u][v]["weight"] = round(rng.uniform(0.05, 5.0), 6)
    return G


def gen_graph(rng, family=None):
    fams = ["er", "er_sparse", "dense", "tree", "unicyclic", "grid", "planted",
            "disconnected", "complete", "bipartite", "theta", "cycle_chords",
            "barbell", "two_blocks"]
    family = family or rng.choice(fams)
    wmode = rng.choice(["unit", "smallint", "int", "float", "float"])
    if family == "er":
        n = rng.randint(5, 13)
        G = nx.gnp_random_graph(n, rng.uniform(0.2, 0.5), seed=rng.randint(0, 10**9))
    elif family == "er_sparse":
        n = rng.randint(8, 20)
        G = nx.gnp_random_graph(n, rng.uniform(0.08, 0.2), seed=rng.randint(0, 10**9))
    elif family == "dense":
        n = rng.randint(5, 10)
        G = nx.gnp_random_graph(n, rng.uniform(0.6, 0.9), seed=rng.randint(0, 10**9))
    elif family == "tree":
        n = rng.randint(2, 15)
        G = nx.random_labeled_tree(n, seed=rng.randint(0, 10**9)) if n > 1 else nx.empty_graph(1)
    elif family == "unicyclic":
        n = rng.randint(4, 15)
        G = nx.random_labeled_tree(n, seed=rng.randint(0, 10**9))
        nonedges = [e for e in nx.non_edges(G)]
        if nonedges:
            G.add_edge(*rng.choice(nonedges))
    elif family == "grid":
        a, b = rng.randint(2, 4), rng.randint(2, 4)
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(a, b))
    elif family == "planted":
        n = rng.randint(8, 16)
        G = nx.gnp_random_graph(n, 0.15, seed=rng.randint(0, 10**9))
        k = rng.randint(3, min(6, n))
        vs = rng.sample(list(G.nodes()), k)
        for i in range(k):
            G.add_edge(vs[i], vs[(i + 1) % k])
        G = _weightify(G, rng, wmode)
        for i in range(k):
            G[vs[i]][vs[(i + 1) % k]]["weight"] = round(rng.uniform(0.01, 0.1), 6)
        return G, family, wmode
    elif family == "disconnected":
        parts = []
        for _ in range(rng.randint(2, 4)):
            n = rng.randint(1, 7)
            parts.append(nx.gnp_random_graph(n, rng.uniform(0.2, 0.8),
                                             seed=rng.randint(0, 10**9)))
        G = nx.disjoint_union_all(parts)
    elif family == "complete":
        G = nx.complete_graph(rng.randint(3, 7))
    elif family == "bipartite":
        a, b = rng.randint(2, 5), rng.randint(2, 5)
        G = nx.convert_node_labels_to_integers(
            nx.bipartite.random_graph(a, b, rng.uniform(0.4, 0.9),
                                      seed=rng.randint(0, 10**9)))
    elif family == "theta":
        # two hubs joined by k internally disjoint paths
        k = rng.randint(2, 4)
        G = nx.Graph()
        s, t = "s", "t"
        idx = 0
        for i in range(k):
            L = rng.randint(1, 4)
            prev = s
            for j in range(L):
                G.add_edge(prev, ("p", idx))
                prev = ("p", idx)
                idx += 1
            G.add_edge(prev, t)
        G = nx.convert_node_labels_to_integers(G)
    elif family == "cycle_chords":
        n = rng.randint(5, 12)
        G = nx.cycle_graph(n)
        for _ in range(rng.randint(0, 4)):
            ne = list(nx.non_edges(G))
            if ne:
                G.add_edge(*rng.choice(ne))
    elif family == "barbell":
        G = nx.barbell_graph(rng.randint(3, 5), rng.randint(0, 4))
    else:  # two_blocks: two cycles sharing exactly one cut vertex
        a, b = rng.randint(3, 6), rng.randint(3, 6)
        G = nx.Graph()
        for i in range(a):
            G.add_edge(("A", i), ("A", (i + 1) % a))
        for i in range(b):
            G.add_edge(("B", i), ("B", (i + 1) % b))
        G.add_edge(("A", 0), ("B", 0))
        G = nx.convert_node_labels_to_integers(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return _weightify(G, rng, wmode), family, wmode


# --------------------------------------------------------------------------
# Validation suite
# --------------------------------------------------------------------------
def validate(trials=4000, seed=12345, verbose=True):
    rng = random.Random(seed)
    mismatches = []
    fam_stats = defaultdict(lambda: [0, 0])
    for t in range(trials):
        G, fam, wmode = gen_graph(rng)
        if G.number_of_nodes() == 0:
            continue
        n = G.number_of_nodes()
        m = G.number_of_edges()
        c = nx.number_connected_components(G)
        mu = m - n + c

        truth = oracle_mwc_edge(G)
        if n <= 9 and m <= 18:
            t2 = oracle_mwc_enum(G)
            if abs(min(t2, INF) - truth) > 1e-7 and not (t2 == INF and truth == INF):
                mismatches.append(("ORACLE-DISAGREE", fam, wmode,
                                   nx.to_dict_of_lists(G),
                                   {(u, v): d["weight"] for u, v, d in G.edges(data=True)},
                                   truth, t2))
                continue

        tie = {v: rng.random() for v in G.nodes()}
        variants = {}
        # 1. all roots, no gamma_0  (baseline / manuscript Algorithm 1)
        order = list(G.nodes())
        rng.shuffle(order)
        variants["all"] = alg1(G, order, tie=tie)[0]
        # 2. transversal roots only (endpoint rule)
        S = transversal_endpoint(G)
        rng.shuffle(S)
        variants["S_endpoint"] = alg1(G, S, tie=tie)[0]
        # 3. transversal roots, other endpoint choice
        S2 = transversal_endpoint(G, pick=1)
        rng.shuffle(S2)
        variants["S_endpoint2"] = alg1(G, S2, tie=tie)[0]
        # 4. greedy vertex-cover transversal
        S3 = transversal_greedy_vc(G)
        rng.shuffle(S3)
        variants["S_greedyVC"] = alg1(G, S3, tie=tie)[0]
        # 5. transversal + gamma_0
        variants["S_gamma0"] = alg1(G, S, gamma_init=gamma0_bound(G), tie=tie)[0]
        # 6. full pipeline (2-core + blocks + per-block transversal + gamma_0)
        variants["pipeline"] = run_pipeline(G, tie=tie, shuffle=True, rng=rng)[0]
        # 7. pipeline with greedy-VC transversal, no gamma_0
        variants["pipeline_novc"] = run_pipeline(
            G, use_gamma0=False, transversal=transversal_greedy_vc,
            tie=tie, shuffle=True, rng=rng)[0]
        # 8. worst-case root order: sorted so that S roots come in a fixed order
        variants["S_sorted"] = alg1(G, sorted(transversal_endpoint(G), key=str),
                                    tie=tie)[0]

        # structural checks
        assert len(S) <= min(n, mu) or mu < 0, (len(S), n, mu)
        assert len(S3) <= min(n, mu) or mu < 0, (len(S3), n, mu)
        g0 = gamma0_bound(G)
        if truth < INF:
            assert g0 >= truth - 1e-9, ("gamma0 < gamma*", g0, truth)
            assert g0 < INF
        else:
            assert g0 == INF
        # S meets every cycle (verified by brute force on small graphs)
        if n <= 9 and m <= 16:
            for cyc in nx.simple_cycles(G):
                if len(cyc) >= 3:
                    assert set(cyc) & set(S), ("S misses a cycle", cyc, S)
                    assert set(cyc) & set(S3), ("S3 misses a cycle", cyc, S3)

        fam_stats[fam][1] += 1
        bad = {k: v for k, v in variants.items() if abs(v - truth) > 1e-7
               and not (v == INF and truth == INF)}
        if bad:
            mismatches.append((fam, wmode, nx.to_dict_of_lists(G),
                               {(u, v): d["weight"] for u, v, d in G.edges(data=True)},
                               truth, bad))
            fam_stats[fam][0] += 1
            if verbose and len(mismatches) <= 5:
                print("MISMATCH", fam, wmode, "truth=", truth, "got=", bad)
    if verbose:
        print(f"\n== validation: {trials} random graphs ==")
        for fam in sorted(fam_stats):
            bad, tot = fam_stats[fam]
            print(f"  {fam:14s} {tot:5d} graphs   mismatches: {bad}")
        print(f"  TOTAL MISMATCHES: {len(mismatches)}")
    return mismatches


# --------------------------------------------------------------------------
# Empirical reduction study
# --------------------------------------------------------------------------
def study(seed=7):
    rng = random.Random(seed)
    print("\n== empirical root-count and work reduction ==")
    hdr = (f"{'family':22s} {'n':>5s} {'m':>6s} {'mu':>6s} "
           f"{'|S|':>5s} {'|S|/n':>6s} {'|Svc|':>5s} {'Svc/n':>6s} "
           f"{'settled_all':>12s} {'settled_S':>10s} {'ratio':>6s} {'pipe':>8s} {'p/all':>6s}")
    print(hdr)
    print("-" * len(hdr))

    cases = []
    cases.append(("grid 20x20", nx.convert_node_labels_to_integers(nx.grid_2d_graph(20, 20))))
    cases.append(("grid 30x30", nx.convert_node_labels_to_integers(nx.grid_2d_graph(30, 30))))
    cases.append(("ER n=200 p=0.03", nx.gnp_random_graph(200, 0.03, seed=1)))
    cases.append(("ER n=200 p=0.1", nx.gnp_random_graph(200, 0.1, seed=2)))
    cases.append(("ER n=400 p=0.01", nx.gnp_random_graph(400, 0.01, seed=3)))
    cases.append(("BA n=300 m=2", nx.barabasi_albert_graph(300, 2, seed=4)))
    cases.append(("WS n=300 k=4 p=.1", nx.watts_strogatz_graph(300, 4, 0.1, seed=5)))
    cases.append(("K_30", nx.complete_graph(30)))
    T = nx.random_labeled_tree(300, seed=6)
    cases.append(("tree n=300 (mu=0)", T.copy()))
    T2 = T.copy()
    ne = list(nx.non_edges(T2))
    rng.shuffle(ne)
    for e in ne[:3]:
        T2.add_edge(*e)
    cases.append(("tree+3 (mu=3)", T2))
    T3 = nx.random_labeled_tree(300, seed=8)
    ne = list(nx.non_edges(T3))
    rng.shuffle(ne)
    for e in ne[:15]:
        T3.add_edge(*e)
    cases.append(("tree+15 (mu=15)", T3))
    cases.append(("RGG n=300 r=.12", nx.random_geometric_graph(300, 0.12, seed=9)))
    # planted short cycle in a heavy grid
    Gp = nx.convert_node_labels_to_integers(nx.grid_2d_graph(20, 20))
    cases.append(("grid20 planted", Gp))
    # disconnected union
    cases.append(("3x ER n=100 p=.05",
                  nx.disjoint_union_all([nx.gnp_random_graph(100, 0.05, seed=s)
                                         for s in (11, 12, 13)])))

    for name, G in cases:
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        for u, v in G.edges():
            G[u][v]["weight"] = round(rng.uniform(0.1, 2.0), 6)
        if name == "grid20 planted":
            vs = [0, 1, 21, 20]
            for i in range(4):
                if G.has_edge(vs[i], vs[(i + 1) % 4]):
                    G[vs[i]][vs[(i + 1) % 4]]["weight"] = 0.01
        n, m = G.number_of_nodes(), G.number_of_edges()
        mu = m - n + nx.number_connected_components(G)
        order = list(G.nodes())
        rng.shuffle(order)
        g_all, st_all = alg1(G, order)
        S = transversal_endpoint(G)
        Svc = transversal_greedy_vc(G)
        g_S, st_S = alg1(G, list(S))
        g_pipe, st_pipe = run_pipeline(G)
        truth = None
        if n <= 60 or m <= 800:
            truth = oracle_mwc_edge(G)
        ok = "" if truth is None else ("" if max(abs(g_all - truth) if g_all < INF else 0,
                                                 abs(g_S - truth) if g_S < INF else 0,
                                                 abs(g_pipe - truth) if g_pipe < INF else 0) < 1e-7
                                       else "  <-- MISMATCH")
        r1 = st_S["settled"] / st_all["settled"] if st_all["settled"] else float('nan')
        r2 = st_pipe["settled"] / st_all["settled"] if st_all["settled"] else float('nan')
        print(f"{name:22s} {n:5d} {m:6d} {mu:6d} {len(S):5d} {len(S)/n:6.3f} "
              f"{len(Svc):5d} {len(Svc)/n:6.3f} {st_all['settled']:12d} "
              f"{st_S['settled']:10d} {r1:6.3f} {st_pipe['settled']:8d} {r2:6.3f}{ok}")
        assert g_all == g_S == g_pipe or (abs(g_all - g_S) < 1e-9 and abs(g_all - g_pipe) < 1e-9), \
            (name, g_all, g_S, g_pipe)


if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    bad = validate(trials=trials)
    study()
    print("\nRESULT:", "ZERO MISMATCHES" if not bad else f"{len(bad)} MISMATCHES")
