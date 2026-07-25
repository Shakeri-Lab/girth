"""Differential test: girth_snapshot implementations vs a brute-force oracle.

Oracle 1 (edge-removal): for each edge (u,v) with weight w, remove it and
compute d_{G-e}(u,v); the MWC length is min over e of d + w.  Exact for
simple undirected graphs with positive weights.

Oracle 2 (enumeration, tiny graphs only): enumerate every simple cycle with
networkx.simple_cycles and take the minimum total weight.  Used to sanity-check
Oracle 1.

Systems under test:
  A. shortest_cycle.sota_shortest_cycle              (Algorithm 1 as implemented)
  B. hybrid_mwc.hybrid_mwc_length (BMSSP-full + binary LCA)
  C. hybrid_mwc.hybrid_mwc_length (Euler LCA)
  D. sota with the `if p and ...` LCA guard repaired (isolates that defect)
"""
import os
import random
import signal
import sys
import traceback


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)


def guarded(fn, *a, **kw):
    """Run fn with a 5 s wall-clock guard (hybrid_mwc._reconstruct_cycle can
    loop forever when the predecessor map is not a forest)."""
    signal.alarm(5)
    try:
        return fn(*a, **kw)
    finally:
        signal.alarm(0)

HERE = os.path.dirname(os.path.abspath(__file__))
GIRTH = os.path.abspath(os.path.join(HERE, "..", "girth_snapshot"))
sys.path.insert(0, GIRTH)

import networkx as nx  # noqa: E402

import shortest_cycle as sc  # noqa: E402
import hybrid_mwc as hm  # noqa: E402
from bmssp_lite import BMSSPLiteStrategy  # noqa: E402
from bmssp_full import BMSSPFullStrategy  # noqa: E402


# ----------------------------------------------------------------- oracles
def oracle_edge_removal(G):
    best = float("inf")
    for u, v, data in list(G.edges(data=True)):
        w = data.get("weight", 1.0)
        G.remove_edge(u, v)
        try:
            d = nx.shortest_path_length(G, u, v, weight="weight")
            best = min(best, d + w)
        except nx.NetworkXNoPath:
            pass
        G.add_edge(u, v, **data)
    return None if best == float("inf") else best


def oracle_enumerate(G):
    best = float("inf")
    for cyc in nx.simple_cycles(G):
        if len(cyc) < 3:
            continue
        tot = sum(G[cyc[i]][cyc[(i + 1) % len(cyc)]]["weight"] for i in range(len(cyc)))
        best = min(best, tot)
    return None if best == float("inf") else best


# ------------------------------------------- variant D: repaired LCA guard
def sota_variant(G, fix_lca=False, prune=True):
    """sota_shortest_cycle with two independently switchable repairs:
       fix_lca:  `if p and ...` -> `if p is not None`, and accept
                 ancestor/descendant non-tree edges (p == u or p == v)
       prune:    keep / drop the `dist + 2*w_min >= gamma` discard sweep."""
    gamma = float("inf")
    active = set(G.nodes())
    wmin = min(d.get("weight", 1.0) for _, _, d in G.edges(data=True))
    nodes = list(G.nodes())
    for node in nodes:
        if node not in active:
            continue
        dist, preds, depth, _ = sc.dijkstra_base(G, node, gamma)
        lca_tree = sc.LCATree(preds, depth, nodes)
        for u, v, attr in G.edges(data=True):
            if dist.get(u, float("inf")) == float("inf"):
                continue
            if dist.get(v, float("inf")) == float("inf"):
                continue
            p = lca_tree.lca(u, v)
            if fix_lca:
                if preds.get(u) == v or preds.get(v) == u:
                    continue  # genuine tree edge -> no cycle
                if p is None:
                    continue
            else:
                if not (p and p != u and p != v):
                    continue
            cl = dist[u] + dist[v] + attr.get("weight", 1.0) - 2 * dist[p]
            if cl < gamma:
                gamma = cl
        if prune:
            for v in G.nodes():
                dv = dist.get(v, float("inf"))
                if v in active and dv != float("inf") and dv + 2 * wmin >= gamma:
                    active.discard(v)
    return None if gamma == float("inf") else gamma


# ------------------------------------------------------------- generators
def rand_graph(rng, n_lo=4, n_hi=9, weights="int"):
    n = rng.randint(n_lo, n_hi)
    p = rng.uniform(0.25, 0.7)
    G = nx.gnp_random_graph(n, p, seed=rng.randint(0, 10 ** 9))
    for u, v in G.edges():
        if weights == "int":
            G[u][v]["weight"] = float(rng.randint(1, 9))
        elif weights == "unit":
            G[u][v]["weight"] = 1.0
        else:
            G[u][v]["weight"] = round(rng.uniform(0.5, 5.0), 3)
    return G


def relabel_no_zero(G):
    """Same graph with labels shifted by 1 so that no vertex is named 0."""
    return nx.relabel_nodes(G, {v: v + 1 for v in G.nodes()}, copy=True)


def is_simple_cycle(G, edge_set, reported_len):
    """True iff edge_set is a genuine simple cycle of G whose weight == reported_len."""
    if not edge_set or len(edge_set) < 3:
        return False
    H = nx.Graph()
    tot = 0.0
    for e in edge_set:
        u, v = e
        if not G.has_edge(u, v):
            return False
        H.add_edge(u, v)
        tot += G[u][v]["weight"]
    if any(d != 2 for _, d in H.degree()):
        return False
    if nx.number_connected_components(H) != 1:
        return False
    return abs(tot - reported_len) < 1e-6


def edges_repr(G):
    return sorted((u, v, G[u][v]["weight"]) for u, v in G.edges())


def approx_eq(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    import math
    if not math.isfinite(a) or not math.isfinite(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main():
    rng = random.Random(20260725)
    N = 400
    counters = {k: 0 for k in
                ("sota", "sota_nozero", "sota_lcafix", "sota_noprune",
                 "sota_lcafix_noprune", "hybrid_bmssp_full", "hybrid_bmssp_lite",
                 "hybrid_dijkstra", "hybrid_euler_full", "hybrid_euler_lite")}
    errors = {k: 0 for k in counters}
    minimal = {}
    oracle_disagree = 0

    for trial in range(N):
        mode = ["int", "unit", "float"][trial % 3]
        G = rand_graph(rng, weights=mode)
        if G.number_of_edges() == 0:
            continue
        truth = oracle_edge_removal(G)
        if G.number_of_nodes() <= 8:
            enum = oracle_enumerate(G)
            if not approx_eq(truth, enum):
                oracle_disagree += 1

        Gz = relabel_no_zero(G)

        def record(name, got):
            if not approx_eq(got, truth):
                counters[name] += 1
                if name not in minimal or G.number_of_edges() < minimal[name][0]:
                    minimal[name] = (G.number_of_edges(), edges_repr(G), truth, got)

        for name, fn, graph in (
            ("sota", lambda g: sc.sota_shortest_cycle(g), G),
            ("sota_nozero", lambda g: sc.sota_shortest_cycle(g), Gz),
            ("sota_lcafix", lambda g: sota_variant(g, fix_lca=True, prune=True), G),
            ("sota_noprune", lambda g: sota_variant(g, fix_lca=False, prune=False), G),
            ("sota_lcafix_noprune", lambda g: sota_variant(g, fix_lca=True, prune=False), G),
            # NOTE: pinned explicitly.  Before the quarantine commit, hybrid_mwc_length()
            # selected BMSSPFullStrategy() implicitly; it no longer does, so relying on the
            # default here would silently measure the baseline Dijkstra path instead.
            ("hybrid_bmssp_full",
             lambda g: hm.HybridMWC(g, sssp=BMSSPFullStrategy(),
                                    lca=hm.BinaryLCAStrategy()).minimum_weight_cycle(), G),
            ("hybrid_bmssp_lite",
             lambda g: hm.HybridMWC(g, sssp=BMSSPLiteStrategy(),
                                    lca=hm.BinaryLCAStrategy()).minimum_weight_cycle(), G),
            ("hybrid_dijkstra",
             lambda g: hm.HybridMWC(g, sssp=hm.DijkstraStrategy(),
                                    lca=hm.BinaryLCAStrategy()).minimum_weight_cycle(), G),
            ("hybrid_euler_full",
             lambda g: hm.HybridMWC(g, sssp=BMSSPFullStrategy(),
                                    lca=hm.EulerLCAStrategy()).minimum_weight_cycle(), G),
            ("hybrid_euler_lite",
             lambda g: hm.HybridMWC(g, sssp=BMSSPLiteStrategy(),
                                    lca=hm.EulerLCAStrategy()).minimum_weight_cycle(), G),
        ):
            try:
                record(name, guarded(fn, graph.copy()))
            except Exception:
                errors[name] += 1
                if name + ":exc" not in minimal:
                    minimal[name + ":exc"] = (G.number_of_edges(), edges_repr(G), truth,
                                              traceback.format_exc().strip().splitlines()[-1])

    # ---- return_edges validation (this is what loop_modulus adds as a QP row)
    rng2 = random.Random(99)
    bad_edges = {"full": 0, "lite": 0}
    tot_edges = {"full": 0, "lite": 0}
    hang_edges = {"full": 0, "lite": 0}
    bad_example = {}
    for _ in range(100):
        G = rand_graph(rng2, weights="int")
        if G.number_of_edges() == 0:
            continue
        for tag, strat in (("full", BMSSPFullStrategy()), ("lite", BMSSPLiteStrategy())):
            try:
                res = guarded(hm.HybridMWC(G.copy(), sssp=strat,
                                           lca=hm.BinaryLCAStrategy()
                                           ).minimum_weight_cycle, return_edges=True)
            except _Timeout:
                hang_edges[tag] += 1
                continue
            except Exception:
                continue
            if not res:
                continue
            edges, glen = res
            if edges is None:
                continue
            tot_edges[tag] += 1
            ok = is_simple_cycle(G, edges, glen)
            if not ok:
                bad_edges[tag] += 1
                if tag not in bad_example:
                    bad_example[tag] = (edges_repr(G), sorted(edges), glen,
                                        oracle_edge_removal(G.copy()))

    lines = []
    lines.append(f"random graphs tested: {N}  (n in [4,9], gnp, int/unit/float weights)")
    lines.append(f"oracle cross-check disagreements (edge-removal vs enumeration): {oracle_disagree}")
    lines.append("")
    lines.append("mismatches vs brute-force oracle:")
    for k in counters:
        lines.append(f"  {k:16s} wrong={counters[k]:4d}/{N}  ({100.0*counters[k]/N:5.1f}%)   exceptions={errors[k]}")
    lines.append("")
    lines.append("return_edges=True: is the returned edge set a genuine simple cycle of G")
    lines.append("with weight equal to the reported length?  (this set becomes a QP row in")
    lines.append("loop_modulus core.py, unvalidated)")
    for tag in ("full", "lite"):
        n_t = tot_edges[tag]
        lines.append(f"  BMSSP-{tag:5s}: invalid={bad_edges[tag]}/{n_t}"
                     f"  ({100.0*bad_edges[tag]/max(1,n_t):5.1f}%)"
                     f"   infinite-loop hangs (>5s) = {hang_edges[tag]}")
    for tag, (ge, ce, gl, tr) in sorted(bad_example.items()):
        lines.append(f"  --- invalid-cycle example, BMSSP-{tag} ---")
        lines.append(f"      G edges: {ge}")
        lines.append(f"      returned edge set: {ce}   reported length={gl}  oracle={tr}")
    lines.append("")
    for k, v in sorted(minimal.items()):
        ne, ed, truth, got = v
        lines.append(f"--- minimal failing example for {k} ({ne} edges) ---")
        lines.append(f"    edges (u,v,w): {ed}")
        lines.append(f"    oracle = {truth}   got = {got}")
    out = "\n".join(lines)
    print(out)

    # ---- targeted probes ------------------------------------------------
    probes = []

    # P1: node 0 as LCA.  Path 1-0-2 plus edge 1-2 -> triangle through node 0.
    Gp = nx.Graph()
    Gp.add_edge(0, 1, weight=1.0)
    Gp.add_edge(0, 2, weight=1.0)
    Gp.add_edge(1, 2, weight=5.0)
    Gp.add_edge(3, 4, weight=1.0)  # keep node 0 from being the only structure
    probes.append(("P1a node-0-falsy triangle (sota)", Gp, oracle_edge_removal(Gp.copy()),
                   sc.sota_shortest_cycle(Gp.copy())))
    Gp_shift = relabel_no_zero(Gp)
    probes.append(("P1b same graph, labels+1 (isolates ancestor/descendant skip)",
                   Gp_shift, oracle_edge_removal(Gp_shift.copy()),
                   sc.sota_shortest_cycle(Gp_shift.copy())))
    probes.append(("P1c same graph, LCA guard repaired", Gp,
                   oracle_edge_removal(Gp.copy()),
                   sota_variant(Gp.copy(), fix_lca=True, prune=True)))

    # P2: ancestor-descendant non-tree edge.  Path 0-1-2-3 + chord 0-3.
    Gp2 = nx.Graph()
    for a, b, w in [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (1, 4, 1.0)]:
        Gp2.add_edge(a, b, weight=w)
    probes.append(("P2 ancestor-descendant chord (square)", Gp2,
                   oracle_edge_removal(Gp2.copy()), sc.sota_shortest_cycle(Gp2.copy())))

    # P3: hybrid_mwc `gamma <= 4*min_edge_w` early break.
    # square 1-2-3-4 (len 4, near the first root) + far-away triangle 8-9-10
    # (len 3).  With a bound the first root cannot see the triangle, and the
    # `gamma <= 4*min_edge_w` break then stops the outer loop for good.
    Gp3 = nx.Graph()
    for a, b in [(1, 2), (2, 3), (3, 4), (4, 1)]:
        Gp3.add_edge(a, b, weight=1.0)
    for a, b in [(8, 9), (9, 10), (10, 8)]:
        Gp3.add_edge(a, b, weight=1.0)
    for a, b in [(3, 5), (5, 6), (6, 7), (7, 8)]:
        Gp3.add_edge(a, b, weight=1.0)
    probes.append(("P3 hybrid `gamma <= 4*min_edge_w` early break (bound=8 -> radius 4)",
                   Gp3, oracle_edge_removal(Gp3.copy()),
                   hm.HybridMWC(Gp3.copy(), sssp=BMSSPLiteStrategy(),
                                lca=hm.BinaryLCAStrategy()).minimum_weight_cycle(bound=8.0)))

    # P3b: BMSSP-full SSSP returns wrong distances (recursion resets dist to 0)
    Gp3b = nx.Graph()
    for a, b in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]:
        Gp3b.add_edge(a, b, weight=1.0)
    df, pf, _ = BMSSPFullStrategy().run(Gp3b, 1, float("inf"))
    dl, pl, _ = BMSSPLiteStrategy().run(Gp3b, 1, float("inf"))
    probes.append(("P3b BMSSP-full dist from src 1 on C6 (true 0,1,2,3,2,1)",
                   Gp3b, str(dict(sorted(dl.items()))), str(dict(sorted(df.items())))))

    # P4: bound semantics.  bound = gamma (true) should not lose the cycle.
    Gp4 = nx.Graph()
    for a, b in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]:
        Gp4.add_edge(a, b, weight=1.0)
    truth4 = 6.0

    def hyb(g, bound):
        return hm.HybridMWC(g, sssp=BMSSPLiteStrategy(),
                            lca=hm.BinaryLCAStrategy()).minimum_weight_cycle(bound=bound)

    probes.append(("P4 bound=None (C6, w=1)", Gp4, truth4, hyb(Gp4.copy(), None)))
    probes.append(("P4 bound=gamma(6): radius becomes 3 = gamma/2 (correct)",
                   Gp4, truth4, hyb(Gp4.copy(), 6.0)))
    probes.append(("P4 bound=gamma/2(3) <- dijkstra.py:54; radius becomes 1.5",
                   Gp4, truth4, hyb(Gp4.copy(), 3.0)))

    # P5: Euler LCA when nodes[0] is unreached / far from the Dijkstra root
    Gp5 = nx.Graph()
    for a, b in [(1, 2), (2, 3), (3, 1)]:
        Gp5.add_edge(a, b, weight=1.0)
    Gp5.add_edge(10, 11, weight=1.0)  # separate component; nodes[0] may be in it
    try:
        r5 = hm.HybridMWC(Gp5.copy(), sssp=BMSSPLiteStrategy(),
                          lca=hm.EulerLCAStrategy()).minimum_weight_cycle()
    except Exception as e:
        r5 = f"EXCEPTION {type(e).__name__}: {e}"
    probes.append(("P5 EulerLCA on disconnected graph", Gp5, 3.0, r5))

    # P6: no cycle is returned/validated by sota_shortest_cycle -- only a scalar
    probes.append(("P6 sota return type (cycle reconstruction absent)", Gp2, "list-of-nodes",
                   type(sc.sota_shortest_cycle(Gp2.copy())).__name__))

    # P7: LCATree binary-lifting depth is hardcoded to 20 -> a shortest-path
    #     tree deeper than 2^20-1 hops silently returns a wrong ancestor.
    probes.append(("P7 LCATree.log_max_depth", None, "ceil(log2(n)) (data dependent)",
                   sc.LCATree({}, {0: 0}, [0]).log_max_depth))

    # P6: LCA binary-lifting depth 20 -> path of >2^20 hops is infeasible, but
    #     depth is *hop* depth; check a 25-node path + chord still works, and
    #     report the structural limit rather than testing 1e6 nodes.
    lines2 = ["", "targeted probes:"]
    for name, g, truth, got in probes:
        if isinstance(truth, str) or isinstance(got, str):
            ok = "OK " if truth == got else "BAD"
        else:
            ok = "OK " if approx_eq(got, truth) else "BAD"
        lines2.append(f"  [{ok}] {name}:\n         expected={truth}\n         got     ={got}")
    print("\n".join(lines2))

    with open(os.path.join(HERE, "diff_results.txt"), "w") as f:
        f.write(out + "\n" + "\n".join(lines2) + "\n")


if __name__ == "__main__":
    main()
