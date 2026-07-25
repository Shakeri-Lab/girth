"""Wall-clock benchmark for the exact MWC algorithm.

Compares, on identical inputs and with the same graph representation, heap and
cycle-reconstruction code:

  oracle      edge-removal baseline, gamma* = min_e [w(e) + d_{G-e}(u,v)]      (m searches)
  allroots    Algorithm 1, exact mode, all n roots, radius-gamma/2 truncation  (n searches)
  transversal Algorithm 1 + 2-core + blocks + gamma_0 seed + transversal roots (|S| searches)

`certify=False` disables the reconstruct-and-re-sum check on every candidate cycle; it is
on by default in the artifact for auditability and is pure overhead once trusted, so we
report it separately rather than folding it into the headline numbers.

Every method's answer is cross-checked; a disagreement aborts the run.
"""
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import networkx as nx  # noqa: E402
from mwc import mwc, mwc_transversal, mwc_oracle  # noqa: E402


def to_adj(G):
    a = {n: {} for n in G}
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        a[u][v] = w
        a[v][u] = w
    return a


def weighted(G, rng):
    for u, v in G.edges():
        G[u][v]["weight"] = round(rng.uniform(0.5, 5.0), 3)
    return G


def families(n, rng):
    """Ordered roughly by increasing mu/n."""
    s = rng.randrange(10 ** 9)
    out = []
    # near-tree: spanning tree plus a handful of extra edges -> mu = O(1)
    T = nx.random_labeled_tree(n, seed=s) if hasattr(nx, "random_labeled_tree") else nx.random_tree(n, seed=s)
    G = nx.Graph(T)
    nodes = list(G)
    for _ in range(3):
        u, v = rng.sample(nodes, 2)
        G.add_edge(u, v)
    out.append(("near_tree", G))
    out.append(("sparse_er", nx.gnp_random_graph(n, 2.5 / n, seed=s)))
    out.append(("grid", nx.convert_node_labels_to_integers(
        nx.grid_2d_graph(int(n ** 0.5), int(n ** 0.5)))))
    out.append(("geometric", nx.random_geometric_graph(n, (2.2 / n) ** 0.5 * 1.6, seed=s)))
    out.append(("small_world", nx.watts_strogatz_graph(n, 4, 0.1, seed=s)))
    out.append(("dense_er", nx.gnp_random_graph(n, 0.15, seed=s)))
    return [(name, weighted(G, rng)) for name, G in out if G.number_of_edges() > 0]


def timeit(fn, repeats=int(os.environ.get('REPEATS','5'))):
    best = math.inf
    val = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        val = fn()
        best = min(best, time.perf_counter() - t0)
    return best, val


def girth_of(res):
    return res if isinstance(res, float) else res.length


def main():
    rng = random.Random(20260725)
    rows = []
    sizes = [int(x) for x in os.environ.get("SIZES", "64,144,256").split(",")]
    for n in sizes:
        for name, G in families(n, rng):
            adj = to_adj(G)
            m = G.number_of_edges()
            nn = G.number_of_nodes()
            mu = m - nn + nx.number_connected_components(G)
            rec = dict(family=name, n=nn, m=m, mu=mu, mu_over_n=round(mu / nn, 3))

            # oracle is O(m * (m + n log n)); skip it when it would dominate the job
            if m <= int(os.environ.get("ORACLE_MAX_M", "900")):
                t, (g_or, _) = timeit(lambda: mwc_oracle(adj), 1)
                rec["t_oracle"] = t
                ref = g_or
            else:
                rec["t_oracle"] = None
                ref = None

            t, r_all = timeit(lambda: mwc(adj, certify=False, collect_stats=True))
            rec["t_allroots"] = t
            rec["settled_allroots"] = r_all.stats["total_settled"]
            rec["roots_allroots"] = r_all.stats["roots_run"]

            t, r_tv = timeit(lambda: mwc_transversal(adj, certify=False, collect_stats=True))
            rec["t_transversal"] = t
            rec["settled_transversal"] = r_tv.stats["total_settled"]
            rec["roots_transversal"] = r_tv.stats["roots_run"]

            t, _ = timeit(lambda: mwc(adj, certify=True, collect_stats=False))
            rec["t_allroots_certified"] = t

            # cross-check
            vals = [girth_of(r_all), girth_of(r_tv)] + ([ref] if ref is not None else [])
            vals = [v for v in vals if v is not None]
            if vals and not all(abs(v - vals[0]) <= 1e-9 or (v == math.inf and vals[0] == math.inf)
                                for v in vals):
                print("DISAGREEMENT", name, nn, vals, flush=True)
                sys.exit(1)
            rec["girth"] = None if not vals or vals[0] == math.inf else round(vals[0], 6)
            rows.append(rec)
            print(json.dumps(rec), flush=True)

    print("\n" + "=" * 112)
    hdr = (f"{'family':<13}{'n':>6}{'m':>7}{'mu/n':>8}{'roots':>7}{'roots_tv':>9}"
           f"{'root_cut':>10}{'allroots_s':>12}{'transv_s':>11}{'speedup':>10}{'oracle_s':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        rc = f"{1 - r['roots_transversal'] / r['roots_allroots']:.0%}"
        sp = f"{r['t_allroots'] / r['t_transversal']:.2f}x"
        oc = f"{r['t_oracle']:.2f}" if r["t_oracle"] else "-"
        print(f"{r['family']:<13}{r['n']:>6}{r['m']:>7}{r['mu_over_n']:>8}"
              f"{r['roots_allroots']:>7}{r['roots_transversal']:>9}{rc:>10}"
              f"{r['t_allroots']:>12.4f}{r['t_transversal']:>11.4f}{sp:>10}{oc:>11}")

    print("\ncertification overhead (reconstruct + re-sum every candidate cycle):")
    for r in rows:
        print(f"  {r['family']:<13} n={r['n']:<6} "
              f"{r['t_allroots_certified'] / r['t_allroots']:.2f}x")

    with open("bench_timing_results.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
