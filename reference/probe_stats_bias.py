"""Is the published speedup contaminated by the instrumentation?

bench_timing.py (which produced Table 5) times both configurations with
collect_stats=True.  Stats collection appends one dict PER ROOT, so the
all-roots configuration pays n of them and the transversal configuration only
|S| -- an overhead that scales with exactly the quantity the table is claiming
to reduce.  If that overhead is material, part of the reported speedup is
measuring the instrumentation rather than the algorithm.

This probe times every family both ways.
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))  #
)
import networkx as nx  # noqa: E402
from mwc import mwc, mwc_transversal  # noqa: E402


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
    s = rng.randrange(10 ** 9)
    out = []
    T = nx.random_labeled_tree(n, seed=s)
    G = nx.Graph(T)
    nodes = list(G)
    for _ in range(3):
        u, v = rng.sample(nodes, 2)
        G.add_edge(u, v)
    out.append(("near_tree", G))
    out.append(("sparse_er", nx.gnp_random_graph(n, 2.5 / n, seed=s)))
    out.append(("grid", nx.convert_node_labels_to_integers(
        nx.grid_2d_graph(int(n ** 0.5), int(n ** 0.5)))))
    out.append(("small_world", nx.watts_strogatz_graph(n, 4, 0.1, seed=s)))
    out.append(("geometric", nx.random_geometric_graph(n, (2.2 / n) ** 0.5 * 1.6, seed=s)))
    return [(k, weighted(g, rng)) for k, g in out]


def best(fn, reps=5):
    t = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - t0)
    return t


rng = random.Random(20260725)
print(f"{'family':<13}{'stats=True':>22}{'stats=False':>22}")
print(f"{'':<13}{'all':>8}{'tv':>7}{'speedup':>7}{'all':>8}{'tv':>7}{'speedup':>7}"
      f"{'  inflation':>12}")
for name, G in families(1600, rng):
    adj = to_adj(G)
    a_on = best(lambda: mwc(adj, certify=False, collect_stats=True))
    t_on = best(lambda: mwc_transversal(adj, certify=False, collect_stats=True))
    a_off = best(lambda: mwc(adj, certify=False, collect_stats=False))
    t_off = best(lambda: mwc_transversal(adj, certify=False, collect_stats=False))
    s_on, s_off = a_on / t_on, a_off / t_off
    print(f"{name:<13}{a_on*1e3:>8.1f}{t_on*1e3:>7.1f}{s_on:>7.2f}"
          f"{a_off*1e3:>8.1f}{t_off*1e3:>7.1f}{s_off:>7.2f}{s_on/s_off:>12.3f}")
print("\nms, best of 5, n=1600.  'inflation' is speedup(stats on)/speedup(stats off):")
print(">1 means the instrumentation flatters the transversal configuration.")
