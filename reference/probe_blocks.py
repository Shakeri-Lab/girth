"""When does the biconnected block decomposition pay for itself?

Measured: adding the block split costs time on five of six synthetic families
and on every large real road network, but helps on near-tree graphs.  The
hypothesis is that it pays exactly when the 2-core is a small fraction of the
graph -- a tree-like graph shatters into many small blocks, whereas a road
network's 2-core is essentially one giant block, so Hopcroft-Tarjan is paid for
nothing.

If that holds, `n_2core / n` is a legitimate predictor AND it is already known
by the time the decision is needed: the 2-core peel happens first, so the
statistic is free.
"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import networkx as nx  # noqa: E402
from mwc import biconnected_components, mwc_transversal, two_core  # noqa: E402


def to_adj(G):
    a = {n: {} for n in G}
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        a[u][v] = w
        a[v][u] = w
    return a


def load_edges(path):
    adj = {}
    with open(path) as f:
        for line in f:
            a, b, w = line.split()[:3]
            a, b, w = int(a), int(b), float(w)
            if a == b:
                continue
            adj.setdefault(a, {})[b] = w
            adj.setdefault(b, {})[a] = w
    return adj


def best(fn, reps=3):
    t = float("inf")
    v = None
    for _ in range(reps):
        t0 = time.perf_counter()
        v = fn()
        t = min(t, time.perf_counter() - t0)
    return t, v


rng = random.Random(20260725)


def synth():
    n = 1600
    s = rng.randrange(10 ** 9)
    T = nx.random_labeled_tree(n, seed=s)
    G = nx.Graph(T)
    nodes = list(G)
    for _ in range(3):
        u, v = rng.sample(nodes, 2)
        G.add_edge(u, v)
    out = [("near_tree", G),
           ("sparse_er", nx.gnp_random_graph(n, 2.5 / n, seed=s)),
           ("grid", nx.convert_node_labels_to_integers(nx.grid_2d_graph(40, 40))),
           ("small_world", nx.watts_strogatz_graph(n, 4, 0.1, seed=s)),
           ("geometric", nx.random_geometric_graph(n, (2.2 / n) ** 0.5 * 1.6, seed=s))]
    for k, G in out:
        for u, v in G.edges():
            G[u][v]["weight"] = round(rng.uniform(0.5, 5.0), 3)
        yield k, to_adj(G)


DATA = os.environ.get("REALNETS", os.path.join(HERE, "realnets"))
def real():
    for name in ["rome99-road", "uspowergrid-synth", "chicago-regional-road",
                 "osm-portland-drive", "sydney-road"]:
        yield name, load_edges(os.path.join(DATA, name + ".edges"))


print(f"{'graph':<23}{'n':>7}{'n_2core/n':>11}{'#blocks':>9}{'big/2core':>11}"
      f"{'blocks_on':>11}{'blocks_off':>11}{'verdict':>10}")
for label, gen in (("synthetic", synth()), ("real", real())):
    print(f"-- {label}")
    for name, adj in gen:
        n = len(adj)
        core = two_core(adj)
        nc = len(core)
        blks = biconnected_components(core) if core else []
        sizes = [len({x for e in b for x in e[:2]}) for b in blks]
        big = max(sizes) if sizes else 0
        t_on, r_on = best(lambda: mwc_transversal(adj, use_blocks=True,
                                                  certify=False, collect_stats=False))
        t_off, r_off = best(lambda: mwc_transversal(adj, use_blocks=False,
                                                    certify=False, collect_stats=False))
        assert abs(r_on.length - r_off.length) < 1e-9, name
        verdict = "blocks" if t_on < t_off else "no blocks"
        print(f"{name:<23}{n:>7,}{nc/n:>11.3f}{len(blks):>9,}"
              f"{(big/nc if nc else 0):>11.3f}{t_on*1e3:>11.1f}{t_off*1e3:>11.1f}"
              f"{verdict:>10}")
print("\nms, best of 3.  'big/2core' is the largest block's share of the 2-core.")
