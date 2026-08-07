#!/usr/bin/env python3
"""
validate.py -- reload every <name>.edges with networkx and assert the contract:
connected, simple, undirected, all weights > 0, ids exactly 0..n-1, and n/m/mu
matching <name>.json and manifest.json.

    uv run --python 3.12 --with networkx,numpy python validate.py
"""
import json
import math
import os
import sys

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "manifest.json")) as fh:
    manifest = json.load(fh)

fails = []
rows = []
for entry in manifest["networks"]:
    name = entry["name"]
    with open(os.path.join(HERE, name + ".json")) as fh:
        meta = json.load(fh)
    path = os.path.join(HERE, name + ".edges")

    seen = set()
    G = nx.Graph()
    nlines = 0
    minw = math.inf
    maxw = -math.inf
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            nlines += 1
            parts = line.split()
            assert len(parts) == 3, f"{name}:{lineno} expected 3 fields, got {parts}"
            u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
            assert u != v, f"{name}:{lineno} self-loop {u}"
            assert w > 0, f"{name}:{lineno} non-positive weight {w}"
            key = (min(u, v), max(u, v))
            assert key not in seen, f"{name}:{lineno} parallel/repeated edge {key}"
            seen.add(key)
            minw, maxw = min(minw, w), max(maxw, w)
            G.add_edge(u, v, weight=w)

    def check(cond, msg):
        if not cond:
            fails.append(f"{name}: {msg}")
        return cond

    n, m = G.number_of_nodes(), G.number_of_edges()
    check(not G.is_directed(), "graph is directed")
    check(not G.is_multigraph(), "graph is a multigraph")
    check(nx.is_connected(G), "graph is NOT connected")
    check(nx.number_of_selfloops(G) == 0, "self-loops present")
    check(m == nlines, f"edge count {m} != {nlines} lines (duplicates?)")
    check(set(G.nodes()) == set(range(n)), "node ids are not exactly 0..n-1")
    check(all(d["weight"] > 0 for _, _, d in G.edges(data=True)), "non-positive weight")
    check(n == meta["n"], f"n mismatch: file {n} vs json {meta['n']}")
    check(m == meta["m"], f"m mismatch: file {m} vs json {meta['m']}")
    check(meta["mu"] == m - n + 1, f"mu mismatch: json {meta['mu']} vs m-n+1 {m-n+1}")
    check(n == entry["n"] and m == entry["m"], "manifest n/m disagree with file")
    for k in ("name", "n", "m", "mu", "source_url", "license_or_terms",
              "weight_semantics", "weights_native", "synthetic_weighting",
              "preprocessing", "date_downloaded"):
        check(k in meta, f"metadata key '{k}' missing")
    check(meta["weights_native"] is True or meta["synthetic_weighting"] is not None
          or meta["weights_native"] is False,
          "weights_native must be a bool")
    if meta["weights_native"] is False and meta["synthetic_weighting"] is None:
        # allowed, but must be a derived-from-real-data weighting -> flagged, not failed
        pass

    deg = [d for _, d in G.degree()]
    rows.append(dict(name=name, n=n, m=m, mu=m - n + 1, mu_over_n=(m - n + 1) / n,
                     dens=2 * m / (n * (n - 1)), meandeg=sum(deg) / n, maxdeg=max(deg),
                     wmin=minw, wmax=maxw,
                     native=meta["weights_native"],
                     synth=meta["synthetic_weighting"] is not None))

hdr = (f"{'network':24s} {'n':>7s} {'m':>7s} {'mu':>7s} {'mu/n':>7s} "
       f"{'<k>':>6s} {'kmax':>5s} {'w_min':>12s} {'w_max':>12s}  weights")
print(hdr)
print("-" * len(hdr))
for r in rows:
    kind = "SYNTHETIC" if r["synth"] else ("native" if r["native"] else "derived")
    print(f"{r['name']:24s} {r['n']:7d} {r['m']:7d} {r['mu']:7d} {r['mu_over_n']:7.3f} "
          f"{r['meandeg']:6.2f} {r['maxdeg']:5d} {r['wmin']:12.6g} {r['wmax']:12.6g}  {kind}")

print()
if fails:
    print(f"VALIDATION FAILED ({len(fails)} problems):")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print(f"VALIDATION PASSED: {len(rows)} networks; all connected, simple, undirected, "
      f"weights > 0, ids 0..n-1, n/m/mu consistent with .json and manifest.json.")
