"""Produce the numbers quoted in RESULTS.md.

    <venv>/bin/python report.py
"""

from __future__ import annotations

import collections
import itertools
import random
import time

import gen
from mwc import INF, cycle_weight, is_simple_cycle, kappa_of, mwc, mwc_oracle, mwc_transversal


def hdr(t):
    print("\n" + t)
    print("-" * len(t))


def main():
    rng = random.Random(20250725)

    # ------------------------------------------------------------------ 1
    hdr("A. exact-mode differential test, by family")
    families = collections.Counter()
    mism = collections.Counter()
    graphs = gen.family_suite(rng, count=1200)
    graphs += [("multiscale", gen.multiscale(rng.randint(5, 12), rng)) for _ in range(600)]
    for name, G in graphs:
        g_star, _ = mwc_oracle(G)
        r = mwc(G)
        rt = mwc_transversal(G)
        families[name] += 1
        if not (abs(r.length - g_star) <= 1e-9 * max(1.0, abs(g_star)) or r.length == g_star):
            mism[name + "/mwc"] += 1
        if not (abs(rt.length - g_star) <= 1e-9 * max(1.0, abs(g_star)) or rt.length == g_star):
            mism[name + "/transversal"] += 1
    for k in sorted(families):
        print(f"  {k:16s} {families[k]:5d} graphs   mismatches: "
              f"mwc={mism[k+'/mwc']}  transversal={mism[k+'/transversal']}")
    print(f"  TOTAL            {sum(families.values()):5d} graphs   "
          f"total mismatches = {sum(mism.values())}")

    # ------------------------------------------------------------------ 2
    hdr("B. graph atlas (n <= 7), exhaustive")
    n_unit = n_w = mism_a = 0
    for name, g in gen.atlas_graphs(max_nodes=7):
        for wf, tag in ((lambda: 1.0, "unit"),
                        (gen.weight_fn("int", rng), "int"),
                        (gen.weight_fn("cont", rng), "cont")):
            G = gen.from_nx(g, wf=wf)
            gs, _ = mwc_oracle(G)
            for r in (mwc(G), mwc(G, root_order=list(reversed(list(G)))), mwc_transversal(G)):
                if not (r.length == gs or abs(r.length - gs) <= 1e-9 * max(1.0, gs)):
                    mism_a += 1
            if tag == "unit":
                n_unit += 1
            else:
                n_w += 1
    print(f"  atlas graphs with 3..7 vertices : {n_unit}")
    print(f"  weighted instances              : {n_w}")
    print(f"  runs per instance               : 3 (mwc, mwc reversed order, mwc_transversal)")
    print(f"  mismatches vs oracle            : {mism_a}")

    # ------------------------------------------------------------------ 3
    hdr("C. approximation ratio vs kappa")
    PAIRS = [(0.0, 0.0), (0.05, 0.0), (0.1, 0.0), (0.2, 0.0), (0.25, 0.05),
             (0.3, 0.1), (0.4, 0.1), (0.45, 0.2), (0.49, 0.0), (0.6, 0.2),
             (0.75, 0.4), (1.0, 0.6)]
    rows = []
    pool = [g for _, g in gen.family_suite(rng, count=600)]
    pool += [gen.multiscale(rng.randint(5, 11), rng) for _ in range(1400)]
    oracle = [mwc_oracle(G)[0] for G in pool]
    for a, b in PAIRS:
        k = kappa_of(a, b)
        worst = 1.0
        dels = 0
        runs = 0
        errs = 0
        for G, gs in zip(pool, oracle):
            order = list(G)
            perm = order[:]
            rng.shuffle(perm)
            for o in (order, list(reversed(order)), perm):
                r = mwc(G, alpha=a, beta=b, root_order=o)
                runs += 1
                dels += r.stats["total_deletions"]
                if gs == INF:
                    assert r.length == INF
                    continue
                assert r.length >= gs - 1e-9
                assert r.length <= k * gs + 1e-9
                if r.length > gs + 1e-9:
                    errs += 1
                worst = max(worst, r.length / gs)
        rows.append((a, b, k, runs, dels, errs, worst))
        print(f"  alpha={a:<5} beta={b:<5} kappa={k:8.3f}  runs={runs:6d} "
              f"deletions={dels:5d} inexact={errs:4d} worst ratio={worst:.6f}")
    print(f"  graphs in pool = {len(pool)}")

    # ----------------------------------------------------------------- 3b
    hdr("C2. randomized adversarial search for approximation error")
    rng2 = random.Random(2024)
    PAIRS2 = [(0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.45, 0.0), (0.49, 0.0),
              (0.6, 0.15), (0.8, 0.35), (0.95, 0.48), (2.0, 1.55), (3.0, 2.6)]
    runs = dels = errs = viol = 0
    worst = 1.0
    worst_info = None
    for _ in range(60000):
        n = rng2.randint(5, 10)
        G = gen.multiscale(n, rng2)
        gs, _ = mwc_oracle(G)
        if gs == INF:
            continue
        a, b = rng2.choice(PAIRS2)
        k = kappa_of(a, b)
        order = list(G)
        rng2.shuffle(order)
        r = mwc(G, alpha=a, beta=b, root_order=order)
        runs += 1
        dels += r.stats["total_deletions"]
        if r.length > k * gs + 1e-9:
            viol += 1
        if r.length > gs + 1e-9:
            errs += 1
        if r.length / gs > worst:
            worst = r.length / gs
            worst_info = (a, b, k, r.length, gs)
    print(f"  runs                      : {runs}")
    print(f"  vertex deletions          : {dels}")
    print(f"  runs with ghat > gamma*   : {errs}")
    print(f"  kappa-bound violations    : {viol}")
    print(f"  worst observed ghat/gamma*: {worst:.6f}  {worst_info}")

    # ------------------------------------------------------------------ 4
    hdr("D. transversal reductions (roots run / settled vertices)")
    print(f"  {'family':16s} {'graphs':>6s} {'roots_full':>10s} {'roots_tv':>9s} "
          f"{'ratio':>6s} {'settled_full':>13s} {'settled_tv':>11s} {'ratio':>6s}")
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    suite = gen.family_suite(rng, count=1200)
    suite += [("grid8", gen.grid(8, 8, rng, "int")) for _ in range(10)]
    suite += [("er40", gen.erdos_renyi(40, 0.12, rng, "int")) for _ in range(20)]
    suite += [("geo40", gen.geometric(40, 0.30, rng, "cont")) for _ in range(20)]
    for name, G in suite:
        full = mwc(G)
        tv = mwc_transversal(G)
        assert tv.length == full.length or abs(tv.length - full.length) <= 1e-9 * max(1.0, full.length)
        a = agg[name]
        a[0] += 1
        a[1] += full.stats["roots_run"]
        a[2] += tv.stats["roots_run"]
        a[3] += full.stats["total_settled"]
        a[4] += tv.stats["total_settled"]
    tot = [0, 0, 0, 0, 0]
    for name in sorted(agg):
        c, rf, rt, sf, st = agg[name]
        for i, v in enumerate((c, rf, rt, sf, st)):
            tot[i] += v
        print(f"  {name:16s} {c:6d} {rf:10d} {rt:9d} {rt / max(1, rf):6.3f} "
              f"{sf:13d} {st:11d} {st / max(1, sf):6.3f}")
    c, rf, rt, sf, st = tot
    print(f"  {'ALL':16s} {c:6d} {rf:10d} {rt:9d} {rt / max(1, rf):6.3f} "
          f"{sf:13d} {st:11d} {st / max(1, sf):6.3f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[report generated in {time.time() - t0:.1f}s]")
