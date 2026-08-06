"""Experiments campaign for the MWC manuscript (2026-08).

Subcommands, each writing one JSON file into --out:

  timing     Table 5 rerun on the tagged benchmark families + the adaptive
             variant (A7) as an extra column.
  ablation   The A0-A7 ladder: what each acceleration actually buys.
  theta      Crossover sweep: speedup of transversal-vs-allroots as a function
             of mu/n, used to FIT the adaptive threshold rather than assume it.
  frontier   Approximate mode on the multiscale family: realized ratio and
             work saved as (alpha,beta) varies -- the deferred speed-accuracy
             frontier.
  argmin     argmin (heap extract-min) counts on d x d grids for Algorithm 1
             versus the edge-removal oracle, with an explicit counting
             protocol.  Regenerates the figure panel withdrawn in the
             2026-08-06 revision.
  real       Real-world networks from a staged directory of .edges files.

Every subcommand cross-checks exactness: each measured configuration must
return the same girth as a reference computation (the edge-removal oracle
where affordable, otherwise the unaccelerated A1 all-roots run).  A
disagreement aborts the task with a non-zero exit code -- a wrong number must
never reach a table.

Provenance: each JSON records the git SHA, hostname, python version, SLURM ids
when present, and the exact parameters, per the standing rule that no number
enters a durable artifact without the file and commit that produced it.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen  # noqa: E402
from adaptive import (ABLATION_LABELS, THETA_DEFAULT, cyclomatic_number,  # noqa: E402
                      mwc_adaptive, run_ablation)
from mwc import INF, mwc, mwc_oracle, mwc_transversal  # noqa: E402

try:
    import networkx as nx
except ImportError:                                   # networkx only needed
    nx = None                                         # for graph generation


# ---------------------------------------------------------------- provenance
def provenance() -> Dict[str, Any]:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        # Record WHICH files differ, not just that some do: a bare boolean
        # cannot distinguish "the measured code was edited" from "a job wrapper
        # path was tweaked", and only the first invalidates the numbers.
        dirty = [l.strip() for l in subprocess.check_output(
            ["git", "-C", here, "status", "--porcelain"],
            stderr=subprocess.DEVNULL).decode().splitlines() if l.strip()]
    except Exception:
        sha, dirty = "unknown", None
    cpu = platform.processor() or ""
    try:                                              # Linux: real model name
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "git_sha": sha, "git_dirty": dirty, "host": socket.gethostname(),
        "cpu": cpu, "python": sys.version.split()[0],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "argv": sys.argv,
    }


def emit(out_dir: str, name: str, payload: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    payload["provenance"] = provenance()
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    print(f"[wrote] {path}", flush=True)
    return path


def timeit(fn, repeats: int):
    """Best-of-`repeats` wall clock, matching bench_timing.py's protocol."""
    best, val = math.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        val = fn()
        best = min(best, time.perf_counter() - t0)
    return best, val


def girth_of(r):
    return r if isinstance(r, float) else r.length


def agree(vals: List[float], tol: float = 1e-9) -> bool:
    vals = [v for v in vals if v is not None]
    if not vals:
        return True
    a = vals[0]
    return all((v == INF and a == INF) or abs(v - a) <= tol for v in vals)


# ------------------------------------------------------------------ families
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


def bench_families(n: int, rng):
    """EXACTLY the families of reference/bench_timing.py, so the rerun is
    comparable to the published Table 5 row for row."""
    s = rng.randrange(10 ** 9)
    out = []
    T = (nx.random_labeled_tree(n, seed=s) if hasattr(nx, "random_labeled_tree")
         else nx.random_tree(n, seed=s))
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
    return [(name, weighted(g, rng)) for name, g in out if g.number_of_edges() > 0]


def density_family(n: int, mu_over_n: float, rng):
    """A connected graph with a prescribed mu/n: spanning tree + exactly
    round(mu_over_n * n) extra edges.  Used by the theta sweep to walk the
    crossover continuously instead of relying on whatever mu/n the named
    families happen to land on."""
    s = rng.randrange(10 ** 9)
    T = (nx.random_labeled_tree(n, seed=s) if hasattr(nx, "random_labeled_tree")
         else nx.random_tree(n, seed=s))
    G = nx.Graph(T)
    target = int(round(mu_over_n * n))
    nodes = list(G)
    guard = 0
    while G.number_of_edges() - n + 1 < target and guard < 200 * (target + 1):
        u, v = rng.sample(nodes, 2)
        if u != v:
            G.add_edge(u, v)
        guard += 1
    return weighted(G, rng)


# ------------------------------------------------------------------- timing
def cmd_timing(args):
    rng = random.Random(args.seed)
    rows = []
    for n in args.sizes:
        for name, G in bench_families(n, rng):
            adj = to_adj(G)
            nn, m = G.number_of_nodes(), G.number_of_edges()
            mu = m - nn + nx.number_connected_components(G)
            rec = dict(family=name, n=nn, m=m, mu=mu, mu_over_n=round(mu / nn, 3))

            ref = None
            if m <= args.oracle_max_m:
                t, (g_or, _) = timeit(lambda: mwc_oracle(adj), 1)
                rec["t_oracle"], ref = t, g_or
            else:
                rec["t_oracle"] = None

            t, r_all = timeit(lambda: mwc(adj, certify=False, collect_stats=True),
                              args.repeats)
            rec.update(t_allroots=t, settled_allroots=r_all.stats["total_settled"],
                       roots_allroots=r_all.stats["roots_run"])

            t, r_tv = timeit(lambda: mwc_transversal(adj, certify=False,
                                                     collect_stats=True), args.repeats)
            rec.update(t_transversal=t, settled_transversal=r_tv.stats["total_settled"],
                       roots_transversal=r_tv.stats["roots_run"])

            t, r_ad = timeit(lambda: mwc_adaptive(adj, theta=args.theta, certify=False,
                                                  collect_stats=True), args.repeats)
            rec.update(t_adaptive=t, settled_adaptive=r_ad.stats.get("total_settled"),
                       roots_adaptive=r_ad.stats.get("roots_run"),
                       adaptive_branch=r_ad.stats.get("branch"))

            # same switch, dense branch unseeded: isolates what gamma_0 buys
            t, r_ad0 = timeit(lambda: mwc_adaptive(adj, theta=args.theta,
                                                   seed_allroots=False, certify=False,
                                                   collect_stats=True), args.repeats)
            rec.update(t_adaptive_noseed=t,
                       settled_adaptive_noseed=r_ad0.stats.get("total_settled"),
                       adaptive_branch_noseed=r_ad0.stats.get("branch"))

            t, _ = timeit(lambda: mwc(adj, certify=True, collect_stats=False),
                          args.repeats)
            rec["t_allroots_certified"] = t

            vals = [girth_of(r_all), girth_of(r_tv), girth_of(r_ad),
                    girth_of(r_ad0)] + ([ref] if ref is not None else [])
            if not agree(vals):
                print(f"DISAGREEMENT {name} n={nn}: {vals}", flush=True)
                sys.exit(1)
            rec["girth"] = None if girth_of(r_all) == INF else round(girth_of(r_all), 6)
            rows.append(rec)
            print(json.dumps(rec), flush=True)
    emit(args.out, "timing.json", {"rows": rows, "theta": args.theta,
                                   "repeats": args.repeats, "seed": args.seed})


# ----------------------------------------------------------------- ablation
def cmd_ablation(args):
    rng = random.Random(args.seed)
    rows = []
    for n in args.sizes:
        for name, G in bench_families(n, rng):
            adj = to_adj(G)
            nn, m = G.number_of_nodes(), G.number_of_edges()
            mu = m - nn + nx.number_connected_components(G)
            ref = None
            for variant in args.variants:
                # A0 has no truncation; on dense graphs that is very slow, so it
                # is skipped above a size cap and the omission is recorded
                # explicitly rather than silently.
                if variant == "A0" and m > args.a0_max_m:
                    rows.append(dict(family=name, n=nn, m=m, mu_over_n=round(mu / nn, 3),
                                     variant=variant, skipped="m > a0_max_m"))
                    continue
                t, (length, cyc, st) = timeit(
                    lambda v=variant: run_ablation(adj, v, theta=args.theta,
                                                   certify=False),
                    args.repeats)
                if ref is None:
                    ref = length
                elif not agree([ref, length]):
                    print(f"DISAGREEMENT {name} n={nn} {variant}: {length} vs {ref}",
                          flush=True)
                    sys.exit(1)
                rows.append(dict(family=name, n=nn, m=m, mu=mu,
                                 mu_over_n=round(mu / nn, 3), variant=variant,
                                 label=ABLATION_LABELS[variant], t=t,
                                 roots_run=st.get("roots_run"),
                                 settled=st.get("total_settled"),
                                 branch=st.get("branch"),
                                 girth=None if length == INF else round(length, 6)))
                print(json.dumps(rows[-1]), flush=True)
    emit(args.out, "ablation.json", {"rows": rows, "labels": ABLATION_LABELS,
                                     "repeats": args.repeats, "seed": args.seed})


# -------------------------------------------------------------------- theta
def cmd_theta(args):
    """Measure the transversal-vs-allroots crossover as mu/n varies, so the
    adaptive threshold is fitted from data."""
    rng = random.Random(args.seed)
    rows = []
    for n in args.sizes:
        for r in args.ratios:
            for rep in range(args.instances):
                G = density_family(n, r, rng)
                adj = to_adj(G)
                nn, m = G.number_of_nodes(), G.number_of_edges()
                mu = m - nn + nx.number_connected_components(G)
                t_all, r_all = timeit(lambda: mwc(adj, certify=False,
                                                  collect_stats=True), args.repeats)
                t_tv, r_tv = timeit(lambda: mwc_transversal(adj, certify=False,
                                                            collect_stats=True),
                                    args.repeats)
                t_ad, r_ad = timeit(lambda: mwc_adaptive(adj, theta=args.theta,
                                                         certify=False,
                                                         collect_stats=True),
                                    args.repeats)
                if not agree([girth_of(r_all), girth_of(r_tv), girth_of(r_ad)]):
                    print(f"DISAGREEMENT theta n={nn} mu/n={r}", flush=True)
                    sys.exit(1)
                rows.append(dict(n=nn, m=m, mu=mu, mu_over_n_target=r,
                                 mu_over_n=round(mu / nn, 4), instance=rep,
                                 t_allroots=t_all, t_transversal=t_tv,
                                 t_adaptive=t_ad,
                                 speedup_tv=t_all / t_tv if t_tv else None,
                                 speedup_ad=t_all / t_ad if t_ad else None,
                                 adaptive_branch=r_ad.stats.get("branch")))
                print(json.dumps(rows[-1]), flush=True)
    emit(args.out, "theta.json", {"rows": rows, "theta_used": args.theta,
                                  "repeats": args.repeats, "seed": args.seed})


# ----------------------------------------------------------------- frontier
def cmd_frontier(args):
    """Approximate mode on the multiscale family: the deferred speed-accuracy
    frontier.  For each (alpha,beta) we report the realized ratio
    gamma_hat/gamma*, the guarantee kappa(alpha,beta), the deletions actually
    performed, and the work (settled vertices) relative to exact mode."""
    from mwc import kappa_of
    rng = random.Random(args.seed)
    rows = []
    pairs = [tuple(float(x) for x in p.split(",")) for p in args.pairs]
    for n in args.sizes:
        for rep in range(args.instances):
            adj = gen.multiscale(n, random.Random(rng.randrange(10 ** 9)))
            if not adj:
                continue
            exact = mwc(adj, certify=False, collect_stats=True)
            gstar = exact.length
            if gstar == INF:
                continue
            base_settled = exact.stats["total_settled"]
            for (alpha, beta) in pairs:
                t, r = timeit(lambda a=alpha, b=beta: mwc(
                    adj, alpha=a, beta=b, certify=False, collect_stats=True),
                    args.repeats)
                ratio = r.length / gstar if gstar else 1.0
                kap = kappa_of(alpha, beta)
                if ratio > kap + 1e-9:                    # guarantee violation
                    print(f"KAPPA VIOLATION n={n} ({alpha},{beta}) "
                          f"ratio={ratio} kappa={kap}", flush=True)
                    sys.exit(1)
                rows.append(dict(n=n, instance=rep, alpha=alpha, beta=beta,
                                 kappa=kap, gamma_star=round(gstar, 6),
                                 gamma_hat=round(r.length, 6),
                                 ratio=round(ratio, 6),
                                 deletions=r.stats.get("total_deletions", 0),
                                 settled=r.stats["total_settled"],
                                 settled_exact=base_settled,
                                 work_frac=r.stats["total_settled"] / base_settled
                                 if base_settled else None,
                                 t=t))
                print(json.dumps(rows[-1]), flush=True)
    emit(args.out, "frontier.json", {"rows": rows, "pairs": pairs,
                                     "repeats": args.repeats, "seed": args.seed})


# ------------------------------------------------------------------- argmin
def _dijkstra_pops(adj, src, avoid=None):
    """Plain Dijkstra returning (dist_to_all, #extract-min operations).

    COUNTING PROTOCOL (stated so the figure caption can state it): one
    argmin operation is counted per vertex REMOVED from the heap and settled,
    i.e. per successful extract-min; stale heap entries popped and discarded
    are not counted.  This is the same quantity as `total_settled` in the
    reference implementation's stats, so the two curves are commensurable.
    """
    dist = {src: 0.0}
    pq = [(0.0, src)]
    settled = set()
    pops = 0
    while pq:
        d, u = heapq.heappop(pq)
        if u in settled:
            continue
        settled.add(u)
        pops += 1
        for v, w in adj[u].items():
            if avoid is not None and ((u, v) == avoid or (v, u) == avoid):
                continue
            if v in settled:
                continue
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist, pops


def cmd_argmin(args):
    """argmin counts on d x d weighted grids: Algorithm 1 (transversal mode)
    and the all-roots variant versus the edge-removal oracle."""
    rows = []
    for d in args.dims:
        rng = random.Random(args.seed + d)
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(d, d))
        adj = to_adj(weighted(G, rng))
        n, m = len(adj), sum(len(x) for x in adj.values()) // 2

        # oracle: one Dijkstra per edge on G - e, counting settled vertices
        oracle_pops = 0
        best = INF
        for (u, v) in [(u, v) for u in adj for v in adj[u] if str(u) < str(v)]:
            dist, p = _dijkstra_pops(adj, u, avoid=(u, v))
            oracle_pops += p
            if v in dist:
                best = min(best, dist[v] + adj[u][v])

        r_all = mwc(adj, certify=False, collect_stats=True)
        r_tv = mwc_transversal(adj, certify=False, collect_stats=True)
        r_ad = mwc_adaptive(adj, theta=args.theta, certify=False, collect_stats=True)
        if not agree([best, r_all.length, r_tv.length, r_ad.length]):
            print(f"DISAGREEMENT argmin d={d}: "
                  f"{[best, r_all.length, r_tv.length, r_ad.length]}", flush=True)
            sys.exit(1)
        rows.append(dict(d=d, n=n, m=m, girth=round(r_all.length, 6),
                         argmin_oracle=oracle_pops,
                         argmin_allroots=r_all.stats["total_settled"],
                         argmin_transversal=r_tv.stats["total_settled"],
                         argmin_adaptive=r_ad.stats.get("total_settled"),
                         adaptive_branch=r_ad.stats.get("branch")))
        print(json.dumps(rows[-1]), flush=True)
    emit(args.out, "argmin.json",
         {"rows": rows, "seed": args.seed,
          "protocol": _dijkstra_pops.__doc__.strip()})


# --------------------------------------------------------------------- real
def load_edges(path) -> Dict[Any, Dict[Any, float]]:
    adj: Dict[Any, Dict[Any, float]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b, w = line.split()[:3]
            a, b, w = int(a), int(b), float(w)
            if a == b:
                continue
            adj.setdefault(a, {})[b] = w
            adj.setdefault(b, {})[a] = w
    return adj


def cmd_real(args):
    rows = []
    files = sorted(f for f in os.listdir(args.data) if f.endswith(".edges"))
    if args.only:
        files = [f for f in files if f[:-6] in args.only]
        if not files:
            print(f"no .edges matching {args.only}", flush=True)
            sys.exit(2)
    for fn in files:
        path = os.path.join(args.data, fn)
        adj = load_edges(path)
        n = len(adj)
        m = sum(len(d) for d in adj.values()) // 2
        if n > args.max_n:
            rows.append(dict(name=fn[:-6], n=n, m=m, skipped="n > max_n"))
            print(json.dumps(rows[-1]), flush=True)
            continue
        mu = cyclomatic_number(adj)
        meta_path = path[:-6] + ".json"
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        rec = dict(name=fn[:-6], n=n, m=m, mu=mu, mu_over_n=round(mu / n, 3),
                   weights_native=meta.get("weights_native"),
                   weight_semantics=meta.get("weight_semantics"))

        t, r_all = timeit(lambda: mwc(adj, certify=False, collect_stats=True),
                          args.repeats)
        rec.update(t_allroots=t, settled_allroots=r_all.stats["total_settled"],
                   roots_allroots=r_all.stats["roots_run"])
        t, r_tv = timeit(lambda: mwc_transversal(adj, certify=False,
                                                 collect_stats=True), args.repeats)
        rec.update(t_transversal=t, settled_transversal=r_tv.stats["total_settled"],
                   roots_transversal=r_tv.stats["roots_run"])
        t, r_ad = timeit(lambda: mwc_adaptive(adj, theta=args.theta, certify=False,
                                              collect_stats=True), args.repeats)
        rec.update(t_adaptive=t, roots_adaptive=r_ad.stats.get("roots_run"),
                   adaptive_branch=r_ad.stats.get("branch"))
        if not agree([girth_of(r_all), girth_of(r_tv), girth_of(r_ad)]):
            print(f"DISAGREEMENT real {fn}", flush=True)
            sys.exit(1)
        rec["girth"] = None if r_all.length == INF else round(r_all.length, 6)
        # certified run: proves the returned cycle is real on every instance
        r_cert = mwc_transversal(adj, certify=True, collect_stats=False)
        rec["certified_girth_matches"] = agree([r_cert.length, r_all.length])
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    name = f"real_{args.tag}.json" if args.tag else "real.json"
    emit(args.out, name, {"rows": rows, "repeats": args.repeats,
                          "data_dir": os.path.abspath(args.data)})


# --------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--theta", type=float, default=THETA_DEFAULT)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("timing")
    t.add_argument("--sizes", type=int, nargs="+", default=[400, 900, 1600])
    t.add_argument("--oracle-max-m", type=int, default=1200)
    t.set_defaults(func=cmd_timing)

    a = sub.add_parser("ablation")
    a.add_argument("--sizes", type=int, nargs="+", default=[400, 900, 1600])
    a.add_argument("--variants", nargs="+",
                   default=["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"])
    a.add_argument("--a0-max-m", type=int, default=20000)
    a.set_defaults(func=cmd_ablation)

    th = sub.add_parser("theta")
    th.add_argument("--sizes", type=int, nargs="+", default=[1600])
    th.add_argument("--ratios", type=float, nargs="+",
                    default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5])
    th.add_argument("--instances", type=int, default=3)
    th.set_defaults(func=cmd_theta)

    f = sub.add_parser("frontier")
    f.add_argument("--sizes", type=int, nargs="+", default=[200, 400])
    f.add_argument("--instances", type=int, default=25)
    # Every pair must satisfy the admissibility guard alpha < beta + 1/2.
    # The set spans all three regimes of kappa(alpha,beta): the exact region
    # (beta >= min{alpha,1/2}, kappa = 1), the sloped region (beta < alpha <= 1/2,
    # kappa = 1/(1-2alpha+2beta)) including two points close to the excluded
    # corner where kappa blows up, and the saturated region (alpha > 1/2,
    # kappa = 1/(2beta), where raising alpha cannot degrade the guarantee).
    f.add_argument("--pairs", nargs="+", default=[
        "0.0,0.0", "0.2,0.3", "0.5,0.5",                    # exact,   kappa = 1
        "0.1,0.0", "0.2,0.0", "0.3,0.0", "0.4,0.0",         # sloped,  kappa 1.25 - 5
        "0.45,0.0", "0.49,0.0",                             # sloped,  kappa 10, 50
        "0.3,0.1", "0.5,0.25",                              # sloped/mixed
        "0.6,0.2", "0.7,0.25", "0.8,0.35", "0.9,0.45"])     # saturated, kappa = 1/(2beta)
    f.set_defaults(func=cmd_frontier)

    g = sub.add_parser("argmin")
    g.add_argument("--dims", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9, 10, 12])
    g.set_defaults(func=cmd_argmin)

    r = sub.add_parser("real")
    r.add_argument("--data", required=True)
    r.add_argument("--max-n", type=int, default=60000)
    r.add_argument("--only", nargs="*", default=None,
                   help="restrict to these network names (one array task per "
                        "network, so the heavy road graphs get their own "
                        "walltime instead of serialising behind the cheap ones)")
    r.add_argument("--tag", default=None,
                   help="suffix for the output filename, so per-network tasks "
                        "do not overwrite each other")
    r.set_defaults(func=cmd_real)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
