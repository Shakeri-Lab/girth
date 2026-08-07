"""The tightness witness for Algorithm 1, INCLUDING its fundamental-cycle seed.

Why this file replaces `tightness_13vertex.py`
----------------------------------------------
The earlier 13-vertex witness made the minimum weight cycle c* a PENDANT cycle:
its only attachment to the rest of its component was the single vertex v_0.
A pendant cycle is a fundamental cycle of EVERY spanning forest, so line 2 of
Algorithm 1,

    gamma_0 = min over non-tree e of [ l_{T_0}(e) + w(e) ],

evaluates to l(c*) = gamma* exactly. The seeded algorithm therefore starts
holding the optimum and returns it, and the witness demonstrates nothing.
Measured on the shipped witness at (alpha,beta,eps) = (0.3, 0, 0.01):
gamma_0 = gamma* = 0.410000, and the algorithm returns 0.410000, ratio 1.000 --
not kappa = 2.5. The old proof silently assumed the first search runs with
gamma_in = infinity, i.e. the UNSEEDED configuration.

The repair
----------
Attach c* to a hub H at two antipodal vertices by heavy edges of weight
W = 1, and present the vertices so that the BFS forest of `mwc.spanning_forest`
restarts at H after finishing the seed component. BFS then reaches v_0 and the
antipode simultaneously at depth 1 and fills the cycle in from both sides,
leaving TWO of its edges non-tree. Neither generates c*: each closes a cycle
that routes through H and is therefore at least 2W = 2 long. So c* is not a
fundamental cycle of T_0, and gamma_0 is pinned at 1 by the separate triangle
C_0 exactly as the proof intends.

Because W = 1 the hub sits at distance D + 1 > 1/2 from the root x, outside the
radius-1/2 horizon, so the search from x is bit-for-bit what it was: the decoy
is detected, c* is not, the trigger fires, and v_0 is the unique deleted
c*-vertex. After v_0 is removed every surviving cycle is at least 1.

One numerical change: D is placed eps/8 strictly INSIDE the deletion threshold
r - beta rather than exactly on it. The knife-edge version depends on a
floating-point tie breaking the right way and does not survive it everywhere --
at (alpha,beta) = (0.5,0.1) the threshold evaluates to 0.39749999999999996
against delta(v_0) = 0.3975, so v_0 escapes and the ratio collapses to 1. The
antipode still clears the horizon, at 1/2 + eps/8 instead of 1/2 + eps/4.

Run this file to reproduce the tightness table.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from alg1_exact_rational import run_alg1, true_girth  # noqa: E402
from mwc import INF, _transversal_for, kappa_of, mwc, mwc_oracle  # noqa: E402

W_HEAVY = 1.0          # hub attachment weight; any W >= 1/2 works
K0 = 3                 # size of the seed cycle C_0 (length 1)


def build(alpha, beta, eps, k=4, nseg=2, hub=True, dl=None, eta=None):
    """Return (adj, root_order, meta).

    `hub=False` reproduces the old pendant witness, for the regression that
    shows the seed defeats it.  Insertion order is load-bearing: `spanning_forest`
    is a BFS over insertion order, so H must precede every c* vertex.
    """
    ae = min(alpha, 0.5)
    if not (0 <= beta < ae):
        raise ValueError("require 0 <= beta < min{alpha, 1/2}")
    if not alpha < beta + 0.5:
        raise ValueError("require alpha < beta + 1/2")
    gstar = (1 - 2 * (ae - beta)) + eps
    if not 0 < gstar < 1:
        raise ValueError(f"require eps < 2(min(alpha,1/2)-beta); got gstar={gstar}")
    if k % 2:
        raise ValueError("k must be even so the antipode sits at arc-distance gstar/2")
    r = ae - eps / 4.0
    D = (ae - beta) - 3.0 * eps / 8.0        # eps/8 inside the threshold r - beta
    if D <= 0:
        raise ValueError("D must be positive")
    if dl is None:
        dl = eps / 8.0
    if eta is None:
        eta = min(eps / 8.0, r / 4.0)
    s = gstar / k

    adj: dict = {}

    def add(u, v, w):
        adj.setdefault(u, {})[v] = w
        adj.setdefault(v, {})[u] = w

    # (A) seed component C_0: one non-tree edge, fundamental cycle of length 1.
    for i in range(K0):
        add(f"c0_{i}", f"c0_{(i + 1) % K0}", 1.0 / K0)

    # (B) hub first, so BFS restarts here and enters c* from both attachments.
    if hub:
        add("H", "v0", W_HEAVY)
        add("H", f"v{k // 2}", W_HEAVY)

    for i in range(k):                       # the minimum weight cycle c*
        add(f"v{i}", f"v{(i + 1) % k}", s)

    seg = r / nseg                           # root x -> b, then the decoy triangle
    prev = "x"
    for j in range(1, nseg):
        add(prev, f"xb{j}", seg)
        prev = f"xb{j}"
    add(prev, "b", seg)
    add("b", "p", eta)
    add("b", "q", eta)
    add("p", "q", 1 + dl - 2 * eta)

    seg = D / nseg                           # root x -> v_0
    prev = "x"
    for j in range(1, nseg):
        add(prev, f"xv{j}", seg)
        prev = f"xv{j}"
    add(prev, "v0", seg)

    order = ["x"] + [u for u in adj if u != "x"]
    meta = dict(gstar=gstar, r=r, D=D, k=k, s=s, hub=hub,
                kappa=kappa_of(alpha, beta),
                cstar={f"v{i}" for i in range(k)})
    return adj, order, meta


def check(alpha, beta, eps, hub=True, **kw):
    """Run BOTH implementations, seeded exactly as Algorithm 1 line 2 prescribes."""
    adj, order, meta = build(alpha, beta, eps, hub=hub, **kw)
    _, gamma0, cyc0, _ = _transversal_for(adj)
    gstar, _ = mwc_oracle(adj)

    res = mwc(adj, alpha=alpha, beta=beta, root_order=order,
              gamma0=gamma0, cycle0=cyc0, certify=True, collect_stats=True)
    deleted = {z for rec in res.stats["per_root"] for z in rec["deleted"]}

    ghat_sim, _log = run_alg1(adj, order, alpha, beta, gamma0=gamma0)

    return dict(alpha=alpha, beta=beta, eps=eps, n=len(adj), k=meta["k"],
                gamma0=gamma0, gstar=gstar, ghat=res.length, ghat_sim=ghat_sim,
                ratio=res.length / gstar, kappa=meta["kappa"],
                cstar_deleted=sorted(deleted & meta["cstar"]),
                agree=abs(res.length - ghat_sim) <= 1e-9)


GRID = [(0.10, 0.00), (0.25, 0.00), (0.30, 0.00), (0.40, 0.00),
        (0.45, 0.20), (0.50, 0.10), (0.55, 0.10), (0.60, 0.15), (0.70, 0.25)]
EPSILONS = (1e-2, 2e-3, 5e-4, 1e-4)


def main():
    adj, order, _ = build(0.3, 0.0, 0.01, hub=False)
    _, g0_pendant, _, _ = _transversal_for(adj)
    gstar_pendant, _ = mwc_oracle(adj)
    print("REGRESSION -- the old pendant witness under the seed:")
    print(f"  gamma_0 = {g0_pendant:.6f}, gamma* = {gstar_pendant:.6f}  "
          f"-> seed hands over the optimum, ratio 1.000, kappa unattained\n")

    print("SEED-RESISTANT WITNESS (hub attachment), both implementations:")
    hdr = (f"{'alpha':>6}{'beta':>6}{'eps':>9}{'n':>4}{'gamma_0':>9}{'gamma*':>10}"
           f"{'ghat':>8}{'ratio':>9}{'kappa':>9}{'agree':>7}  deleted")
    print(hdr)
    print("-" * len(hdr))
    bad = 0
    for (a, b) in GRID:
        for eps in EPSILONS:
            if eps >= 2 * (min(a, 0.5) - b) - 1e-12:
                continue
            R = check(a, b, eps)
            ok = (abs(R["gamma0"] - 1.0) < 1e-12
                  and abs(R["ghat"] - 1.0) < 1e-12
                  and R["cstar_deleted"] == ["v0"]
                  and R["agree"]
                  and R["ratio"] <= R["kappa"] + 1e-12)
            bad += not ok
            print(f"{a:6.2f}{b:6.2f}{eps:9.5f}{R['n']:4d}{R['gamma0']:9.5f}"
                  f"{R['gstar']:10.6f}{R['ghat']:8.5f}{R['ratio']:9.4f}"
                  f"{R['kappa']:9.4f}{str(R['agree']):>7}  {R['cstar_deleted']}"
                  f"{'' if ok else '   <-- FAILED'}")
    print()
    if bad:
        print(f"FAILED: {bad} configuration(s) did not satisfy the witness invariants")
        sys.exit(1)
    print("All configurations: gamma_0 = 1, ghat = 1, exactly v0 deleted, "
          "both implementations agree, ratio <= kappa.")

    out = os.environ.get("OUTDIR")
    if out:
        import json
        rows = [check(a, b, e) for (a, b) in GRID for e in EPSILONS
                if e < 2 * (min(a, 0.5) - b) - 1e-12]
        payload = {"rows": rows, "grid": GRID, "epsilons": list(EPSILONS),
                   "note": "seeded Algorithm 1; gamma_0 must be 1, not gamma*"}
        try:                                   # reuse the campaign's provenance
            from campaign import provenance
            payload["provenance"] = provenance()
        except Exception:
            pass
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "tightness.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=1, default=str)
        print(f"[wrote] {path}")


if __name__ == "__main__":
    main()
