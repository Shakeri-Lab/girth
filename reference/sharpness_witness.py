"""Witnesses for Proposition (Sharpness), seed-resistant.

Both parts of the proposition previously used a minimum weight cycle that is a
fundamental cycle of every spanning forest -- part (i) an isolated 4-cycle
component, part (ii) a 4-cycle pendant on a path. Line 2 of Algorithm 1 then
sets gamma_0 = gamma* before any root search, so the algorithm returns gamma*
and neither part demonstrates what it claims. This is the same defect as in the
tightness witness; see `tightness_witness.py` for the full account.

The repair is the same device: attach c* to a hub H by two heavy edges at
antipodal vertices, and order the vertices so the BFS forest of
`mwc.spanning_forest` enters c* from both attachments. Two of c*'s edges are
then non-tree and each closes a cycle through H of length >= 2W, so c* is not a
fundamental cycle and gamma_0 is pinned at 1 by the separate triangle C_0.

(i) The radius constant 1/2 is optimal: with radius tau*gamma for any tau < 1/2
    the antipode of the root escapes the horizon and c* is never detected.
(ii) A root set meeting every component but no cycle of c* fails.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from mwc import _transversal_for, mwc, mwc_oracle, spanning_forest  # noqa: E402

W_HEAVY = 1.0


def _add(adj, u, v, w):
    adj.setdefault(u, {})[v] = w
    adj.setdefault(v, {})[u] = w


def _is_cycle_transversal(adj, S):
    """True iff every simple cycle of `adj` meets S.  Checked directly: delete S
    and confirm what remains is a forest (m < n per component)."""
    S = set(S)
    rest = {u: {v: w for v, w in nb.items() if v not in S}
            for u, nb in adj.items() if u not in S}
    n = len(rest)
    m = sum(len(nb) for nb in rest.values()) // 2
    seen, comps = set(), 0
    for s in rest:
        if s in seen:
            continue
        comps += 1
        stack = [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            for v in rest[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
    return m == n - comps                      # acyclic iff mu = 0


def _seed_triangle(adj):
    """C_0: length exactly 1, its own component -- this is what pins gamma_0."""
    for i in range(3):
        _add(adj, f"c0_{i}", f"c0_{(i + 1) % 3}", 1.0 / 3.0)


def build_radius(tau, eps=None, hub=True):
    """(i) c* is a uniform 4-cycle of length 1-eps; every root lies on it."""
    if not 0 < tau < 0.5:
        raise ValueError("require 0 < tau < 1/2")
    if eps is None:
        eps = (1 - 2 * tau) / 2.0
    if not 0 < eps < 1 - 2 * tau:
        raise ValueError("require 0 < eps < 1-2tau")
    gstar = 1.0 - eps
    adj: dict = {}
    _seed_triangle(adj)
    if hub:                      # inserted before any c* vertex: BFS restarts here
        _add(adj, "H", "v0", W_HEAVY)
        _add(adj, "H", "v2", W_HEAVY)
    for i in range(4):
        _add(adj, f"v{i}", f"v{(i + 1) % 4}", gstar / 4.0)
    order = ["c0_0"] + [u for u in adj if u != "c0_0"]
    return adj, order, gstar


def build_transversal(hub=True):
    """(ii) S meets every component but no vertex of c*."""
    gstar = 0.9
    adj: dict = {}
    _seed_triangle(adj)
    if hub:
        _add(adj, "H", "v0", W_HEAVY)
        _add(adj, "H", "v2", W_HEAVY)
    for i in range(4):
        _add(adj, f"v{i}", f"v{(i + 1) % 4}", gstar / 4.0)
    _add(adj, "h", "hm", 0.305)          # h --0.61-- v0, in two segments
    _add(adj, "hm", "v0", 0.305)
    order = ["c0_0"] + [u for u in adj if u != "c0_0"]
    return adj, order, gstar


def report():
    print("Proposition (Sharpness), part (i): the radius constant 1/2 is optimal")
    print(f"{'tau':>6}{'hub':>6}{'gamma_0':>10}{'gamma*':>9}{'ghat':>9}  verdict")
    ok = True
    for tau in (0.10, 0.25, 0.40, 0.49):
        for hub in (False, True):
            adj, order, gstar = build_radius(tau, hub=hub)
            _, g0, c0, _ = _transversal_for(adj)
            # S must be a genuine CYCLE TRANSVERSAL, or the construction says
            # nothing about the theorem it is meant to make sharp.  {s_0} u V(c*)
            # meets C_0, c*, and (with the hub) every cycle through h; processing
            # s_0 changes nothing because the seed already stands at 1.
            S = ["c0_0"] + [f"v{i}" for i in range(4)]
            assert _is_cycle_transversal(adj, S), "S is not a cycle transversal"
            res = mwc(adj, root_order=order, roots=S, gamma0=g0, cycle0=c0,
                      radius_factor=tau, certify=True, collect_stats=False)
            strict = res.length > gstar + 1e-12
            if hub and not strict:
                ok = False
            print(f"{tau:6.2f}{str(hub):>6}{g0:10.5f}{gstar:9.5f}{res.length:9.5f}"
                  f"  {'returns > gamma*' if strict else 'returns gamma* (witness fails)'}")

    print()
    print("Proposition (Sharpness), part (ii): meeting every component is not enough")
    print(f"{'hub':>6}{'gamma_0':>10}{'gamma*':>9}{'ghat':>9}  verdict")
    for hub in (False, True):
        adj, order, gstar = build_transversal(hub=hub)
        _, g0, c0, _ = _transversal_for(adj)
        S = ["c0_0", "h"]
        res = mwc(adj, root_order=order, roots=S, gamma0=g0, cycle0=c0,
                  certify=True, collect_stats=False)
        strict = res.length > gstar + 1e-12
        if hub and not strict:
            ok = False
        print(f"{str(hub):>6}{g0:10.5f}{gstar:9.5f}{res.length:9.5f}"
              f"  {'returns > gamma*' if strict else 'returns gamma* (witness fails)'}")

    # structural evidence: with the hub, c* is not a fundamental cycle
    adj, _, _ = build_transversal(hub=True)
    _, _, non_tree, _ = spanning_forest(adj)
    sq = {frozenset((f"v{i}", f"v{(i + 1) % 4}")) for i in range(4)}
    n_sq_nontree = sum(1 for u, v, _ in non_tree if frozenset((u, v)) in sq)
    print(f"\nwith hub: {n_sq_nontree} of the 4 edges of c* are non-tree "
          f"(>=2 means c* is not a fundamental cycle, so the seed cannot find it)")
    print(f"true girth = {mwc_oracle(adj)[0]:.5f}")
    if not ok:
        print("\nFAILED: a hub configuration did not return a value above gamma*")
        sys.exit(1)
    print("\nBoth parts hold for Algorithm 1 including its line-2 seed.")


if __name__ == "__main__":
    report()
