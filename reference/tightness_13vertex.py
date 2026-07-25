"""Q2: simplified SINGLE-DELETION tightness construction for 0 <= beta < alpha <= 1/2.

Scale is fixed by ghat = 1.
  gamma* = (1 - 2(alpha-beta)) + eps          (true MWC length, < 1)
  r      = alpha - eps/4                       (= d_to_cycle at the decoy root x)
  D      = (alpha - beta) - eps/4              (= r - beta*gamma, the discard threshold)
  l_best = 1 + dl,  dl < eps/4                 (decoy cycle length, > gamma = 1)
  c*     = uniform k-cycle, edge weight s = gamma*/k

Components:
  A) seed cycle C_0, uniform cycle of length 1, its own component  -> sets gamma = 1
  B) decoy root x -- path(r) --> b -- triangle(eta, eta, 1+dl-2eta)   (the decoy cycle C_1)
     x -- path(D) --> v_0 in c*
Root order: [seed vertex of C_0, x, everything else].
"""
from alg1 import add, path, run_alg1, true_girth, INF


def build(alpha, beta, eps, k=None, dl=None, eta=None, k0=3, nseg=2):
    assert 0 <= beta < alpha <= 0.5
    gstar = (1 - 2 * (alpha - beta)) + eps
    assert 0 < gstar < 1, (gstar,)              # needs eps < 2(alpha-beta)
    r = alpha - eps / 4.0
    D = (alpha - beta) - eps / 4.0
    assert D > 0
    if dl is None:
        dl = eps / 8.0
    if eta is None:
        eta = min(eps / 8.0, r / 4.0)
    if k is None:
        # k even => antipodal vertex of c* sits at D + gamma*/2 = 1/2 + eps/4 > 1/2.
        # (odd k additionally needs s = gamma*/k <= eps/2, i.e. k >= 2 gamma*/eps)
        k = 2 * max(2, int(gstar / eps) + 1)
    assert dl < eps / 4 and eta < eps / 4
    assert 1 + dl - 2 * eta > 0

    adj = {}
    # (A) seed cycle C_0, length 1, own component
    s0 = 1.0 / k0
    for i in range(k0):
        add(adj, f"c0_{i}", f"c0_{(i+1)%k0}", s0)
    # (B) MWC c*: uniform k-cycle
    s = gstar / k
    for i in range(k):
        add(adj, f"v{i}", f"v{(i+1)%k}", s)
    # decoy root x -> b, decoy triangle
    path(adj, "x", "b", r, nseg, "xb")
    add(adj, "b", "p", eta)
    add(adj, "b", "q", eta)
    add(adj, "p", "q", 1 + dl - 2 * eta)
    # x -> v0
    path(adj, "x", "v0", D, nseg, "xv")

    order = ["c0_0", "x"] + [u for u in adj if u not in ("c0_0", "x")]
    meta = dict(gstar=gstar, r=r, D=D, k=k, s=s, dl=dl, eta=eta,
                ghat_target=1.0,
                kappa=(1.0 / (1 - 2 * alpha + 2 * beta)
                       if 1 - 2 * alpha + 2 * beta > 1e-15 else INF))
    return adj, order, meta


def check(alpha, beta, eps, verbose=False, **kw):
    adj, order, meta = build(alpha, beta, eps, **kw)
    ghat, log = run_alg1(adj, order, alpha, beta, trace=verbose)
    gstar_true = true_girth(adj)
    x_rec = [e for e in log if e['root'] == 'x'][0]
    return dict(alpha=alpha, beta=beta, eps=eps, k=meta['k'], s=meta['s'],
                ghat=ghat, gstar_pred=meta['gstar'], gstar_true=gstar_true,
                ratio=ghat / gstar_true, kappa=meta['kappa'],
                fired=x_rec['fired'], v0_removed='v0' in x_rec['removed'],
                n=len(adj))


if __name__ == "__main__":
    import sys, json
    grid = [(0.5, 0.0), (0.4, 0.0), (0.3, 0.0), (0.25, 0.0), (0.1, 0.0),
            (0.5, 0.1), (0.45, 0.2), (0.4, 0.3), (0.5, 0.4), (0.35, 0.05)]
    print(f"{'alpha':>6} {'beta':>5} {'eps':>8} {'k':>6} {'ghat':>10} "
          f"{'gamma*':>10} {'ratio':>10} {'kappa':>10} {'fire':>5} {'del':>4} {'n':>5}")
    rows = []
    for (a, b) in grid:
        for eps in (0.2, 0.05, 0.01, 0.002, 0.0005):
            if eps >= 2 * (a - b) - 1e-12:
                continue
            R = check(a, b, eps)
            rows.append(R)
            print(f"{a:6.2f} {b:5.2f} {eps:8.4f} {R['k']:6d} {R['ghat']:10.6f} "
                  f"{R['gstar_true']:10.6f} {R['ratio']:10.6f} {R['kappa']:10.6f} "
                  f"{str(R['fired']):>5} {str(R['v0_removed']):>4} {R['n']:5d}")
    json.dump(rows, open("q2_results.json", "w"), indent=1)
