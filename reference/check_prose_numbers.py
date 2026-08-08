"""Check the manuscript's campaign-derived numbers against the result JSON.

Every round of this revision produced at least one PARTIAL update: a number
refreshed in one sentence and left stale four lines away, or refreshed in the
body and left stale in a generated caption.  Greps cannot catch that, because
a stale number is still a well-formed number.  This does the arithmetic.

    uv run --python 3.12 python check_prose_numbers.py \
        --results results --tex ../../manuscript/main.tex

Exit status is non-zero if any check fails, so it can gate a build.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

TOL = 0.02          # relative tolerance for a quoted value
CHECKS: list[tuple[str, str, float]] = []   # (label, regex, expected)


def load_real(results):
    out = {}
    for f in glob.glob(os.path.join(results, "real_*.json")):
        r = json.load(open(f))["rows"][0]
        out[r.get("name") or r.get("tag")] = r
    return out


def med(rows, variant, family, key="t", scale=1.0):
    xs = [r[key] * scale for r in rows
          if r.get("variant") == variant and r.get("family") == family]
    return st.median(xs) if xs else None


def build_checks(results):
    real = load_real(results)

    # --- real networks: all-roots vs transversal on the named graphs --------
    for name, pat in [
        ("sydney-road",
         r"costing all roots \$([\d.]+)\\times\$ on \\texttt\{sydney-road\}"),
        ("chicago-regional-road",
         r"\$([\d.]+)\\times\$ on \\texttt\{chicago-regional-road\}"),
    ]:
        r = real[name]
        CHECKS.append((f"{name}: all-roots/transversal", pat,
                       r["t_allroots"] / r["t_transversal"]))

    # --- the five resolvable gaps quoted in the switch paragraph -----------
    for name in ("uspowergrid-synth", "chicago-sketch-road", "rome99-road",
                 "sydney-road", "lesmis"):
        r = real[name]
        gap = max(r["t_allroots"], r["t_transversal"]) / min(r["t_allroots"], r["t_transversal"])
        CHECKS.append((f"{name}: resolvable gap",
                       r"\\texttt\{" + re.escape(name) + r"\} \$([\d.]+)\\times\$", gap))

    # --- oracle ------------------------------------------------------------
    op = os.path.join(results, "oracle", "timing.json")
    if os.path.exists(op):
        o = [r for r in json.load(open(op))["rows"] if r.get("t_oracle") is not None]
        A = [r["t_oracle"] / r["t_allroots"] for r in o]
        T = [r["t_oracle"] / r["t_transversal"] for r in o]
        B = [max(a, t) for a, t in zip(A, T)]
        CHECKS.append(("oracle: better-of-two median",
                       r"is a median \$(\d+)\\times\$ faster than the oracle", st.median(B)))
        CHECKS.append(("oracle: all-roots median",
                       r"all roots has median \$(\d+)\\times\$", st.median(A)))
        CHECKS.append(("oracle: transversal median",
                       r"transversal roots median \$(\d+)\\times\$", st.median(T)))
        CHECKS.append(("oracle: within-job control (oracle/all-roots)",
                       r"is \$([\d.]+)\$ and \$[\d.]+\$ across the two runs",
                       st.median([r["t_oracle"] / r["t_allroots"] for r in o])))
        slow = [r for r in o if r["t_oracle"] / r["t_allroots"] < 1]
        CHECKS.append(("oracle: count of all-roots losses",
                       r"On \$(\d+)\$ of the \$192\$ instances", float(len(slow))))
        if slow:
            tt = [r["t_oracle"] / r["t_transversal"] for r in slow]
            CHECKS.append(("oracle: transversal low end on those",
                           r"the transversal variant is \$(\d+)\$--\$\d+\\times\$ faster", min(tt)))
            CHECKS.append(("oracle: transversal high end on those",
                           r"the transversal variant is \$\d+\$--\$(\d+)\\times\$ faster", max(tt)))

    # --- ablation ----------------------------------------------------------
    ap = os.path.join(results, "ablation.json")
    if os.path.exists(ap):
        rows = [r for r in json.load(open(ap))["rows"]
                if r.get("n") == 1600 and "t" in r]
        fams = ["near_tree", "sparse_er", "grid", "small_world", "geometric", "dense_er"]
        ratios = [med(rows, "A6", f) / med(rows, "A5", f) for f in fams]
        CHECKS.append(("A6/A5 low end",
                       r"slower than A5 on every family, by \$([\d.]+)\\times\$", min(ratios)))
        CHECKS.append(("A6/A5 high end",
                       r"by \$[\d.]+\\times\$ to \$([\d.]+)\\times\$", max(ratios)))

    return real


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--results", default="results")
    ap_.add_argument("--tex", required=True)
    a = ap_.parse_args()

    build_checks(a.results)
    tex = open(a.tex, encoding="utf-8", errors="ignore").read()

    bad = 0
    print(f"{'check':<44}{'in paper':>11}{'artifact':>11}  status")
    print("-" * 78)
    for label, pat, expected in CHECKS:
        m = re.search(pat, tex)
        if not m:
            print(f"{label:<44}{'--':>11}{expected:>11.3f}  PATTERN NOT FOUND")
            bad += 1
            continue
        got = float(m.group(1))
        ok = abs(got - expected) <= TOL * max(abs(expected), 1e-9)
        # counts are exact
        if expected == int(expected) and expected < 1000:
            ok = abs(got - expected) < 0.5
        print(f"{label:<44}{got:>11.3f}{expected:>11.3f}  {'ok' if ok else 'STALE'}")
        bad += not ok

    print()
    if bad:
        print(f"{bad} check(s) failed -- a quoted number disagrees with the artifact")
        sys.exit(1)
    print(f"all {len(CHECKS)} campaign-derived numbers match the artifact")


if __name__ == "__main__":
    main()
