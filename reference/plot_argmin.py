"""Regenerate the argmin-count figure from campaign.py's argmin.json.

The preliminary version of the paper carried a plot of argmin counts on d x d
grids whose counting protocol was never stated and whose generating script did
not survive; it was withdrawn in the 2026-08-06 revision.  This script
reproduces the comparison from a recorded protocol (see `--protocol` in the
JSON) and emits a vector PDF sized so that LaTeX does not rescale it -- the
figure is included at exactly its natural width, so the 10 pt labels stay
10 pt on the page, as JGAA requires.

Usage:
    uv run --python 3.12 --with matplotlib python plot_argmin.py \
        --json results/argmin.json --out Figures/argmin_counts.pdf
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Target typeset width: 0.55 * \textwidth, \textwidth = 15cm -> 8.25cm = 3.25in.
# The figure is included with width=0.55\textwidth, i.e. at natural size, so
# no scaling is applied and nominal font sizes survive to the page.
FIGSIZE = (3.25, 2.45)
BASE_PT = 10

plt.rcParams.update({
    "font.size": BASE_PT,
    "axes.labelsize": BASE_PT,
    "axes.titlesize": BASE_PT,
    "xtick.labelsize": BASE_PT,
    "ytick.labelsize": BASE_PT,
    "legend.fontsize": BASE_PT,
    "pdf.fonttype": 42,          # embed TrueType, no Type-3 bitmaps
    "ps.fonttype": 42,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.load(open(args.json))
    rows = sorted(data["rows"], key=lambda r: r["d"])
    d = [r["d"] for r in rows]
    oracle = [r["argmin_oracle"] for r in rows]
    allroots = [r["argmin_allroots"] for r in rows]
    transv = [r["argmin_transversal"] for r in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(d, oracle, "o-", color="black", lw=1.4, ms=3.5,
            label="edge removal")
    ax.plot(d, allroots, "s--", color="#1f6fb4", lw=1.4, ms=3.5,
            label="Alg.\\ 1, all roots")
    ax.plot(d, transv, "^:", color="#c0392b", lw=1.6, ms=4,
            label="Alg.\\ 1, transversal")
    ax.set_yscale("log")
    ax.set_xlabel(r"grid side $d$")
    ax.set_ylabel("argmin operations")
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.6)
    ax.legend(frameon=False, loc="upper left", handlelength=1.8,
              borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(args.out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"[wrote] {args.out}")
    print(f"  d range {d[0]}..{d[-1]}; oracle/transversal ratio "
          f"{oracle[0]/transv[0]:.1f}x .. {oracle[-1]/transv[-1]:.1f}x")


if __name__ == "__main__":
    main()
