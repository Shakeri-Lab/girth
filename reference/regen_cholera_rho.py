"""Regenerate the Cholera rho* figure from the CERTIFIED pipeline, as vector.

Why this exists
---------------
The figure shipped with the manuscript was produced by
`test_cholera.py`, which plots the output of the *uncertified*
`calculate_loop_modulus_rho_preprocessed` and bakes the resulting value into the
panel title -- it read "Final Rho* Values (Mod2~101.71)". That number appears
nowhere in the paper: Table 1 reports 0.333 and 78.333 for the uncertified
configurations and 103.927 for the two certified ones. A figure asserting a
fourth, uncertified modulus contradicts the section it illustrates.

It was also a two-panel PNG (initial cycle set + rho*) while the caption
describes only the rho* panel, and it was a bitmap line drawing.

This script fixes all three: it runs the certified solver, plots ONE panel of
the certified rho*, encodes rho* in both edge colour and edge width (as the
caption claims), writes vector PDF at the printed width so the text lands above
10pt, and prints the modulus so the caption can quote it from the same run.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.colors as mcolors      # noqa: E402
import numpy as np                       # noqa: E402
import networkx as nx                    # noqa: E402

LM = "/project/shakeri-lab/graph_alg/loop_modulus"
sys.path.insert(0, LM)
sys.path.insert(0, os.path.dirname(LM))   # so `loop_modulus.core` resolves

from certified_modulus import certified_loop_modulus   # noqa: E402

TOL = 1e-4
OUT = os.environ.get("OUTDIR", "/scratch/hs9hd/cholera_fig")

# Printed at 0.75\textwidth; \textwidth = 15cm = 5.906in -> 4.43in.
# Emitting at the printed width means LaTeX does not rescale, so 10pt stays 10pt.
FIGW = 4.43
BASE_PT = 10
plt.rcParams.update({
    "font.size": BASE_PT, "axes.labelsize": BASE_PT,
    "xtick.labelsize": BASE_PT, "ytick.labelsize": BASE_PT,
    "legend.fontsize": BASE_PT, "figure.titlesize": BASE_PT,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_cholera_graph():
    import geopandas
    from libpysal import weights
    from libpysal.cg import voronoi_frames
    cases = geopandas.read_file(os.path.join(LM, "cholera_cases.gpkg"))
    coords = np.column_stack((cases.geometry.x, cases.geometry.y))
    cells, _ = voronoi_frames(coords, clip="convex hull")
    delaunay = weights.Rook.from_dataframe(cells, use_index=False)
    G = nx.Graph(delaunay.to_networkx())
    pos = dict(zip(G.nodes, coords))
    return G, pos


def main():
    os.makedirs(OUT, exist_ok=True)
    G, pos = load_cholera_graph()
    print(f"graph: n={G.number_of_nodes()} m={G.number_of_edges()}", flush=True)

    res = certified_loop_modulus(G, tolerance=TOL, mode="certified")
    print(json.dumps({k: v for k, v in res.items() if k != "rho"},
                     indent=1, default=str), flush=True)
    if not res["certified"]:
        print("REFUSING to plot an uncertified density", flush=True)
        sys.exit(1)

    rho = res["rho"]
    edges = [tuple(sorted(e)) for e in G.edges()]
    vals = np.array([rho.get(e, 0.0) for e in edges])
    norm = mcolors.Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(norm(v)) for v in vals]
    # caption says thickness AND colour encode rho*, so encode both
    widths = 0.35 + 2.1 * (vals - vals.min()) / max(vals.max() - vals.min(), 1e-12)

    fig, ax = plt.subplots(figsize=(FIGW, FIGW * 0.82))
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors,
                           width=widths, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=1.6, node_color="black",
                           alpha=0.75, ax=ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"$\rho^*$")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    path = os.path.join(OUT, "cholera_rho_certified.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"[wrote] {path}", flush=True)
    print(f"MODULUS_FOR_CAPTION {res['modulus']:.6f} "
          f"m_rho={res['m_rho']:.6f} certified={res['certified']}", flush=True)

    with open(os.path.join(OUT, "cholera_rho_certified.json"), "w") as f:
        json.dump({k: v for k, v in res.items() if k != "rho"}, f,
                  indent=1, default=str)


if __name__ == "__main__":
    main()
