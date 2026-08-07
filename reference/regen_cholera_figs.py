"""Both Cholera figures as genuine vector art, with no raster objects.

Why this replaces the earlier pair
----------------------------------
The shipped Delaunay figure was a PNG whose background was an OpenStreetMap
tile layer -- inherently raster, and carrying map attribution text baked into
the panel. The density figure was also a PNG, and its continuous colorbar
survives PDF export as a raster QuadMesh even when everything else is vector.

Here both are drawn from the data alone:
  * Voronoi cell boundaries, Delaunay edges and case locations as vector paths;
  * the density rho* encoded by a DISCRETE colour scale, so the colorbar is a
    handful of filled rectangles rather than an interpolated image;
  * `rasterized=False` everywhere, and `pdf.fonttype 42` so text stays text.

Losing the basemap loses geographic context. That is the price of the vector
rule, and it is the right trade for a figure whose content is a graph: the
spatial layout is preserved exactly, since vertex positions are the case
coordinates.

Verify with `pdfimages -list` -- it must report no images for either file.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import matplotlib.colors as mcolors        # noqa: E402
import numpy as np                         # noqa: E402
import networkx as nx                      # noqa: E402

LM = "/project/shakeri-lab/graph_alg/loop_modulus"
sys.path.insert(0, LM)
sys.path.insert(0, os.path.dirname(LM))
from certified_modulus import certified_loop_modulus     # noqa: E402

TOL = 1e-4
OUT = os.environ.get("OUTDIR", "/scratch/hs9hd/cholera_fig")
N_LEVELS = 8                                # discrete colour bands

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def load():
    import geopandas
    from libpysal import weights
    from libpysal.cg import voronoi_frames
    cases = geopandas.read_file(os.path.join(LM, "cholera_cases.gpkg"))
    coords = np.column_stack((cases.geometry.x, cases.geometry.y))
    cells, _ = voronoi_frames(coords, clip="convex hull")
    delaunay = weights.Rook.from_dataframe(cells, use_index=False)
    G = nx.Graph(delaunay.to_networkx())
    pos = dict(zip(G.nodes, coords))
    return G, pos, cells


def save(fig, stem):
    for ext in ("pdf", "svg"):
        p = os.path.join(OUT, f"{stem}.{ext}")
        fig.savefig(p, format=ext, bbox_inches="tight", pad_inches=0.02)
        print(f"[wrote] {p}", flush=True)
    plt.close(fig)


def fig_delaunay(G, pos, cells, width=3.9):
    fig, ax = plt.subplots(figsize=(width, width * 0.95))
    cells.boundary.plot(ax=ax, color="0.75", linewidth=0.4)     # vector Voronoi
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="0.25", width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2.0,
                           node_color="black", linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    save(fig, "cholera_delaunay_vector")


def fig_rho(G, pos, rho, width=4.43):
    edges = [tuple(sorted(e)) for e in G.edges()]
    vals = np.array([rho.get(e, 0.0) for e in edges])
    lo, hi = float(vals.min()), float(vals.max())
    # Discrete scale => the colorbar is filled rectangles, not an image.
    bounds = np.linspace(lo, hi, N_LEVELS + 1)
    cmap = plt.get_cmap("viridis", N_LEVELS)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    colors = [cmap(norm(v)) for v in vals]
    widths = 0.35 + 2.1 * (vals - lo) / max(hi - lo, 1e-12)

    fig, ax = plt.subplots(figsize=(width, width * 0.82))
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors,
                           width=widths, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=1.6, node_color="black",
                           alpha=0.75, linewidths=0, ax=ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02, boundaries=bounds,
                      ticks=bounds[::2], spacing="proportional")
    cb.set_label(r"$\rho^*$")
    if cb.solids is not None:
        cb.solids.set_rasterized(False)     # keep the bar as vector patches
    cb.outline.set_linewidth(0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    save(fig, "cholera_rho_certified")


def main():
    os.makedirs(OUT, exist_ok=True)
    G, pos, cells = load()
    print(f"graph: n={G.number_of_nodes()} m={G.number_of_edges()}", flush=True)

    res = certified_loop_modulus(G, tolerance=TOL, mode="certified")
    if not res["certified"]:
        print("REFUSING to plot an uncertified density", flush=True)
        sys.exit(1)
    print(f"MODULUS {res['modulus']:.6f}  m_rho {res['m_rho']:.6f}", flush=True)

    fig_delaunay(G, pos, cells)
    fig_rho(G, pos, res["rho"])
    with open(os.path.join(OUT, "cholera_figs.json"), "w") as f:
        json.dump({k: v for k, v in res.items() if k != "rho"}, f,
                  indent=1, default=str)


if __name__ == "__main__":
    main()
