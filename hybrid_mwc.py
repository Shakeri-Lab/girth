"""Hybrid Minimum Weight Cycle (MWC) framework
------------------------------------------------
Pure-Python scaffolding that will let us swap different
single-source shortest-path (SSSP) strategies while re-using the
cycle-checking logic already present in `shortest_cycle.sota_shortest_cycle`.

Initially it just delegates to the existing Dijkstra-based code so we can
verify correctness without changing behaviour.  Future commits will add
BMSSP variants.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Iterable

import networkx as nx

# Re-use helper from the existing implementation
from shortest_cycle import dijkstra_base, LCATree


class SSSPStrategy(ABC):
    """Strategy interface for bounded SSSP computations."""

    @abstractmethod
    def run(self, G: nx.Graph, source: int, bound: float) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, int]]:
        """Return (distances, predecessors, depth) limited to *bound*.

        *distances[v]* may be **> bound** (or `inf`) but must be defined for
        *all* vertices so that downstream LCA construction can proceed.
        """


class DijkstraStrategy(SSSPStrategy):
    """Current baseline using binary-heap Dijkstra from `shortest_cycle`."""

    def __init__(self, use_heapdict: bool = False):
        self.use_heapdict = use_heapdict

    def run(self, G: nx.Graph, source: int, bound: float):  # type: ignore[override]
        distances, preds, depth, _ = dijkstra_base(
            G, source, bound * 2,  # dijkstra_base expects *gamma*, we have bound = gamma/2
            use_heapdict=self.use_heapdict,
        )
        return distances, preds, depth


class HybridMWC:
    """Hybrid MWC algorithm (skeleton).

    For now it simply iterates over all vertices and executes the chosen SSSP
    strategy using an ever-shrinking γ.  Cycle detection still relies on the
    existing LCA + edge-scan logic copied from `shortest_cycle` to keep the
    diff minimal.  Once BMSSP variants are implemented the SSSP cost will drop.
    """

    def __init__(self, G: nx.Graph, sssp: SSSPStrategy | None = None):
        if G.is_directed():
            raise ValueError("Graph must be undirected")
        self.G = G
        self.sssp = sssp or DijkstraStrategy()
        self.gamma = float("inf")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def minimum_weight_cycle(self) -> float | None:
        """Compute the MWC length (skeleton version).

        Returns *None* if the graph is acyclic.
        """
        if len(self.G) <= 1 or self.G.number_of_edges() == 0:
            return None

        nodes = list(self.G.nodes())
        min_edge_weight = min(w for _, _, w in self.G.edges(data="weight", default=1.0))

        for s in nodes:
            distances, preds, depth = self.sssp.run(self.G, s, self.gamma / 2)
            lca = LCATree(preds, depth, nodes)

            # Edge scan (undirected => iterate each edge once)
            for u, v, attr in self.G.edges(data=True):
                if distances[u] == float("inf") or distances[v] == float("inf"):
                    continue
                p = lca.lca(u, v)
                if p and p != u and p != v:
                    cycle_len = distances[u] + distances[v] + attr.get("weight", 1.0) - 2 * distances[p]
                    if cycle_len < self.gamma:
                        self.gamma = cycle_len

            # Early exit heuristic: if the current best γ is already the
            # minimum possible (4 * min edge weight) we can break.
            if self.gamma <= 4 * min_edge_weight:
                break

        return None if self.gamma == float("inf") else self.gamma


# -------------------------------------------------------------------------
# Convenience helper
# -------------------------------------------------------------------------
def hybrid_mwc_length(G: nx.Graph) -> float | None:
    """Functional wrapper mirroring old API style."""
    return HybridMWC(G).minimum_weight_cycle()
