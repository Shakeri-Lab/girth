"""Hybrid Minimum Weight Cycle (MWC) framework
------------------------------------------------
This module integrates pluggable SSSP strategies (Dijkstra, BMSSP-lite, …)
with pluggable LCA back-ends (binary lifting, Euler tour).  It provides a
common driver to experiment with algorithmic variants.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Iterable, Set

import networkx as nx

# ------------------------------------------------------------------
# SSSP strategy interface + baseline implementations
# ------------------------------------------------------------------

class SSSPStrategy(ABC):
    @abstractmethod
    def run(self, G: nx.Graph, source: int | Set[int], bound: float):
        """Return (distances, predecessors, depth) dictionaries."""


from shortest_cycle import dijkstra_base  # re-use proven code

class DijkstraStrategy(SSSPStrategy):
    def __init__(self, use_heapdict: bool = False):
        self.use_heapdict = use_heapdict

    def run(self, G, source, bound):  # type: ignore[override]
        distances, preds, depth, _ = dijkstra_base(
            G, next(iter(source)) if isinstance(source, set) else source,
            bound * 2,
            use_heapdict=self.use_heapdict,
        )
        return distances, preds, depth

# Lazy import to avoid circular deps when bmssp_lite imports this file
try:
    from bmssp_lite import BMSSPLiteStrategy  # noqa: F401
from bmssp_full import BMSSPFullStrategy  # noqa: F401
except ModuleNotFoundError:
    BMSSPLiteStrategy = None  # type: ignore

# ------------------------------------------------------------------
# LCA back-end selection
# ------------------------------------------------------------------
from shortest_cycle import LCATree  # binary lifting
try:
    from lca_euler import EulerLCA
except ModuleNotFoundError:
    EulerLCA = None  # type: ignore

class LCAStrategy(ABC):
    @abstractmethod
    def build(self, parents: Dict[int, int], depth: Dict[int, int], nodes: list[int]):
        pass

class BinaryLCAStrategy(LCAStrategy):
    def build(self, parents, depth, nodes):  # type: ignore[override]
        return LCATree(parents, depth, nodes)

class EulerLCAStrategy(LCAStrategy):
    def build(self, parents, depth, nodes):  # type: ignore[override]
        if EulerLCA is None:
            raise ImportError("lca_euler module not found")
        return EulerLCA(parents, depth, nodes)

# ------------------------------------------------------------------
# Hybrid driver
# ------------------------------------------------------------------

class HybridMWC:
    def __init__(
        self,
        G: nx.Graph,
        *,
        sssp: SSSPStrategy | None = None,
        lca: LCAStrategy | None = None,
    ):
        if G.is_directed():
            raise ValueError("Graph must be undirected")
        self.G = G
        self.sssp = sssp or DijkstraStrategy()
        self.lca_backend = lca or BinaryLCAStrategy()
        self.gamma = float("inf")

    def minimum_weight_cycle(self) -> float | None:
        if len(self.G) <= 1 or self.G.number_of_edges() == 0:
            return None

        nodes = list(self.G.nodes())
        min_edge_w = min(w for *_, w in self.G.edges(data="weight", default=1.0))

        for s in nodes:
            dist, preds, depth = self.sssp.run(self.G, s, self.gamma / 2)
            lca_struct = self.lca_backend.build(preds, depth, nodes)

            for u, v, attr in self.G.edges(data=True):
                if dist[u] == float("inf") or dist[v] == float("inf"):
                    continue
                p = lca_struct.lca(u, v)
                if p and p != u and p != v:
                    cycle_len = dist[u] + dist[v] + attr.get("weight", 1.0) - 2 * dist[p]
                    if cycle_len < self.gamma:
                        self.gamma = cycle_len
            if self.gamma <= 4 * min_edge_w:
                break
        return None if self.gamma == float("inf") else self.gamma


def hybrid_mwc_length(G: nx.Graph, *, use_bmssp_lite: bool = False, use_euler_lca: bool = False):
    sssp: SSSPStrategy | None = None
    if use_bmssp_lite:
        from bmssp_lite import BMSSPLiteStrategy  # local import
        sssp = BMSSPLiteStrategy()
    elif use_bmssp_lite is False and 'BMSSPFullStrategy' in globals():
        # default to full BMSSP when available and flag not set
        pass
    lca_strat = EulerLCAStrategy() if use_euler_lca else BinaryLCAStrategy()
    return HybridMWC(G, sssp=sssp, lca=lca_strat).minimum_weight_cycle()
