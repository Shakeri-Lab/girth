"""Simplified BMSSP (Bounded Multi-Source Shortest Paths).

This *lite* version accepts an arbitrary set of sources S and returns regular
distance/predecessor/depth dictionaries for all vertices whose distance ≤
`bound`; vertices beyond the bound keep distance = inf so downstream code
continues to work.

Implementation: heap-based multi-source Dijkstra with early cutoff when the
current distance > bound.  Good enough for first integration; later we will
add Duan's pivot recursion.
"""
from __future__ import annotations

import heapq
from typing import Dict, Tuple, Set

import networkx as nx
from hybrid_mwc import SSSPStrategy  # type: ignore


class BMSSPLiteStrategy(SSSPStrategy):
    """Early-exit multi-source Dijkstra."""

    def run(self, G: nx.Graph, sources: int | Set[int], bound: float):  # type: ignore[override]
        if isinstance(sources, int):
            src_set: Set[int] = {sources}
        else:
            src_set = set(sources)

        INF = float("inf")
        dist: Dict[int, float] = {n: INF for n in G.nodes()}
        depth: Dict[int, int] = {n: 0 for n in G.nodes()}
        preds: Dict[int, int] = {}

        heap: list[Tuple[float, int]] = []
        for s in src_set:
            dist[s] = 0.0
            heap.append((0.0, s))

        heapq.heapify(heap)
        while heap:
            d, u = heapq.heappop(heap)
            if d > bound:
                break  # All remaining vertices are beyond bound
            if d != dist[u]:
                continue  # Outdated entry
            for v in G.neighbors(u):
                new_d = d + G[u][v].get("weight", 1.0)
                if new_d < dist[v] and new_d <= bound:
                    dist[v] = new_d
                    preds[v] = u
                    depth[v] = depth[u] + 1
                    heapq.heappush(heap, (new_d, v))

        return dist, preds, depth
