"""Recursive BMSSP (Duan et al.) – initial scaffold.

This is *not* the full optimised algorithm yet.  It sets the structure:
• recursion by level
• pivot selection stub
• block-based PQ for frontier expansion
"""
from __future__ import annotations

from typing import Dict, Set, Tuple, List

import networkx as nx

from hybrid_mwc import SSSPStrategy
from block_pq import BlockBasedPQ

INF = float("inf")


class BMSSPFullStrategy(SSSPStrategy):
    def __init__(self, k: int | None = None):
        self.k = k  # branching parameter, default computed from log n later

    # ------------------------------------------------------------------
    def run(self, G: nx.Graph, source: int | Set[int], bound: float):  # type: ignore[override]
        if isinstance(source, int):
            src_set = {source}
        else:
            src_set = set(source)
        n = len(G)
        if self.k is None:
            import math

            self.k = int(max(2, math.log(n) ** (1 / 3)))
        # distance / pred / depth initial
        dist = {v: INF for v in G.nodes()}
        depth = {v: 0 for v in G.nodes()}
        pred: Dict[int, int] = {}
        # multi-source initial frontier
        pq = BlockBasedPQ()
        for s in src_set:
            dist[s] = 0.0
            pq.insert(0.0, s)

        # For now: single-level – behaves like BMSSP-lite but with block PQ
        while not pq.empty():
            d, u = pq.pop()
            if d > bound:
                break
            if d != dist[u]:
                continue
            for v in G.neighbors(u):
                w = G[u][v].get("weight", 1.0)
                nd = d + w
                if nd < dist[v] and nd <= bound:
                    dist[v] = nd
                    pred[v] = u
                    depth[v] = depth[u] + 1
                    pq.insert(nd, v)
        return dist, pred, depth
