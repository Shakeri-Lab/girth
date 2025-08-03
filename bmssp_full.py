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
    """Multi-level BMSSP with block-PQ batching (first working recursion)."""

    def __init__(self, k: int | None = None):
        self.k = k  # branching factor (≈ log^{1/3} n)

    # ------------------------------------------------------------------
    def run(self, G: nx.Graph, source: int | Set[int], bound: float):  # type: ignore[override]
        import math
        if isinstance(source, int):
            sources = {source}
        else:
            sources = set(source)
        n = len(G)
        if self.k is None:
            self.k = int(max(2, math.log(n) ** (1 / 3)))
        # parameters for recursion depth
        self.t = int(math.ceil(math.log(n, 2) ** (2 / 3))) or 1
        self.max_level = int(math.ceil(math.log(n, 2) / self.t))

        # global state across recursion
        self.G = G
        self.bound = bound
        self.dist: Dict[int, float] = {v: INF for v in G.nodes()}
        self.pred: Dict[int, int] = {}
        self.depth: Dict[int, int] = {v: 0 for v in G.nodes()}

        # initialise sources
        for s in sources:
            self.dist[s] = 0.0
        self._bmssp(self.max_level, bound, sources)
        return self.dist, self.pred, self.depth

    # ------------------------------------------------------------------
    def _bmssp(self, level: int, B: float, S: Set[int]):
        from bmssp_lite import BMSSPLiteStrategy
        if level == 0 or not S:
            # base case – reuse BMSSP-lite (multi-source Dijkstra with cutoff)
            lite = BMSSPLiteStrategy()
            dist2, pred2, depth2 = lite.run(self.G, S, B)
            for v, d in dist2.items():
                if d < self.dist[v]:
                    self.dist[v] = d
                    if v in pred2:
                        self.pred[v] = pred2[v]
                        self.depth[v] = depth2[v]
            return

        # find pivots to shrink frontier
        pivots, _ = self._find_pivots(self.G, B, S, level)

        import math
        block_const = max(1.0, B / (self.k * (math.log(len(self.G)) + 1)))
        pq = BlockBasedPQ(block_size=block_const)
        for p in pivots:
            pq.insert(self.dist[p], p)
        U: Set[int] = set()
        threshold = self.k * (2 ** (level * self.t))
        while len(U) < threshold and not pq.empty():
            B_i, S_i = pq.pull_batch(self.k)
            self._bmssp(level - 1, B_i, S_i)
            # relax edges of returned set
            for u in S_i:
                du = self.dist[u]
                if du > B_i:
                    continue
                for v in self.G.neighbors(u):
                    nd = du + self.G[u][v].get("weight", 1.0)
                    if nd < self.dist[v] and nd <= B:
                        self.dist[v] = nd
                        self.pred[v] = u
                        self.depth[v] = self.depth[u] + 1
                        pq.insert(nd, v)
            U |= S_i

    # ------------------------------------------------------------------
    def _find_pivots(self, G: nx.Graph, bound: float, sources: Set[int], level: int):
        """Simple k-iteration BFS to shrink frontier (sub-task 1)."""
        dist_local: Dict[int, float] = {s: 0.0 for s in sources}
        frontier = set(sources)
        reached = set(sources)
        preds: Dict[int, int] = {}
        for _ in range(self.k):
            new_frontier: Set[int] = set()
            for u in frontier:
                du = dist_local[u]
                if du > bound:
                    continue
                for v in G.neighbors(u):
                    w = G[u][v].get("weight", 1.0)
                    nd = du + w
                    if nd <= bound and nd < dist_local.get(v, INF):
                        dist_local[v] = nd
                        preds[v] = u
                        new_frontier.add(v)
            if not new_frontier:
                break
            reached |= new_frontier
            if len(reached) > self.k * len(sources):
                return sources, reached  # degenerate fallback
            frontier = new_frontier

        # Count subtree sizes (naïve): each reached vertex climbs to nearest source
        child_count = {s: 0 for s in sources}
        for v in reached:
            if v in sources:
                continue
            # climb preds until hitting a source or missing
            u = v
            while u not in sources and u in preds:
                u = preds[u]
            if u in sources:
                child_count[u] += 1
        pivot_thresh = self.k * (2 ** max(0, level * self.t - 1))
        pivots = {s for s, c in child_count.items() if c >= pivot_thresh}
        if not pivots:
            pivots = set(list(sources)[: max(1, len(sources)//self.k)])
        return pivots, reached
