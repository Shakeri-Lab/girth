"""Static O(1) LCA via Euler tour + Sparse Table.

Pure-Python implementation suitable for trees built once and queried many
(times).  Pre-processing: O(n log n).  Query: O(1).
"""
from __future__ import annotations

from math import log2, floor
from typing import List, Dict

import numpy as np
import networkx as nx


class EulerLCA:
    """Builds Euler tour & sparse table for constant-time LCA queries."""

    __slots__ = (
        "first_occ",
        "euler",
        "depths",
        "st",
        "log",
    )

    def __init__(self, tree_parents: Dict[int, int], depth: Dict[int, int], nodes: List[int]):
        # Build adjacency list from parent map (tree is rooted but undirected)
        adj = {u: [] for u in nodes}
        for child, par in tree_parents.items():
            adj[child].append(par)
            adj[par].append(child)

        root = nodes[0]
        self.euler: List[int] = []
        self.depths: List[int] = []
        self.first_occ: Dict[int, int] = {}

        def dfs(u: int, p: int):
            self.first_occ.setdefault(u, len(self.euler))
            self.euler.append(u)
            self.depths.append(depth[u])
            for v in adj[u]:
                if v == p:
                    continue
                dfs(v, u)
                # add parent again on return
                self.euler.append(u)
                self.depths.append(depth[u])

        dfs(root, -1)

        # Sparse table over depths array storing index of minimum depth in range
        m = len(self.depths)
        k_max = floor(log2(m)) + 1
        self.log = np.empty(m, dtype=int)
        self.log[1:] = np.floor(np.log2(np.arange(1, m))).astype(int)

        st = np.empty((k_max, m), dtype=int)
        st[0] = np.arange(m)  # indices
        for k in range(1, k_max):
            span = 1 << k
            half = span >> 1
            prev = st[k - 1]
            for i in range(m - span + 1):
                left = prev[i]
                right = prev[i + half]
                st[k, i] = left if self.depths[left] <= self.depths[right] else right
        self.st = st

    # ------------------------------------------------------------------
    def lca(self, u: int, v: int) -> int:
        l, r = self.first_occ[u], self.first_occ[v]
        if l > r:
            l, r = r, l
        j = self.log[r - l + 1]
        left = self.st[j, l]
        right = self.st[j, r - (1 << j) + 1]
        return self.euler[left] if self.depths[left] <= self.depths[right] else self.euler[right]
