"""Simple block-based bucket priority queue.

Keys are non-negative floats.  Keys are bucketed by floor(key / block_size).
Within each bucket we keep a min-heap.  Operations are O(1) amortised for
randomly distributed keys; worst-case O(log n).
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Dict, Tuple, List, Any


class BlockBasedPQ:
    def __init__(self, block_size: float = 1.0):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.buckets: Dict[int, List[Tuple[float, Any]]] = defaultdict(list)
        self.min_bucket: int | None = None
        self.size = 0

    # --------------------------------------------------------------
    def _bucket_id(self, key: float) -> int:
        return int(key // self.block_size)

    # --------------------------------------------------------------
    def insert(self, key: float, item):
        b = self._bucket_id(key)
        heapq.heappush(self.buckets[b], (key, item))
        if self.min_bucket is None or b < self.min_bucket:
            self.min_bucket = b
        self.size += 1

    def empty(self) -> bool:
        return self.size == 0

    def __len__(self):
        return self.size

    def pop(self):
        if self.empty():
            raise IndexError("pop from empty BlockBasedPQ")
        # advance min_bucket until non-empty
        while self.min_bucket not in self.buckets or not self.buckets[self.min_bucket]:
            self.min_bucket += 1  # type: ignore[arg-type]
        key, item = heapq.heappop(self.buckets[self.min_bucket])
        self.size -= 1
        return key, item

    # --------------------------------------------------------------
    def pull_batch(self, limit: int):
        """Pop up to *limit* items sharing the smallest bucket.

        Returns (max_key_in_batch, set_of_items).  Useful for BMSSP recursion.
        """
        if limit <= 0:
            raise ValueError("limit must be positive")
        if self.empty():
            raise IndexError("pull from empty BlockBasedPQ")
        # ensure min_bucket positioned correctly
        while self.min_bucket not in self.buckets or not self.buckets[self.min_bucket]:
            self.min_bucket += 1  # type: ignore[arg-type]
        batch_items = set()
        max_key = 0.0
        bucket_heap = self.buckets[self.min_bucket]
        while bucket_heap and len(batch_items) < limit:
            key, item = heapq.heappop(bucket_heap)
            max_key = max(max_key, key)
            batch_items.add(item)
            self.size -= 1
        if not bucket_heap:
            # bucket depleted
            del self.buckets[self.min_bucket]
        return max_key, batch_items
