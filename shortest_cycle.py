"""
Shortest Cycle Algorithms with Mathematically Certified Pruning
==============================================================

Implements minimum-weight cycle (MWC) detection with certified pruning,
degeneracy preprocessing, strict lexicographical tie-breaking, piecewise
frontier sentinels, dynamic certified pruning, historical certificate
accumulation, structural heavy-edge filtering, and epoch-based snapshot
parallelism.
"""

import heapq
import time
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import concurrent.futures
import os
import numpy as np
import networkx as nx

# Global flag for Fibonacci heap availability
# Optional compiled C++/Cython acceleration
try:
    from c_extensions import fast_mwc
    _HAS_ACCELERATION = True
except (ImportError, ModuleNotFoundError, ValueError):
    try:
        from girth.c_extensions import fast_mwc
        _HAS_ACCELERATION = True
    except (ImportError, ModuleNotFoundError, ValueError):
        try:
            import fast_mwc
            _HAS_ACCELERATION = True
        except (ImportError, ModuleNotFoundError, ValueError):
            fast_mwc = None
            _HAS_ACCELERATION = False


EXTERNAL_FIB_HEAP_AVAILABLE = False
EXTERNAL_FIB_HEAP_IMPORTED = False


# ---------------------------------------------------------------------------
# Graph Generation Helpers
# ---------------------------------------------------------------------------

def create_grid_graph(size: int, default_weight: float = 10.0, cycle_weight: float = 1.0) -> nx.Graph:
    """Create a grid graph with a hidden low-weight cycle."""
    G = nx.grid_2d_graph(size, size)
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        G[u][v]['weight'] = default_weight
    hidden_nodes = [size * size - size - 2, size * size - size - 1, size * size - 1, size * size - 2]
    for i in range(4):
        u, v = hidden_nodes[i], hidden_nodes[(i + 1) % 4]
        G[u][v]['weight'] = cycle_weight
    return G


def create_spatial_graph(n_nodes: int, radius: float = 0.3, default_weight: float = 10.0, cycle_weight: float = 1.0) -> nx.Graph:
    """Create a random geometric graph with a hidden low-weight cycle."""
    G = nx.random_geometric_graph(n_nodes, radius)
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        G[u][v]['weight'] = default_weight
    hidden_nodes = np.random.choice(n_nodes, 4, replace=False)
    for i in range(4):
        u, v = hidden_nodes[i], hidden_nodes[(i + 1) % 4]
        if G.has_edge(u, v):
            G[u][v]['weight'] = cycle_weight
        else:
            G.add_edge(u, v, weight=cycle_weight)
    return G


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WitnessRecord:
    """Immutable record storing an incumbent girth and its witness simple cycle."""
    length: float
    cycle: Tuple[Any, ...]


class DynamicLCATree:
    """
    Binary-lifting Lowest Common Ancestor over a rooted directed tree spanning Q.
    Table height is dynamically sized to max(1, len(Q).bit_length()).
    """
    __slots__ = ("nodes", "pos", "depth", "up", "log_depth", "queries")

    def __init__(self, nodes: Sequence[Any], parent: Dict[Any, Any], root: Any):
        self.nodes = list(nodes)
        self.pos = {v: i for i, v in enumerate(self.nodes)}
        n = len(self.nodes)
        self.queries = 0
        self.log_depth = max(1, n.bit_length())

        par = [0] * n
        for v, i in self.pos.items():
            if v == root:
                par[i] = i
            else:
                p = parent.get(v)
                par[i] = self.pos[p] if (p is not None and p in self.pos) else i

        depth = [-1] * n
        for i in range(n):
            if depth[i] >= 0:
                continue
            stack = []
            curr = i
            while depth[curr] < 0 and par[curr] != curr:
                stack.append(curr)
                curr = par[curr]
            base = depth[curr] if depth[curr] >= 0 else 0
            depth[curr] = base
            while stack:
                node = stack.pop()
                base += 1
                depth[node] = base
        self.depth = depth

        up = [par]
        for k in range(1, self.log_depth):
            prev = up[k - 1]
            up.append([prev[prev[i]] for i in range(n)])
        self.up = up

    def lca(self, u: Any, v: Any) -> Optional[Any]:
        self.queries += 1
        if u not in self.pos or v not in self.pos:
            return None
        iu, iv = self.pos[u], self.pos[v]
        du, dv = self.depth[iu], self.depth[iv]
        if du < dv:
            iu, iv = iv, iu
            du, dv = dv, du

        diff = du - dv
        k = 0
        while diff:
            if diff & 1:
                iu = self.up[k][iu]
            diff >>= 1
            k += 1

        if iu == iv:
            return self.nodes[iu]

        for k in range(self.log_depth - 1, -1, -1):
            if self.up[k][iu] != self.up[k][iv]:
                iu = self.up[k][iu]
                iv = self.up[k][iv]
        return self.nodes[self.up[0][iu]]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "query_count": self.queries,
            "cache_hits": 0,
            "cache_hit_ratio": 0.0,
            "cache_size": 0
        }


class LCATree:
    """Backward-compatible wrapper for legacy LCATree interface."""
    def __init__(self, parents: Dict[Any, Any], depth: Dict[Any, int], nodes: Sequence[Any]):
        self.nodes = list(nodes)
        root = None
        for n in nodes:
            if parents.get(n) is None:
                root = n
                break
        if root is None and nodes:
            root = nodes[0]
        self._tree = DynamicLCATree(nodes, parents, root)
        self.log_max_depth = self._tree.log_depth
        self.node_to_index = self._tree.pos
        self.depth = {n: self._tree.depth[self._tree.pos[n]] for n in self._tree.nodes}
        self.cache = {}
        self.query_count = 0
        self.cache_hits = 0

    def lca(self, u: Any, v: Any) -> Optional[Any]:
        self.query_count += 1
        return self._tree.lca(u, v)

    def get_stats(self) -> Dict[str, Any]:
        return self._tree.get_stats()


class FibonacciHeap:
    """A simple Fibonacci/priority heap simulation for legacy compatibility."""
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.counter = 0
        self.REMOVED = '<removed>'
        self.operations = 0

    def push(self, item, priority):
        self.operations += 1
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.heap, entry)
        self.counter += 1

    def remove_item(self, item):
        self.operations += 1
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def pop(self):
        self.operations += 1
        while self.heap:
            priority, _, item = heapq.heappop(self.heap)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return priority, item
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        return not any(item[-1] is not self.REMOVED for item in self.heap)


def dijkstra_base(G, start, gamma, use_fib_heap=False):
    """Legacy dijkstra_base maintained for backward compatibility."""
    distances = {n: float('inf') for n in G.nodes()}
    depth = {n: 0 for n in G.nodes()}
    distances[start] = 0.0
    preds = {}
    visited = set()
    ops = 0
    heap = [(0.0, start)]

    while heap:
        current_dist, u = heapq.heappop(heap)
        ops += 1
        if u in visited or current_dist > gamma / 2.0:
            continue
        visited.add(u)
        for v in G.neighbors(u):
            new_dist = current_dist + G[u][v].get('weight', 1.0)
            if new_dist < distances[v]:
                distances[v] = new_dist
                preds[v] = u
                depth[v] = depth[u] + 1
                heapq.heappush(heap, (new_dist, v))
                ops += 1

    return distances, preds, depth, ops


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _get_node_index(nodes: Sequence[Any]) -> Dict[Any, int]:
    return {node: i for i, node in enumerate(nodes)}


def reconstruct_cycle(u: Any, v: Any, p: Any, parent: Dict[Any, Any]) -> List[Any]:
    """
    Reconstructs simple cycle [p ... u, v ... p] closed by cotree chord (u, v).
    Handles p == u and p == v correctly.
    """
    path_u = [u]
    curr = u
    while curr != p:
        curr = parent[curr]
        path_u.append(curr)
    path_u.reverse()

    path_v = [v]
    curr = v
    while curr != p:
        curr = parent[curr]
        path_v.append(curr)

    cycle = path_u + path_v[:-1] + [p]
    return cycle


def prune_low_degree(H: nx.Graph, active_set: Set[Any]) -> List[Any]:
    """Iteratively remove vertices with degree < 2 from active subgraph H."""
    removed = []
    queue = [v for v in list(H.nodes()) if H.degree(v) < 2]
    while queue:
        v = queue.pop()
        if v not in H:
            continue
        nbrs = list(H.neighbors(v))
        H.remove_node(v)
        active_set.discard(v)
        removed.append(v)
        for nbr in nbrs:
            if nbr in H and H.degree(nbr) < 2:
                queue.append(nbr)
    return removed


def filter_heavy_edges(H: nx.Graph, gamma: float, weight: str = 'weight') -> int:
    """Remove all edges with weight >= gamma from H."""
    if gamma == float('inf'):
        return 0
    heavy = [(u, v) for u, v, d in H.edges(data=True) if d.get(weight, 1.0) >= gamma]
    if heavy:
        H.remove_edges_from(heavy)
    return len(heavy)


# ---------------------------------------------------------------------------
# Preprocessing Module (3-Pass Linear-Time Degeneracy)
# ---------------------------------------------------------------------------

def preprocess_degeneracy(
    G: Union[nx.Graph, nx.MultiGraph],
    weight: str = 'weight'
) -> Tuple[nx.Graph, float, Optional[List[Any]], bool]:
    """
    Executes 3-pass linear-time degeneracy preprocessing (Proposition 1.2).
    Pass 1: Multigraph 2-cycles
    Pass 2: Zero-weight DFS guard
    Pass 3: Spanning forest seed & acyclic forest guard
    
    Returns:
        (G_simple, gamma_0, cycle_0, is_done)
    """
    if G.is_directed():
        raise ValueError("Input graph must be undirected")

    gamma_0 = float('inf')
    cycle_0 = None

    is_multi = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    G_simple = nx.Graph()
    G_simple.add_nodes_from(G.nodes())

    if is_multi:
        edge_groups: Dict[Tuple[Any, Any], List[float]] = {}
        for u, v, k, data in G.edges(keys=True, data=True):
            if u == v:
                continue
            w = data.get(weight, 1.0)
            if w < 0:
                raise ValueError("Negative edge weights are not supported")
            pair = (u, v) if str(u) <= str(v) else (v, u)
            edge_groups.setdefault(pair, []).append(w)

        for (u, v), weights in edge_groups.items():
            weights.sort()
            G_simple.add_edge(u, v, **{weight: weights[0]})
            if len(weights) >= 2:
                two_cycle_len = weights[0] + weights[1]
                if two_cycle_len == 0.0:
                    return G_simple, 0.0, [u, v, u], True
                if two_cycle_len < gamma_0:
                    gamma_0 = two_cycle_len
                    cycle_0 = [u, v, u]
    else:
        for u, v, data in G.edges(data=True):
            if u == v:
                continue
            w = data.get(weight, 1.0)
            if w < 0:
                raise ValueError("Negative edge weights are not supported")
            G_simple.add_edge(u, v, **{weight: w})

    if len(G_simple) <= 2 or G_simple.number_of_edges() == 0:
        return G_simple, gamma_0, cycle_0, True

    # Pass 2: Zero-weight DFS guard
    e0_edges = [(u, v) for u, v, d in G_simple.edges(data=True) if d.get(weight, 1.0) == 0.0]
    if e0_edges:
        G0 = nx.Graph()
        G0.add_nodes_from(G_simple.nodes())
        G0.add_edges_from(e0_edges)
        visited = set()
        for root in G0.nodes():
            if root in visited or G0.degree(root) == 0:
                continue
            parent_map = {root: None}
            stack = [(root, None)]
            visited.add(root)
            while stack:
                u, p = stack.pop()
                for v in G0.neighbors(u):
                    if v == p:
                        continue
                    if v in visited:
                        path_u = []
                        curr = u
                        while curr is not None:
                            path_u.append(curr)
                            curr = parent_map.get(curr)
                        path_v = []
                        curr = v
                        while curr is not None:
                            path_v.append(curr)
                            curr = parent_map.get(curr)
                        anc = set(path_u)
                        lca = next(node for node in path_v if node in anc)
                        idx_u = path_u.index(lca)
                        idx_v = path_v.index(lca)
                        cycle = path_u[:idx_u + 1] + list(reversed(path_v[:idx_v])) + [path_u[0]]
                        return G_simple, 0.0, cycle, True
                    parent_map[v] = u
                    visited.add(v)
                    stack.append((v, u))

    # Pass 3: Spanning forest seed & acyclic forest guard
    num_nodes = G_simple.number_of_nodes()
    num_edges = G_simple.number_of_edges()
    comps = list(nx.connected_components(G_simple))
    num_comps = len(comps)

    if num_edges == num_nodes - num_comps:
        return G_simple, gamma_0, cycle_0, True

    T0 = nx.minimum_spanning_tree(G_simple, weight=weight, algorithm="prim")
    
    # Fast rooted Tree-LCA prefix distance queries
    prefix_dist: Dict[Any, float] = {}
    parent_map: Dict[Any, Any] = {}
    comp_map: Dict[Any, int] = {}
    lca_trees: Dict[int, DynamicLCATree] = {}

    for comp_idx, comp_nodes in enumerate(comps):
        if not comp_nodes:
            continue
        comp_list = list(comp_nodes)
        root = comp_list[0]
        prefix_dist[root] = 0.0
        parent_map[root] = root
        comp_map[root] = comp_idx

        queue = [root]
        order = [root]
        visited_comp = {root}
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            for nbr in T0.neighbors(curr):
                if nbr not in visited_comp:
                    visited_comp.add(nbr)
                    w_edge = T0[curr][nbr].get(weight, 1.0)
                    prefix_dist[nbr] = prefix_dist[curr] + w_edge
                    parent_map[nbr] = curr
                    comp_map[nbr] = comp_idx
                    queue.append(nbr)
                    order.append(nbr)
        lca_trees[comp_idx] = DynamicLCATree(order, parent_map, root)

    for u, v, d in G_simple.edges(data=True):
        if not T0.has_edge(u, v):
            c_u = comp_map.get(u)
            c_v = comp_map.get(v)
            if c_u is None or c_u != c_v:
                continue
            w = d.get(weight, 1.0)
            tree_lca = lca_trees[c_u]
            p = tree_lca.lca(u, v)
            if p is None:
                continue

            # Prefix distance query: d(u, p) + d(v, p) = dist[u] + dist[v] - 2 * dist[p]
            tree_dist = prefix_dist[u] + prefix_dist[v] - 2.0 * prefix_dist[p]
            cycle_len = tree_dist + w
            if cycle_len < gamma_0:
                path_u = []
                curr = u
                while curr != p:
                    path_u.append(curr)
                    curr = parent_map[curr]
                path_u.append(p)

                path_v = []
                curr = v
                while curr != p:
                    path_v.append(curr)
                    curr = parent_map[curr]

                tree_path = path_u + list(reversed(path_v))
                actual_len = sum(T0[tree_path[i]][tree_path[i + 1]].get(weight, 1.0) for i in range(len(tree_path) - 1)) + w
                if actual_len < gamma_0:
                    gamma_0 = actual_len
                    cycle_0 = tree_path + [u]

    return G_simple, gamma_0, cycle_0, False


# ---------------------------------------------------------------------------
# Truncated Dijkstra with Strict Lexicographical Tie-Breaking
# ---------------------------------------------------------------------------

def truncated_dijkstra_lex(
    H: nx.Graph,
    root: Any,
    radius: float,
    active_set: Set[Any],
    node_index: Dict[Any, int],
    weight: str = 'weight'
) -> Tuple[Dict[Any, float], Dict[Any, int], Dict[Any, Any], List[Any], Set[Any], float]:
    """
    Truncated Dijkstra with fixed radius and strict lexicographical tie-breaking.
    """
    dist: Dict[Any, float] = {root: 0.0}
    hops: Dict[Any, int] = {root: 0}
    parent: Dict[Any, Any] = {}
    settled: Set[Any] = set()
    Q_list: List[Any] = []

    heap = [(0.0, 0, node_index[root], root)]
    frontier_dists: List[float] = []
    has_cross_edges = False

    while heap:
        d, h, idx, u = heapq.heappop(heap)
        if u in settled:
            continue
        if d >= radius:
            frontier_dists.append(d)
            has_cross_edges = True
            for rem_d, _, _, rem_u in heap:
                if rem_u not in settled:
                    frontier_dists.append(rem_d)
            break

        settled.add(u)
        Q_list.append(u)

        for v in H.neighbors(u):
            if v not in active_set or v in settled:
                continue
            w = H[u][v].get(weight, 1.0)
            cand_d = d + w
            cand_h = h + 1
            cand_par_idx = node_index[u]

            curr_d = dist.get(v, float('inf'))
            curr_h = hops.get(v, float('inf'))
            curr_par_idx = node_index[parent[v]] if v in parent else float('inf')

            if (cand_d < curr_d or
                (cand_d == curr_d and cand_h < curr_h) or
                (cand_d == curr_d and cand_h == curr_h and cand_par_idx < curr_par_idx)):

                dist[v] = cand_d
                hops[v] = cand_h
                parent[v] = u
                if cand_d < radius:
                    heapq.heappush(heap, (cand_d, cand_h, node_index[v], v))
                else:
                    frontier_dists.append(cand_d)
                    has_cross_edges = True

    for u in settled:
        for v in H.neighbors(u):
            if v in active_set and v not in settled:
                has_cross_edges = True
                w = H[u][v].get(weight, 1.0)
                frontier_dists.append(dist[u] + w)

    if not has_cross_edges and not frontier_dists:
        tau_x = float('inf')
    else:
        tau_x = radius

    return dist, hops, parent, Q_list, settled, tau_x


# ---------------------------------------------------------------------------
# Core Certified Pruning Implementation
# ---------------------------------------------------------------------------

def _certified_pruning_parallel(
    H: nx.Graph,
    gamma: float,
    best_cycle: Optional[List[Any]],
    active: Set[Any],
    node_index: Dict[Any, int],
    lower_bank: Dict[Any, float],
    K: float,
    weight: str,
    n_workers: Optional[int]
) -> Tuple[float, List[Any]]:
    """Synchronous epoch-based batched snapshot parallel pruning."""
    if n_workers is None or n_workers <= 0:
        n_workers = min(os.cpu_count() or 4, 4)

    eps_mach = 2.220446049250313e-16
    root_order = sorted(list(H.nodes()), key=lambda v: H.degree(v), reverse=True)
    root_ptr = 0

    while root_ptr < len(root_order) and len(active) > 2 and H.number_of_edges() > 0:
        batch_roots = []
        while root_ptr < len(root_order) and len(batch_roots) < n_workers:
            r = root_order[root_ptr]
            root_ptr += 1
            if r in active and r in H:
                if lower_bank.get(r, 0.0) >= gamma / K:
                    active.discard(r)
                else:
                    batch_roots.append(r)

        if not batch_roots:
            break

        H_snap = H.copy()
        current_gamma = gamma
        current_active = set(active)
        snap_max_w = max((H_snap[a][b].get(weight, 1.0) for a, b in H_snap.edges()), default=1.0)

        def process_root(x_root: Any):
            radius = current_gamma / 2.0
            dist, hops, parent, Q, settled, tau_x = truncated_dijkstra_lex(
                H_snap, x_root, radius, current_active, node_index, weight=weight
            )
            lca_tree = DynamicLCATree(Q, parent, x_root)
            s_x = float('inf')
            local_best_len = float('inf')
            local_best_cycle = None

            for u in Q:
                iu = node_index[u]
                for v in H_snap.neighbors(u):
                    if v not in settled:
                        continue
                    iv = node_index[v]
                    if iu >= iv:
                        continue
                    if parent.get(u) == v or parent.get(v) == u:
                        continue
                    w = H_snap[u][v].get(weight, 1.0)
                    p = lca_tree.lca(u, v)
                    if p is None:
                        continue
                    l_c = dist[u] + dist[v] + w - 2.0 * dist[p]
                    sigma_e = dist[u] + dist[v] + w - dist[p]
                    if sigma_e < s_x:
                        s_x = sigma_e
                    if l_c < local_best_len:
                        cand = reconstruct_cycle(u, v, p, parent)
                        if len(cand) >= 4 and len(set(cand[:-1])) == len(cand) - 1:
                            local_best_len = l_c
                            local_best_cycle = cand

            B_x = min(2.0 * tau_x, s_x)
            local_prune = set()
            local_L = {}

            if B_x == float('inf'):
                local_prune = set(settled)
                for z in settled:
                    local_L[z] = float('inf')
            else:
                eps_guard = 8.0 * len(H_snap) * eps_mach * snap_max_w
                safe_B = max(0.0, B_x - eps_guard)
                thresh = (safe_B - current_gamma / K) / 2.0
                for z in settled:
                    L_xz = max(0.0, safe_B - 2.0 * dist[z])
                    local_L[z] = L_xz
                    if dist[z] <= thresh or L_xz >= current_gamma / K:
                        local_prune.add(z)

            return x_root, local_best_len, local_best_cycle, local_prune, local_L

        if len(batch_roots) == 1:
            results = [process_root(batch_roots[0])]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_roots)) as executor:
                results = list(executor.map(process_root, batch_roots))

        union_prune = set()
        gamma_improved = False

        for r_node, l_len, l_cyc, prunes, L_dict in results:
            if l_len < gamma:
                gamma = l_len
                best_cycle = l_cyc
                gamma_improved = True
            for z, l_val in L_dict.items():
                lower_bank[z] = max(lower_bank.get(z, 0.0), l_val)
            union_prune.update(prunes)

        for z in union_prune:
            active.discard(z)
            if z in H:
                H.remove_node(z)
        if union_prune:
            prune_low_degree(H, active)

        if gamma_improved:
            filter_heavy_edges(H, gamma, weight=weight)
            prune_low_degree(H, active)

        if gamma < float('inf'):
            cutoff = gamma / K
            retro = [v for v in list(active) if lower_bank.get(v, 0.0) >= cutoff]
            for v in retro:
                active.discard(v)
                if v in H:
                    H.remove_node(v)
            if retro:
                prune_low_degree(H, active)

    return gamma, (best_cycle if best_cycle is not None else [])


def _build_csr(H: nx.Graph, node_index: Dict[Any, int], weight: str = "weight"):
    n = len(node_index)
    indptr = np.zeros(n + 1, dtype=np.int32)
    num_edges = H.number_of_edges() * 2
    indices = np.empty(num_edges, dtype=np.int32)
    weights = np.empty(num_edges, dtype=np.float64)
    pos = 0
    adj = H._adj
    for u, iu in node_index.items():
        u_nbrs = adj.get(u)
        if u_nbrs:
            for v, data in u_nbrs.items():
                if u == v:
                    continue
                indices[pos] = node_index[v]
                weights[pos] = data.get(weight, 1.0)
                pos += 1
        indptr[iu + 1] = pos
    if pos < num_edges:
        indices = indices[:pos].copy()
        weights = weights[:pos].copy()
    return indptr, indices, weights


def _certified_pruning_accelerated(
    H: nx.Graph,
    gamma: float,
    best_cycle: List[Any],
    active: Set[Any],
    node_index: Dict[Any, int],
    weight: str = "weight",
    K: float = 2.0
) -> Tuple[float, List[Any]]:
    """Accelerated certified pruning execution via C++/Cython FastDijkstraEngine."""
    rev_index = {i: v for v, i in node_index.items()}
    n = len(node_index)

    indptr, indices, weights = _build_csr(H, node_index, weight=weight)
    engine = fast_mwc.FastDijkstraEngine(n, indptr, indices, weights)

    mask = np.zeros(n, dtype=np.uint8)
    for v in active:
        mask[node_index[v]] = 1
    engine.set_active_mask(mask)

    root_order = sorted(list(H.nodes()), key=lambda v: H.degree(v), reverse=True)
    eps_mach = 2.220446049250313e-16
    max_w = max((H[u][v].get(weight, 1.0) for u, v in H.edges()), default=1.0)

    for x in root_order:
        if x not in active or x not in H:
            continue
        ix = node_index[x]
        if engine.get_lower_cert(ix) >= gamma / K:
            active.discard(x)
            engine.deactivate_vertex(ix)
            if x in H:
                H.remove_node(x)
            removed_low = prune_low_degree(H, active)
            for v in removed_low:
                engine.deactivate_vertex(node_index[v])
            continue

        eps_guard = 8.0 * len(H) * eps_mach * max_w

        (gamma_improved, new_l_c, best_u, best_v, best_p,
         tau_x, s_x, B_x, to_discard, q_list) = engine.run_certified_iteration(
            ix, gamma, K, eps_guard
        )

        if gamma_improved and new_l_c < gamma:
            parent_arr = engine.get_parent()
            parent = {rev_index[i]: rev_index[parent_arr[i]] for i in q_list if parent_arr[i] != i and parent_arr[i] >= 0}
            u_node = rev_index[best_u]
            v_node = rev_index[best_v]
            p_node = rev_index[best_p]
            cand_cycle = reconstruct_cycle(u_node, v_node, p_node, parent)
            if len(cand_cycle) >= 4 and len(set(cand_cycle[:-1])) == len(cand_cycle) - 1:
                gamma = new_l_c
                best_cycle = cand_cycle
                num_heavy = filter_heavy_edges(H, gamma, weight=weight)
                if num_heavy > 0:
                    removed_low = prune_low_degree(H, active)
                    for v in removed_low:
                        engine.deactivate_vertex(node_index[v])
                if H.number_of_edges() > 0:
                    max_w = max((H[u][v].get(weight, 1.0) for u, v in H.edges()), default=1.0)

        for iz in to_discard:
            z = rev_index[iz]
            active.discard(z)
            if z in H:
                H.remove_node(z)
        if to_discard:
            removed_low = prune_low_degree(H, active)
            for v in removed_low:
                engine.deactivate_vertex(node_index[v])

        if gamma < float("inf"):
            cutoff = gamma / K
            retrospective = [v for v in list(active) if engine.get_lower_cert(node_index[v]) >= cutoff]
            for v in retrospective:
                active.discard(v)
                engine.deactivate_vertex(node_index[v])
                if v in H:
                    H.remove_node(v)
            if retrospective:
                removed_low = prune_low_degree(H, active)
                for v in removed_low:
                    engine.deactivate_vertex(node_index[v])

        if len(active) <= 2 or H.number_of_edges() == 0:
            break

    return gamma, (best_cycle if best_cycle is not None else [])


def certified_pruning_core(
    G: Union[nx.Graph, nx.MultiGraph],
    weight: str = 'weight',
    K: float = 2.0,
    parallel: bool = False,
    n_workers: Optional[int] = None,
    use_acceleration: bool = True
) -> Tuple[float, List[Any]]:
    """Core certified pruning driver."""
    H, gamma, best_cycle, is_done = preprocess_degeneracy(G, weight=weight)
    if is_done:
        return gamma, (best_cycle if best_cycle is not None else [])

    active = set(H.nodes())
    node_index = _get_node_index(list(H.nodes()))

    if gamma < float('inf'):
        filter_heavy_edges(H, gamma, weight=weight)
        prune_low_degree(H, active)

    if len(active) <= 2 or H.number_of_edges() == 0:
        return gamma, (best_cycle if best_cycle is not None else [])

    lower_bank: Dict[Any, float] = {v: 0.0 for v in H.nodes()}

    if parallel:
        return _certified_pruning_parallel(
            H, gamma, best_cycle, active, node_index, lower_bank, K, weight, n_workers
        )

    if use_acceleration and _HAS_ACCELERATION:
        return _certified_pruning_accelerated(
            H, gamma, best_cycle, active, node_index, weight=weight, K=K
        )

    root_order = sorted(list(H.nodes()), key=lambda v: H.degree(v), reverse=True)
    eps_mach = 2.220446049250313e-16
    max_w = max((H[u][v].get(weight, 1.0) for u, v in H.edges()), default=1.0)

    for x in root_order:
        if x not in active or x not in H:
            continue
        if lower_bank.get(x, 0.0) >= gamma / K:
            active.discard(x)
            if x in H:
                H.remove_node(x)
            prune_low_degree(H, active)
            continue

        radius = gamma / 2.0
        dist, hops, parent, Q, settled, tau_x = truncated_dijkstra_lex(
            H, x, radius, active, node_index, weight=weight
        )

        lca_tree = DynamicLCATree(Q, parent, x)
        s_x = float('inf')

        gamma_improved = False

        for u in Q:
            iu = node_index[u]
            for v in list(H.neighbors(u)):
                if v not in settled:
                    continue
                iv = node_index[v]
                if iu >= iv:
                    continue
                if parent.get(u) == v or parent.get(v) == u:
                    continue

                w = H[u][v].get(weight, 1.0)
                p = lca_tree.lca(u, v)
                if p is None:
                    continue

                l_c = dist[u] + dist[v] + w - 2.0 * dist[p]
                sigma_e = dist[u] + dist[v] + w - dist[p]

                if sigma_e < s_x:
                    s_x = sigma_e

                if l_c < gamma:
                    cand_cycle = reconstruct_cycle(u, v, p, parent)
                    if len(cand_cycle) >= 4 and len(set(cand_cycle[:-1])) == len(cand_cycle) - 1:
                        gamma = l_c
                        best_cycle = cand_cycle
                        gamma_improved = True

        if gamma_improved:
            num_heavy = filter_heavy_edges(H, gamma, weight=weight)
            if num_heavy > 0:
                prune_low_degree(H, active)
            if H.number_of_edges() > 0:
                max_w = max((H[u][v].get(weight, 1.0) for u, v in H.edges()), default=1.0)

        B_x = min(2.0 * tau_x, s_x)

        if B_x == float('inf'):
            for z in settled:
                active.discard(z)
                lower_bank[z] = float('inf')
                if z in H:
                    H.remove_node(z)
            prune_low_degree(H, active)
        else:
            eps_guard = 8.0 * len(H) * eps_mach * max_w
            safe_B = max(0.0, B_x - eps_guard)
            threshold = (safe_B - gamma / K) / 2.0

            to_discard = []
            for z in settled:
                L_xz = max(0.0, safe_B - 2.0 * dist[z])
                lower_bank[z] = max(lower_bank.get(z, 0.0), L_xz)
                if dist[z] <= threshold or lower_bank[z] >= gamma / K:
                    to_discard.append(z)

            for z in to_discard:
                active.discard(z)
                if z in H:
                    H.remove_node(z)
            if to_discard:
                prune_low_degree(H, active)

        if gamma < float('inf'):
            cutoff = gamma / K
            retrospective_discard = [v for v in list(active) if lower_bank.get(v, 0.0) >= cutoff]
            for v in retrospective_discard:
                active.discard(v)
                if v in H:
                    H.remove_node(v)
            if retrospective_discard:
                prune_low_degree(H, active)

        if len(active) <= 2 or H.number_of_edges() == 0:
            break

    return gamma, (best_cycle if best_cycle is not None else [])


# ---------------------------------------------------------------------------
# Exact Edge-Elimination Oracle
# ---------------------------------------------------------------------------

def exact_oracle(
    G: Union[nx.Graph, nx.MultiGraph],
    weight: str = 'weight'
) -> Tuple[float, List[Any]]:
    """Exact edge-elimination oracle for minimum-weight cycle."""
    if G.is_directed():
        raise ValueError("Input graph must be undirected")

    H, gamma_0, cycle_0, is_done = preprocess_degeneracy(G, weight=weight)
    if is_done:
        return gamma_0, (cycle_0 if cycle_0 is not None else [])

    best_len = gamma_0
    best_cycle = cycle_0

    for u, v, d in list(H.edges(data=True)):
        w = d.get(weight, 1.0)
        H.remove_edge(u, v)
        try:
            path = nx.shortest_path(H, u, v, weight=weight)
            path_len = sum(H[path[i]][path[i + 1]][weight] for i in range(len(path) - 1))
            total_len = path_len + w
            if total_len < best_len:
                best_len = total_len
                best_cycle = path + [u]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        H.add_edge(u, v, **d)

    return best_len, (best_cycle if best_cycle is not None else [])


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------

def minimum_weight_cycle(
    G: Union[nx.Graph, nx.MultiGraph],
    weight: str = 'weight',
    method: str = 'certified_pruning',
    K: float = 2.0,
    parallel: bool = False,
    n_workers: Optional[int] = None,
    use_acceleration: bool = True
) -> Tuple[float, List[Any]]:
    """
    Finds the minimum-weight cycle in an undirected edge-weighted multigraph or simple graph.
    Returns:
        (cycle_length, cycle_nodes) where cycle_nodes is an ordered list of vertices [u, v, ..., u].
        If acyclic, returns (float('inf'), []).
    """
    if not isinstance(G, (nx.Graph, nx.MultiGraph)):
        raise TypeError("Input must be an undirected NetworkX graph")
    if G.is_directed():
        raise ValueError("Input graph must be undirected")
    if K < 1.0:
        raise ValueError("Approximation factor K must be >= 1.0")
    if method not in ('certified_pruning', 'dijkstra', 'exact_oracle'):
        raise ValueError(f"Unknown method: {method}")

    if method == 'exact_oracle' or method == 'dijkstra':
        return exact_oracle(G, weight=weight)

    return certified_pruning_core(
        G, weight=weight, K=K, parallel=parallel, n_workers=n_workers, use_acceleration=use_acceleration
    )


def sota_shortest_cycle(
    G: Union[nx.Graph, nx.MultiGraph],
    use_fib_heap: bool = False,
    use_lca: bool = True,
    return_stats: bool = False,
    K: float = 1.0,
    weight: str = 'weight'
) -> Union[Optional[float], Tuple[Optional[float], Dict[str, Any]]]:
    """
    Backward-compatible drop-in replacement for legacy sota_shortest_cycle.
    Returns float or None if acyclic (or (float, stats) if return_stats=True).
    """
    start_time = time.time()
    if not isinstance(G, (nx.Graph, nx.MultiGraph)):
        raise TypeError("Input must be an undirected NetworkX graph")
    if G.is_directed():
        raise ValueError("Input graph must be undirected")

    if not use_lca:
        return traditional_shortest_cycle(G, return_stats)

    length, cycle = certified_pruning_core(G, weight=weight, K=K)
    res = None if length == float('inf') else float(length)

    if return_stats:
        stats = {
            'execution_time': time.time() - start_time,
            'gamma': res,
            'cycle_found': cycle is not None and len(cycle) > 0,
            'total_operations': 0,
            'dijkstra_calls': 0,
            'nodes_processed': 0,
            'edges_checked': 0,
            'nodes_pruned': 0,
            'lca_trees_built': 0,
            'lca_stats': []
        }
        return res, stats
    return res


def traditional_shortest_cycle(G: nx.Graph, return_stats: bool = False):
    """Traditional edge-elimination approach for shortest cycle."""
    if not isinstance(G, (nx.Graph, nx.MultiGraph)):
        raise TypeError("Input must be an undirected NetworkX graph")
    if G.is_directed():
        raise ValueError("Input graph must be undirected")

    start_time = time.time()
    length, cycle = exact_oracle(G)
    res = None if length == float('inf') else float(length)

    if return_stats:
        stats = {
            'total_operations': 0,
            'edges_checked': G.number_of_edges(),
            'path_computations': 0,
            'execution_time': time.time() - start_time
        }
        return res, stats
    return res


def shortest_cycle_nodes(G: Union[nx.Graph, nx.MultiGraph], weight: str = 'weight') -> Optional[List[Any]]:
    """Returns list of nodes forming shortest cycle [u, v, ..., u], or None if acyclic."""
    if not isinstance(G, (nx.Graph, nx.MultiGraph)):
        raise TypeError("Input must be an undirected NetworkX graph")
    if G.is_directed():
        raise ValueError("Input graph must be undirected")

    length, cycle = certified_pruning_core(G, weight=weight, K=1.0)
    if length == float('inf') or not cycle:
        return None
    return list(cycle)


def visualize_shortest_cycle(G, cycle=None):
    """Visualizes graph and highlights cycle if provided."""
    try:
        import matplotlib.pyplot as plt
        H = G.copy()
        pos = nx.spring_layout(H, seed=42)
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(H, pos, node_size=500, node_color='lightblue')
        nx.draw_networkx_labels(H, pos)
        edge_labels = {(u, v): f"{d.get('weight', 1.0):.1f}" for u, v, d in H.edges(data=True)}
        nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels)
        nx.draw_networkx_edges(H, pos, width=1.0, alpha=0.5)
        if cycle:
            cycle_edges = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]
            nx.draw_networkx_edges(H, pos, edgelist=cycle_edges, width=3.0, edge_color='red')
        plt.axis('off')
        plt.tight_layout()
        return plt.gcf()
    except ImportError:
        return None


def run_with_profiling(graph_size=20, use_fib_heap=False, use_lca=True):
    """Run with profiling for legacy compatibility."""
    import cProfile
    G = create_grid_graph(graph_size)
    profiler = cProfile.Profile()
    profiler.enable()
    result, stats = sota_shortest_cycle(G, use_fib_heap=use_fib_heap, use_lca=use_lca, return_stats=True)
    profiler.disable()
    print(f"Shortest cycle length: {result}")
    return result, stats
