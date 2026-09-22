#include "mwc_kernel.hpp"
#include <iostream>

FastDijkstraKernel::FastDijkstraKernel(int num_vertices,
                                       const int* ptr_indptr,
                                       const int* ptr_indices,
                                       const double* ptr_weights,
                                       int num_edges)
    : n(num_vertices), max_edges(num_edges) {
    
    indptr.assign(ptr_indptr, ptr_indptr + n + 1);
    indices.assign(ptr_indices, ptr_indices + max_edges);
    weights.assign(ptr_weights, ptr_weights + max_edges);
    active.assign(n, 1);
    lower_bank.assign(n, 0.0);

    max_log = 1;
    while ((1 << max_log) <= (n + 2)) {
        max_log++;
    }
    max_log += 1;

    dist.resize(n, std::numeric_limits<double>::infinity());
    hops.resize(n, std::numeric_limits<int>::max());
    parent.resize(n, -1);
    settled.resize(n, 0);
    depth.resize(n, 0);
    up.resize(max_log * n, 0);
    Q.reserve(n);
}

void FastDijkstraKernel::set_active_mask(const uint8_t* mask) {
    for (int i = 0; i < n; ++i) {
        active[i] = mask[i];
    }
}

void FastDijkstraKernel::deactivate_vertex(int v) {
    if (v >= 0 && v < n) {
        active[v] = 0;
    }
}

void FastDijkstraKernel::run_truncated(int root, double radius) {
    for (int u : Q) {
        dist[u] = std::numeric_limits<double>::infinity();
        hops[u] = std::numeric_limits<int>::max();
        parent[u] = -1;
        settled[u] = 0;
    }
    Q.clear();

    dist[root] = 0.0;
    hops[root] = 0;
    parent[root] = root;

    std::priority_queue<DijkstraNode, std::vector<DijkstraNode>, std::greater<DijkstraNode>> pq;
    pq.push({0.0, 0, root, root});

    bool has_cross_edges = false;

    while (!pq.empty()) {
        DijkstraNode top = pq.top();
        pq.pop();
        int u = top.u;
        double d = top.dist;
        int h = top.hops;

        if (settled[u]) continue;

        if (d >= radius) {
            has_cross_edges = true;
            break;
        }

        settled[u] = 1;
        Q.push_back(u);

        int start_edge = indptr[u];
        int end_edge = indptr[u + 1];

        for (int e = start_edge; e < end_edge; ++e) {
            int v = indices[e];
            if (!active[v] || settled[v]) continue;

            double w = weights[e];
            double cand_d = d + w;
            int cand_h = h + 1;
            int cand_par = u;

            bool improves = false;
            if (cand_d < dist[v]) {
                improves = true;
            } else if (cand_d == dist[v]) {
                if (cand_h < hops[v]) {
                    improves = true;
                } else if (cand_h == hops[v]) {
                    if (parent[v] == -1 || cand_par < parent[v]) {
                        improves = true;
                    }
                }
            }

            if (improves) {
                dist[v] = cand_d;
                hops[v] = cand_h;
                parent[v] = cand_par;
                if (cand_d < radius) {
                    pq.push({cand_d, cand_h, cand_par, v});
                } else {
                    has_cross_edges = true;
                }
            }
        }
    }

    if (!has_cross_edges) {
        for (int u : Q) {
            int start_edge = indptr[u];
            int end_edge = indptr[u + 1];
            for (int e = start_edge; e < end_edge; ++e) {
                int v = indices[e];
                if (active[v] && !settled[v]) {
                    has_cross_edges = true;
                    break;
                }
            }
            if (has_cross_edges) break;
        }
    }
}

void FastDijkstraKernel::build_lca(int root) {
    for (int u : Q) {
        if (u == root) {
            depth[u] = 0;
            up[0 * n + u] = root;
        } else {
            int p = parent[u];
            depth[u] = (p >= 0) ? depth[p] + 1 : 0;
            up[0 * n + u] = (p >= 0) ? p : u;
        }
    }

    for (int j = 1; j < max_log; ++j) {
        int prev_offset = (j - 1) * n;
        int cur_offset = j * n;
        for (int u : Q) {
            int anc = up[prev_offset + u];
            up[cur_offset + u] = up[prev_offset + anc];
        }
    }
}

int FastDijkstraKernel::query_lca(int u, int v) const {
    if (u == v) return u;
    int du = depth[u];
    int dv = depth[v];
    if (du < dv) {
        std::swap(u, v);
        std::swap(du, dv);
    }

    int diff = du - dv;
    for (int j = 0; diff > 0 && j < max_log; ++j) {
        if (diff & (1 << j)) {
            u = up[j * n + u];
            diff &= ~(1 << j);
        }
    }

    if (u == v) return u;

    for (int j = max_log - 1; j >= 0; --j) {
        if (up[j * n + u] != up[j * n + v]) {
            u = up[j * n + u];
            v = up[j * n + v];
        }
    }

    return up[0 * n + u];
}

double FastDijkstraKernel::query_path_dist(int u, int v, int lca) const {
    return dist[u] + dist[v] - 2.0 * dist[lca];
}

IterationResult FastDijkstraKernel::run_certified_iteration(int root, double gamma, double K, double eps_guard) {
    double radius = gamma / 2.0;
    run_truncated(root, radius);
    build_lca(root);

    IterationResult res;
    res.q_size = static_cast<int>(Q.size());
    res.best_l_c = gamma;
    res.gamma_improved = false;
    res.best_u = -1;
    res.best_v = -1;
    res.best_p = -1;
    res.s_x = std::numeric_limits<double>::infinity();

    bool has_cross = false;
    for (int u : Q) {
        int start_edge = indptr[u];
        int end_edge = indptr[u + 1];
        for (int e = start_edge; e < end_edge; ++e) {
            int v = indices[e];
            if (active[v] && !settled[v]) {
                has_cross = true;
                break;
            }
        }
        if (has_cross) break;
    }
    res.tau_x = has_cross ? radius : std::numeric_limits<double>::infinity();

    for (int u : Q) {
        int start_edge = indptr[u];
        int end_edge = indptr[u + 1];
        for (int e = start_edge; e < end_edge; ++e) {
            int v = indices[e];
            if (u >= v || !settled[v]) continue;
            if (parent[u] == v || parent[v] == u) continue;

            double w = weights[e];
            int p = query_lca(u, v);
            double l_c = dist[u] + dist[v] + w - 2.0 * dist[p];
            double sigma_e = dist[u] + dist[v] + w - dist[p];

            if (sigma_e < res.s_x) {
                res.s_x = sigma_e;
            }

            if (l_c < res.best_l_c) {
                res.best_l_c = l_c;
                res.best_u = u;
                res.best_v = v;
                res.best_p = p;
                res.gamma_improved = true;
            }
        }
    }

    res.B_x = std::min(2.0 * res.tau_x, res.s_x);

    // Accumulate lower certificates & prune
    double cutoff = gamma / K;
    if (std::isinf(res.B_x)) {
        for (int z : Q) {
            lower_bank[z] = std::numeric_limits<double>::infinity();
            active[z] = 0;
            res.to_discard.push_back(z);
        }
    } else {
        double safe_B = std::max(0.0, res.B_x - eps_guard);
        double threshold = (safe_B - cutoff) / 2.0;
        for (int z : Q) {
            double L_xz = std::max(0.0, safe_B - 2.0 * dist[z]);
            if (L_xz > lower_bank[z]) {
                lower_bank[z] = L_xz;
            }
            if (dist[z] <= threshold || lower_bank[z] >= cutoff) {
                active[z] = 0;
                res.to_discard.push_back(z);
            }
        }
    }

    return res;
}
