#ifndef MWC_KERNEL_HPP
#define MWC_KERNEL_HPP

#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdint>

struct DijkstraNode {
    double dist;
    int hops;
    int parent_idx;
    int u;

    bool operator>(const DijkstraNode& o) const {
        if (dist != o.dist) return dist > o.dist;
        if (hops != o.hops) return hops > o.hops;
        if (parent_idx != o.parent_idx) return parent_idx > o.parent_idx;
        return u > o.u;
    }
};

struct IterationResult {
    double tau_x;
    double s_x;
    double B_x;
    bool gamma_improved;
    double best_l_c;
    int best_u;
    int best_v;
    int best_p;
    int q_size;
    std::vector<int> to_discard;
};

class FastDijkstraKernel {
public:
    int n;
    int max_edges;
    int max_log;

    // Graph CSR representation
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> weights;
    std::vector<uint8_t> active;

    // Lower certificate accumulation bank
    std::vector<double> lower_bank;

    // Preallocated buffers for Dijkstra & LCA
    std::vector<double> dist;
    std::vector<int> hops;
    std::vector<int> parent;
    std::vector<uint8_t> settled;
    std::vector<int> Q;
    std::vector<int> depth;
    std::vector<int> up; // size: max_log * n

    FastDijkstraKernel(int num_vertices,
                       const int* ptr_indptr,
                       const int* ptr_indices,
                       const double* ptr_weights,
                       int num_edges);

    void set_active_mask(const uint8_t* mask);
    void deactivate_vertex(int v);
    bool is_active(int v) const { return v >= 0 && v < n && active[v]; }
    double get_lower_cert(int v) const { return (v >= 0 && v < n) ? lower_bank[v] : 0.0; }

    // Truncated Dijkstra core
    void run_truncated(int root, double radius);

    // Dynamic Binary-Lifting LCA
    void build_lca(int root);
    int query_lca(int u, int v) const;
    double query_path_dist(int u, int v, int lca) const;

    // Full certified pruning root iteration with certificate updates
    IterationResult run_certified_iteration(int root, double gamma, double K, double eps_guard);
};

#endif // MWC_KERNEL_HPP
