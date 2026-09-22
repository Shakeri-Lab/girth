# distutils: language = c++
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint8_t
from libcpp.vector cimport vector

cnp.import_array()

cdef extern from "mwc_kernel.hpp":
    struct IterationResult:
        double tau_x
        double s_x
        double B_x
        bint gamma_improved
        double best_l_c
        int best_u
        int best_v
        int best_p
        int q_size
        vector[int] to_discard

    cdef cppclass FastDijkstraKernel:
        int n
        vector[double] dist
        vector[int] hops
        vector[int] parent
        vector[uint8_t] settled
        vector[int] Q

        FastDijkstraKernel(int num_vertices,
                           const int* ptr_indptr,
                           const int* ptr_indices,
                           const double* ptr_weights,
                           int num_edges) except +
        void set_active_mask(const uint8_t* mask)
        void deactivate_vertex(int v)
        bint is_active(int v) const
        double get_lower_cert(int v) const
        void run_truncated(int root, double radius)
        void build_lca(int root)
        int query_lca(int u, int v) const
        double query_path_dist(int u, int v, int lca) const
        IterationResult run_certified_iteration(int root, double gamma, double K, double eps_guard)

cdef class FastDijkstraEngine:
    cdef FastDijkstraKernel* kernel
    cdef int n_vertices

    def __cinit__(self, int n_vertices, const int[::1] indptr, const int[::1] indices, const double[::1] weights):
        self.n_vertices = n_vertices
        cdef int num_edges = indices.shape[0]
        self.kernel = new FastDijkstraKernel(n_vertices, &indptr[0], &indices[0], &weights[0], num_edges)

    def __dealloc__(self):
        if self.kernel != NULL:
            del self.kernel
            self.kernel = NULL

    cpdef void set_active_mask(self, const uint8_t[::1] mask):
        self.kernel.set_active_mask(&mask[0])

    cpdef void deactivate_vertex(self, int v):
        self.kernel.deactivate_vertex(v)

    cpdef bint is_active(self, int v):
        return self.kernel.is_active(v)

    cpdef double get_lower_cert(self, int v):
        return self.kernel.get_lower_cert(v)

    cpdef void run_truncated(self, int root, double radius, object adj_list=None):
        self.kernel.run_truncated(root, radius)
        self.kernel.build_lca(root)

    cpdef int query_lca(self, int u, int v):
        return self.kernel.query_lca(u, v)

    cpdef double query_path_dist(self, int u, int v, int lca):
        return self.kernel.query_path_dist(u, v, lca)

    cpdef tuple run_certified_iteration(self, int root, double gamma, double K, double eps_guard):
        cdef IterationResult res = self.kernel.run_certified_iteration(root, gamma, K, eps_guard)
        cdef list to_discard = [res.to_discard[i] for i in range(res.to_discard.size())]
        cdef list q_list = [self.kernel.Q[i] for i in range(res.q_size)]
        return (
            res.gamma_improved,
            res.best_l_c,
            res.best_u,
            res.best_v,
            res.best_p,
            res.tau_x,
            res.s_x,
            res.B_x,
            to_discard,
            q_list
        )

    cpdef object get_dist(self):
        cdef double[:] view = <double[:self.n_vertices]>&self.kernel.dist[0]
        return np.asarray(view).copy()

    cpdef object get_hops(self):
        cdef int[:] view = <int[:self.n_vertices]>&self.kernel.hops[0]
        return np.asarray(view).copy()

    cpdef object get_parent(self):
        cdef int[:] view = <int[:self.n_vertices]>&self.kernel.parent[0]
        return np.asarray(view).copy()

    cpdef object get_settled(self):
        cdef uint8_t[:] view = <uint8_t[:self.n_vertices]>&self.kernel.settled[0]
        return np.asarray(view).copy()

    cpdef list get_Q(self):
        return [self.kernel.Q[i] for i in range(self.kernel.Q.size())]

FastDijkstraTree = FastDijkstraEngine
