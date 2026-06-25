// track_a_impl.cpp
// Track A batch controller: medoid-block BF16 GEMM at hop 0
// NJIT HSRI 2026 — Frederick Rajakumar
//
// Defines Index<T,TagT,LabelT>::search_batch_track_a.
// Explicitly instantiated for <float, uint32_t, uint32_t> (float path; used
// for both SIFT1M and GIST1M via BF16).
// Compiled separately and linked with libdiskann.a; do NOT add to CMakeLists.

#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <limits>
#include <shared_mutex>
#include <vector>
#include <omp.h>
#include <mkl.h>

#include "index.h"
#include "scratch.h"
#include "neighbor.h"
#include "defaults.h"   // MAX_POINTS_FOR_USING_BITSET
#define MAX_POINTS_FOR_USING_BITSET 10000000

template <typename T, typename TagT, typename LabelT>
void diskann::Index<T, TagT, LabelT>::search_batch_track_a(
    const T  *queries,      // N x aligned_dim, row-major
    size_t    N,            // query count
    size_t    K,            // top-K
    uint32_t  L,            // beam width
    uint32_t  num_threads,  // OMP threads for Phase 2
    size_t    aligned_dim,  // stride between queries (elements)
    uint32_t *indices)      // output N x K
{
    if (N == 0) return;

    // Optional phase timing — set TRACK_A_TIMING=1 to enable. Off by default
    // so benchmark timed regions stay clean.
    static const bool ta_timing = (std::getenv("TRACK_A_TIMING") != nullptr);
    std::chrono::high_resolution_clock::time_point _t0, _t1, _t2;
    if (ta_timing) _t0 = std::chrono::high_resolution_clock::now();

    // -------------------------------------------------------------------
    // Phase 1: one BF16 GEMM for all N queries x R medoid neighbors
    // -------------------------------------------------------------------

    const size_t D = _data_store->get_aligned_dim();

    // Get medoid neighbor ids under lock (same pattern as iterate_to_fixed_point)
    _locks[_start].lock();
    std::vector<uint32_t> nbr_ids(_graph_store->get_neighbours(_start));
    _locks[_start].unlock();
    const uint32_t R = (uint32_t)nbr_ids.size();
    if (R == 0) return;  // degenerate index guard

    // Extract neighbor vectors (R x D) and precompute ||x||^2
    std::vector<float> x_vecs(R * D, 0.0f);
    std::vector<float> x_norms(R, 0.0f);
    for (uint32_t j = 0; j < R; j++) {
        _data_store->get_vector(nbr_ids[j], x_vecs.data() + j * D);
        const float *v = x_vecs.data() + j * D;
        float s = 0.0f;
        for (size_t d = 0; d < D; d++) s += v[d] * v[d];
        x_norms[j] = s;
    }

    // Compute ||q_i||^2 for each query (parallel — N is large)
    std::vector<float> q_norms(N, 0.0f);
    omp_set_num_threads((int)num_threads);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)N; i++) {
        const T *q = queries + i * aligned_dim;
        float s = 0.0f;
        for (size_t d = 0; d < D; d++) s += (float)q[d] * (float)q[d];
        q_norms[i] = s;
    }

    // BF16 conversion: truncate low 16 bits of IEEE 754 representation
    auto to_bf16 = [](float f) -> MKL_BF16 {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(float));
        return static_cast<MKL_BF16>(u >> 16);
    };

    // Build BF16 matrices: A = queries (N x D), B = neighbors (R x D)
    std::vector<MKL_BF16> A_bf16(N * D);
    std::vector<MKL_BF16> B_bf16(R * D);

    // Query block is the bulk of conversion work — parallelize it.
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)N; i++) {
        const T *q = queries + i * aligned_dim;
        for (size_t d = 0; d < D; d++)
            A_bf16[i * D + d] = to_bf16((float)q[d]);
    }
    // Neighbor block is tiny (R rows) — serial is fine.
    for (uint32_t j = 0; j < R; j++) {
        const float *v = x_vecs.data() + j * D;
        for (size_t d = 0; d < D; d++)
            B_bf16[j * D + d] = to_bf16(v[d]);
    }

    // GEMM: C[i][j] = dot(q_i, x_j)   shape: N x R
    // C = A * B^T,  A: N x D,  B: R x D
    std::vector<float> C(N * R, 0.0f);
    cblas_gemm_bf16bf16f32(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        (MKL_INT)N, (MKL_INT)R, (MKL_INT)D,
        1.0f,
        A_bf16.data(), (MKL_INT)D,
        B_bf16.data(), (MKL_INT)D,
        0.0f, C.data(), (MKL_INT)R);

    // Scatter to full L2 distances (parallel — N rows):
    //   dist[i][j] = ||q_i||^2 + ||x_j||^2 - 2 * dot(q_i, x_j)
    // Same formula as DistanceL2Float::compare — same scale as hop-1+ distances.
    std::vector<float> dist_mat(N * R);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)N; i++) {
        const float *ci = C.data()        + i * R;
        float       *di = dist_mat.data() + i * R;
        for (uint32_t j = 0; j < R; j++)
            di[j] = q_norms[i] + x_norms[j] - 2.0f * ci[j];
    }

    // -------------------------------------------------------------------
    // Phase 2 (parallel): seeded greedy traversal from hop 1+
    // Each thread seeds its priority queue with the GEMM distances and
    // runs the standard while loop from iterate_to_fixed_point, skipping
    // hop 0 (the medoid expansion that Phase 1 already did).
    // -------------------------------------------------------------------

    const auto   total_pts = _max_points + _num_frozen_pts;
    const bool   fast_iter = (total_pts <= MAX_POINTS_FOR_USING_BITSET);

    if (ta_timing) _t1 = std::chrono::high_resolution_clock::now();

    // Shared (read) lock — same as Index::search()
    std::shared_lock<std::shared_timed_mutex> idx_lock(_update_lock);

    omp_set_num_threads((int)num_threads);
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t qi = 0; qi < (int64_t)N; qi++) {

        // Get per-thread scratch from pool (cleared by ScratchStoreManager dtor)
        ScratchStoreManager<InMemQueryScratch<T>> mgr(_query_scratch);
        InMemQueryScratch<T> *sc = mgr.scratch_space();
        if (L > sc->get_L()) sc->resize_for_new_L(L);

        // Copy and preprocess query (same as iterate_to_fixed_point)
        T *aq = sc->aligned_query();
        std::memcpy(aq, queries + qi * aligned_dim, aligned_dim * sizeof(T));
        _pq_data_store->preprocess_query(aq, sc);

        // Visited-set setup (mirrors iterate_to_fixed_point exactly)
        auto &bs = sc->inserted_into_pool_bs();
        auto &rs = sc->inserted_into_pool_rs();
        if (fast_iter && bs.size() < total_pts) {
            size_t rsz = (2 * total_pts > MAX_POINTS_FOR_USING_BITSET)
                         ? MAX_POINTS_FOR_USING_BITSET
                         : 2 * total_pts;
            bs.resize(rsz);
        }

        auto mark_visited = [&](uint32_t id) {
            if (fast_iter) bs[id] = 1;
            else           rs.insert(id);
        };
        auto unvisited = [&](uint32_t id) -> bool {
            return fast_iter ? (bs[id] == 0)
                             : (rs.find(id) == rs.end());
        };

        // Mark medoid as visited — Phase 1 already expanded it
        mark_visited(_start);

        // Seed priority queue with hop-0 GEMM distances
        NeighborPriorityQueue &pq = sc->best_l_nodes();
        pq.clear();
        pq.reserve(L);
        const float *row = dist_mat.data() + qi * R;
        for (uint32_t j = 0; j < R; j++) {
            uint32_t id = nbr_ids[j];
            if (id >= total_pts) continue;
            mark_visited(id);
            pq.insert(Neighbor(id, row[j]));
        }

        // Greedy traversal (hop 1+) — static index branch of iterate_to_fixed_point
        auto &ids  = sc->id_scratch();
        auto &dsts = sc->dist_scratch();
        ids.clear();
        dsts.clear();

        while (pq.has_unexpanded_node()) {
            auto     nbr = pq.closest_unexpanded();
            uint32_t n   = nbr.id;

            ids.clear();
            dsts.clear();

            _locks[n].lock();
            auto nbrs = _graph_store->get_neighbours(n);
            _locks[n].unlock();

            for (uint32_t id : nbrs) {
                if (id < total_pts && unvisited(id))
                    ids.push_back(id);
            }
            for (uint32_t id : ids) mark_visited(id);

            // Full-precision L2 via _pq_data_store (== _data_store when PQ disabled)
            // Distances are on the same scale as the GEMM scatter above.
            _pq_data_store->get_distance(aq, ids, dsts, sc);

            for (size_t m = 0; m < ids.size(); m++)
                pq.insert(Neighbor(ids[m], dsts[m]));
        }

        // Extract top-K (pq is sorted best-first)
        uint32_t *out = indices + qi * K;
        size_t    pos = 0;
        for (size_t k = 0; k < pq.size() && pos < K; k++) {
            if (pq[k].id < _max_points)
                out[pos++] = pq[k].id;
        }
        // Pad if fewer than K valid results (shouldn't happen with L >= K)
        while (pos < K) out[pos++] = std::numeric_limits<uint32_t>::max();
    }

    if (ta_timing) {
        _t2 = std::chrono::high_resolution_clock::now();
        double p1 = std::chrono::duration<double, std::milli>(_t1 - _t0).count();
        double p2 = std::chrono::duration<double, std::milli>(_t2 - _t1).count();
        double tot = p1 + p2;
        std::fprintf(stderr,
            "[TRACK_A_TIMING] Phase1(setup+GEMM)=%.3f ms (%.2f%%)  "
            "Phase2(traversal)=%.3f ms (%.2f%%)  total=%.3f ms\n",
            p1, 100.0 * p1 / tot, p2, 100.0 * p2 / tot, tot);
    }
}

// Explicit instantiation — BF16 path, float only (GIST1M primary target).
// SIFT1M (uint8) would need INT8 GEMM; that is future work.
template void diskann::Index<float, uint32_t, uint32_t>::search_batch_track_a(
    const float *, size_t, size_t, uint32_t, uint32_t, size_t, uint32_t *);
