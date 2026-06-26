// medoid_batch_benchmark.cpp
// Benchmark driver: baseline per-query search vs the medoid hop-0 batch controller
// NJIT HSRI 2026 — Frederick Rajakumar
//
// Measures QPS and Recall@K for both paths on the same index/queries.
// arch_prctl is called at startup — required for AMX tile registers.

#include <sys/syscall.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <boost/program_options.hpp>

#include "index.h"
#include "utils.h"
#include "index_factory.h"

#define ARCH_REQ_XCOMP_PERM  0x1023
#define XFEATURE_XTILEDATA   18

namespace po = boost::program_options;

using CIndex = diskann::Index<float, uint32_t, uint32_t>;

// ------------------------------------------------------------------
// Baseline: per-query search, same OMP pattern as search_memory_index
// ------------------------------------------------------------------
double run_baseline(CIndex *idx,
                    const float *queries, size_t N, size_t K, uint32_t L,
                    uint32_t T, size_t aligned_dim,
                    std::vector<uint32_t> &out)
{
    omp_set_num_threads((int)T);
    auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t)N; i++) {
        idx->search(queries + i * aligned_dim, K, L,
                    out.data() + i * K, (float *)nullptr);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// ------------------------------------------------------------------
// Medoid hop-0 batching: one BF16 GEMM at hop 0, then per-query traversal from hop 1
// ------------------------------------------------------------------
double run_medoid_hop0(CIndex *idx,
                   const float *queries, size_t N, size_t K, uint32_t L,
                   uint32_t T, size_t aligned_dim,
                   std::vector<uint32_t> &out)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    idx->search_batch_medoid_hop0(queries, N, K, L, T, aligned_dim, out.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// ------------------------------------------------------------------
// Recall@K computation (set intersection, top-K of GT)
// ------------------------------------------------------------------
double calc_recall(const uint32_t *result, const uint32_t *gt,
                   size_t N, size_t K, size_t gt_dim)
{
    size_t correct = 0;
    for (size_t i = 0; i < N; i++) {
        std::set<uint32_t> gt_set(gt + i * gt_dim, gt + i * gt_dim + K);
        for (size_t k = 0; k < K; k++)
            if (gt_set.count(result[i * K + k])) correct++;
    }
    return 100.0 * (double)correct / (double)(N * K);
}

int main(int argc, char *argv[])
{
    // ----------------------------------------------------------------
    // Enable AMX tile registers — MUST happen before any BF16 GEMM.
    // Without this, cblas_gemm_bf16bf16f32 silently routes to AVX-512.
    // ----------------------------------------------------------------
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) {
        std::cerr << "ERROR: arch_prctl XFEATURE_XTILEDATA failed\n";
        return 1;
    }
    std::cout << "AMX XTILEDATA enabled\n";

    // ----------------------------------------------------------------
    // Arguments
    // ----------------------------------------------------------------
    std::string index_path, query_file, gt_file;
    uint32_t K, L, T;

    po::options_description desc("Medoid hop-0 batch controller benchmark");
    desc.add_options()
        ("index_path_prefix", po::value<std::string>(&index_path)->required(),
         "Index path prefix (same as search_memory_index)")
        ("query_file", po::value<std::string>(&query_file)->required(),
         "Query .bin file")
        ("gt_file",    po::value<std::string>(&gt_file)->required(),
         "Ground truth .bin file")
        ("K", po::value<uint32_t>(&K)->default_value(10),  "Top-K")
        ("L", po::value<uint32_t>(&L)->default_value(100), "Beam width")
        ("T", po::value<uint32_t>(&T)->default_value(32),  "OMP threads");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n" << desc << "\n";
        return 1;
    }

    // ----------------------------------------------------------------
    // Load queries
    // ----------------------------------------------------------------
    float   *queries = nullptr;
    size_t   query_num, query_dim, query_aligned_dim;
    diskann::load_aligned_bin<float>(query_file, queries,
                                     query_num, query_dim, query_aligned_dim);
    std::cout << "Queries: " << query_num
              << "  dim=" << query_dim
              << "  aligned_dim=" << query_aligned_dim << "\n";

    // ----------------------------------------------------------------
    // Load ground truth
    // ----------------------------------------------------------------
    uint32_t *gt_ids   = nullptr;
    float    *gt_dists = nullptr;
    size_t    gt_num, gt_dim;
    diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
    if (gt_num != query_num) {
        std::cerr << "ERROR: query_num=" << query_num
                  << " gt_num=" << gt_num << " mismatch\n";
        return 1;
    }

    // ----------------------------------------------------------------
    // Build and load index (L2, float, static, no PQ, no tags)
    // ----------------------------------------------------------------
    size_t num_frozen = diskann::get_graph_num_frozen_points(index_path);

    auto config = diskann::IndexConfigBuilder()
        .with_metric(diskann::Metric::L2)
        .with_dimension(query_dim)
        .with_max_points(0)
        .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
        .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
        .with_data_type("float")
        .with_label_type("uint32")
        .with_tag_type("uint32")
        .is_dynamic_index(false)
        .is_enable_tags(false)
        .is_concurrent_consolidate(false)
        .is_pq_dist_build(false)
        .is_use_opq(false)
        .with_num_pq_chunks(0)
        .with_num_frozen_pts(num_frozen)
        .build();

    auto factory = diskann::IndexFactory(config);
    auto index   = factory.create_instance();
    index->load(index_path.c_str(), T, L);
    std::cout << "Index loaded\n";

    // Cast to concrete type to call typed methods directly
    auto *idx = dynamic_cast<CIndex *>(index.get());
    if (!idx) {
        std::cerr << "ERROR: dynamic_cast to Index<float,uint32_t,uint32_t> failed\n";
        return 1;
    }

    // ----------------------------------------------------------------
    // Warmup (not timed — fills caches, JIT-compiles OMP, warms MKL)
    // ----------------------------------------------------------------
    std::vector<uint32_t> result_base(query_num * K, 0);
    std::vector<uint32_t> result_ta  (query_num * K, 0);

    run_baseline(idx, queries, query_num, K, L, T, query_aligned_dim, result_base);
    run_medoid_hop0 (idx, queries, query_num, K, L, T, query_aligned_dim, result_ta);

    // ----------------------------------------------------------------
    // Timed runs — interleaved repeated measurement with mean +/- stddev.
    // Interleaving (base, ta, base, ta, ...) so any drift hits both paths
    // equally. Reports QPS mean/stddev and a noise-aware verdict.
    // Override iteration count with MEDOID_ITERS (default 15).
    // ----------------------------------------------------------------
    int iters = 15;
    if (const char *e = std::getenv("MEDOID_ITERS")) {
        int v = std::atoi(e);
        if (v > 1) iters = v;
    }

    std::vector<double> qps_base_samples, qps_ta_samples;
    qps_base_samples.reserve(iters);
    qps_ta_samples.reserve(iters);

    for (int r = 0; r < iters; r++) {
        double tb = run_baseline(idx, queries, query_num, K, L, T, query_aligned_dim, result_base);
        double tt = run_medoid_hop0 (idx, queries, query_num, K, L, T, query_aligned_dim, result_ta);
        qps_base_samples.push_back((double)query_num / tb);
        qps_ta_samples.push_back((double)query_num / tt);
    }

    auto mean_stddev = [](const std::vector<double> &v, double &mean, double &sd) {
        double s = 0.0;
        for (double x : v) s += x;
        mean = s / v.size();
        double var = 0.0;
        for (double x : v) var += (x - mean) * (x - mean);
        sd = std::sqrt(var / (v.size() - 1));  // sample stddev
    };

    double qps_base, sd_base, qps_ta, sd_ta;
    mean_stddev(qps_base_samples, qps_base, sd_base);
    mean_stddev(qps_ta_samples,   qps_ta,   sd_ta);

    double rec_base = calc_recall(result_base.data(), gt_ids, query_num, K, gt_dim);
    double rec_ta   = calc_recall(result_ta.data(),   gt_ids, query_num, K, gt_dim);

    double speedup = qps_ta / qps_base;
    // Relative noise: combined coefficient of variation of the two means.
    double cv_base = sd_base / qps_base;
    double cv_ta   = sd_ta   / qps_ta;
    double noise_band = std::sqrt(cv_base * cv_base + cv_ta * cv_ta); // ~rel. sigma of ratio

    // ----------------------------------------------------------------
    // Results
    // ----------------------------------------------------------------
    std::cout << "\n=== Medoid hop-0 batching results (L=" << L
              << " K=" << K << " T=" << T
              << ", " << iters << " iters) ===\n";
    std::cout.precision(2);
    std::cout << "Baseline  QPS=" << (uint64_t)qps_base
              << " +/- " << (uint64_t)sd_base
              << " (" << 100.0 * cv_base << "%)"
              << "  Recall@" << K << "=" << rec_base << "%\n";
    std::cout << "Medoid-batch QPS=" << (uint64_t)qps_ta
              << " +/- " << (uint64_t)sd_ta
              << " (" << 100.0 * cv_ta << "%)"
              << "  Recall@" << K << "=" << rec_ta   << "%\n";
    std::cout.precision(4);
    std::cout << "Speedup:  " << speedup << "x"
              << "  (noise band +/- " << 100.0 * noise_band << "%)\n";
    std::cout.precision(2);
    std::cout << "Recall drop: " << rec_base - rec_ta << "%\n";

    // Noise-aware verdict: is the speedup distinguishable from 1.0?
    double delta = std::fabs(speedup - 1.0);
    if (delta <= noise_band) {
        std::cout << "Verdict:  WITHIN NOISE (no measurable difference from baseline)\n";
    } else if (speedup > 1.0) {
        std::cout << "Verdict:  medoid-batch faster beyond noise band\n";
    } else {
        std::cout << "Verdict:  medoid-batch slower beyond noise band\n";
    }

    diskann::aligned_free(queries);
    delete[] gt_ids;
    delete[] gt_dists;
    return 0;
}
