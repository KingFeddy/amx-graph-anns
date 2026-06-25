#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/index_io.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <fstream>
#include <vector>

#define DIM 128
#define N_BASE 1000000
#define N_QUERIES 10000
#define NLIST 1024
#define K 10

// Timer
double now_sec() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// Load fvecs file into flat float array
std::vector<float> load_fvecs(const char* path, int max_n) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", path); exit(1); }
    std::vector<float> data;
    int dim;
    while ((int)data.size() / DIM < max_n) {
        if (fread(&dim, sizeof(int), 1, f) != 1) break;
        std::vector<float> vec(dim);
        fread(vec.data(), sizeof(float), dim, f);
        data.insert(data.end(), vec.begin(), vec.end());
    }
    fclose(f);
    printf("Loaded %zu vectors from %s\n", data.size() / DIM, path);
    return data;
}

float compute_recall(const faiss::idx_t* results, const int* gt, int nq, int k, int gt_k) {
    int correct = 0;
    for (int q = 0; q < nq; q++) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < gt_k; j++) {
                if (results[q * k + i] == gt[q * gt_k + j]) {
                    correct++;
                    break;
                }
            }
        }
    }
    return (float)correct / (nq * k);
}

std::vector<int> load_ivecs(const char* path, int nq, int& gt_k) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", path); exit(1); }
    fread(&gt_k, sizeof(int), 1, f);
    std::vector<int> gt(nq * gt_k);
    fread(gt.data(), sizeof(int), gt_k, f);
    for (int q = 1; q < nq; q++) {
        int dim;
        fread(&dim, sizeof(int), 1, f);
        fread(gt.data() + q * gt_k, sizeof(int), gt_k, f);
    }
    fclose(f);
    printf("Loaded ground truth: %d queries x %d neighbors\n", nq, gt_k);
    return gt;
}

int main() {
    printf("=== IVF Baseline — Granite Rapids (Xeon 6767P) ===\n\n");

    // Load data
    auto base    = load_fvecs("data/sift/sift_base.fvecs", N_BASE);
    auto queries = load_fvecs("data/sift/sift_query.fvecs", N_QUERIES);
    int gt_k;
    auto gt = load_ivecs("data/sift/sift_groundtruth.ivecs", N_QUERIES, gt_k);

    // Build or load index
    faiss::IndexIVFFlat* index;
    const char* index_path = "data/sift/ivf_index.faiss";
    std::ifstream check(index_path);
    if (check.good()) {
        printf("Loading saved index from %s\n", index_path);
        index = dynamic_cast<faiss::IndexIVFFlat*>(faiss::read_index(index_path));
    } else {
        printf("Building IVF index: nlist=%d...\n", NLIST);
        faiss::IndexFlatL2 quantizer(DIM);
        index = new faiss::IndexIVFFlat(&quantizer, DIM, NLIST);
        double t0 = now_sec();
        index->train(N_BASE, base.data());
        index->add(N_BASE, base.data());
        printf("Index built in %.2f seconds\n", now_sec() - t0);
        faiss::write_index(index, index_path);
        printf("Index saved to %s\n", index_path);
    }

    // Results buffers
    std::vector<faiss::idx_t>  labels(N_QUERIES * K);
    std::vector<float> dists(N_QUERIES * K);

    // Run search at multiple nprobe values
    printf("\n--- Throughput Scaling (nprobe) ---\n");
    printf("nprobe | QPS | Recall@%d\n", K);
    printf("-------|-----|----------\n");

    int nprobes[] = {1, 4, 8, 16, 32, 64, 128};
    int n_nprobes = 7;

    for (int p = 0; p < n_nprobes; p++) {
        index->nprobe = nprobes[p];

        // Warmup
        index->search(100, queries.data(), K, dists.data(), labels.data());

        // Timed run
        double t0 = now_sec();
        index->search(N_QUERIES, queries.data(), K, dists.data(), labels.data());
        double elapsed = now_sec() - t0;

        float qps     = N_QUERIES / elapsed;
        float recall  = compute_recall(labels.data(), gt.data(), N_QUERIES, K, gt_k);

        printf("%6d | %9.0f | %.4f\n", nprobes[p], qps, recall);
    }

    printf("\nDone.\n");
    delete index;
    return 0;
}
