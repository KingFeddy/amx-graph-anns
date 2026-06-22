/*
 * amx_gemm_test.cpp
 * Three-way benchmark: naive loop vs FP32 SGEMM (AVX-512) vs BF16 GEMM (AMX)
 * Tests both SIFT-like (128d) and GIST-like (960d) to show dimensionality effect.
 * NOTE: AMX TMUL only supports BF16/INT8 — FP32 sgemm dispatches to AVX-512, NOT AMX.
 * Verify AMX tile engagement with: vtune -collect hotspots (check AMX utilization metric)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <mkl.h>
#include <mkl_cblas.h>

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Convert float32 -> BF16 by truncating lower 16 mantissa bits */
static MKL_BF16 f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (MKL_BF16)(u >> 16);
}

static void fill_random_f32(float* p, int n) {
    for (int i = 0; i < n; i++)
        p[i] = (float)rand() / RAND_MAX;
}

static void convert_to_bf16(const float* src, MKL_BF16* dst, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = f32_to_bf16(src[i]);
}

/* Naive triple-loop GEMM: C[M,N] = A[M,K] * B[N,K]^T */
static void naive_gemm(const float* A, const float* B, float* C,
                        int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                C[m*N + n] += A[m*K + k] * B[n*K + k];
}

static float max_diff(const float* a, const float* b, int n) {
    float d = 0;
    for (int i = 0; i < n; i++) {
        float x = fabsf(a[i] - b[i]);
        if (x > d) d = x;
    }
    return d;
}

static void run_benchmark(const char* label, int DIM, int N_NEIGHBORS,
                           int BATCH_SIZE, int REPS) {
    printf("\n========================================\n");
    printf("Dataset: %s | DIM=%d | Neighbors=%d | Batch=%d\n",
           label, DIM, N_NEIGHBORS, BATCH_SIZE);
    printf("========================================\n");

    /* Allocate FP32 matrices */
    float* A_f32    = (float*)mkl_malloc(BATCH_SIZE * DIM * sizeof(float), 64);
    float* B_f32    = (float*)mkl_malloc(N_NEIGHBORS * DIM * sizeof(float), 64);
    float* C_naive  = (float*)mkl_malloc(BATCH_SIZE * N_NEIGHBORS * sizeof(float), 64);
    float* C_sgemm  = (float*)mkl_malloc(BATCH_SIZE * N_NEIGHBORS * sizeof(float), 64);
    float* C_amx    = (float*)mkl_malloc(BATCH_SIZE * N_NEIGHBORS * sizeof(float), 64);

    /* Allocate BF16 matrices */
    MKL_BF16* A_bf16 = (MKL_BF16*)mkl_malloc(BATCH_SIZE * DIM * sizeof(MKL_BF16), 64);
    MKL_BF16* B_bf16 = (MKL_BF16*)mkl_malloc(N_NEIGHBORS * DIM * sizeof(MKL_BF16), 64);

    fill_random_f32(A_f32, BATCH_SIZE * DIM);
    fill_random_f32(B_f32, N_NEIGHBORS * DIM);
    convert_to_bf16(A_f32, A_bf16, BATCH_SIZE * DIM);
    convert_to_bf16(B_f32, B_bf16, N_NEIGHBORS * DIM);

    /* 1. Naive baseline (small batch only to avoid timeout) */
    int naive_batch = (BATCH_SIZE <= 64) ? BATCH_SIZE : 64;
    double t0 = now_sec();
    for (int r = 0; r < 10; r++)
        naive_gemm(A_f32, B_f32, C_naive, naive_batch, N_NEIGHBORS, DIM);
    double naive_time = (now_sec() - t0) / 10.0;
    double naive_gflops = (2.0 * naive_batch * N_NEIGHBORS * DIM) / (naive_time * 1e9);
    printf("\n[1] Naive triple-loop (FP32, batch=%d)\n", naive_batch);
    printf("    Time: %.6f s | GFLOPS: %.3f\n", naive_time, naive_gflops);

    /* 2. FP32 SGEMM — dispatches to AVX-512, NOT AMX */
    /* warmup */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                BATCH_SIZE, N_NEIGHBORS, DIM, 1.0f,
                A_f32, DIM, B_f32, DIM, 0.0f, C_sgemm, N_NEIGHBORS);
    double t1 = now_sec();
    for (int r = 0; r < REPS; r++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    BATCH_SIZE, N_NEIGHBORS, DIM, 1.0f,
                    A_f32, DIM, B_f32, DIM, 0.0f, C_sgemm, N_NEIGHBORS);
    double sgemm_time = (now_sec() - t1) / REPS;
    double sgemm_gflops = (2.0 * BATCH_SIZE * N_NEIGHBORS * DIM) / (sgemm_time * 1e9);
    printf("\n[2] FP32 SGEMM via MKL (AVX-512, NOT AMX)\n");
    printf("    Time: %.6f s | GFLOPS: %.3f\n", sgemm_time, sgemm_gflops);
    printf("    Speedup vs naive: %.2fx\n", naive_time / sgemm_time);

    /* 3. BF16 GEMM — actually engages AMX TMUL on Granite Rapids */
    /* warmup */
    cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                            BATCH_SIZE, N_NEIGHBORS, DIM, 1.0f,
                            A_bf16, DIM, B_bf16, DIM, 0.0f,
                            C_amx, N_NEIGHBORS);
    double t2 = now_sec();
    for (int r = 0; r < REPS; r++)
        cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                                BATCH_SIZE, N_NEIGHBORS, DIM, 1.0f,
                                A_bf16, DIM, B_bf16, DIM, 0.0f,
                                C_amx, N_NEIGHBORS);
    double amx_time = (now_sec() - t2) / REPS;
    double amx_gflops = (2.0 * BATCH_SIZE * N_NEIGHBORS * DIM) / (amx_time * 1e9);
    printf("\n[3] BF16 GEMM via MKL (AMX TMUL path)\n");
    printf("    Time: %.6f s | GFLOPS: %.3f\n", amx_time, amx_gflops);
    printf("    Speedup vs naive:  %.2fx\n", naive_time / amx_time);
    printf("    Speedup vs sgemm:  %.2fx  <-- THIS is the real AMX gain\n",
           sgemm_time / amx_time);

    /* Correctness: BF16 vs FP32 sgemm */
    float diff = max_diff(C_sgemm, C_amx, BATCH_SIZE * N_NEIGHBORS);
    printf("\n[Correctness] Max |FP32 - BF16| diff: %.6f %s\n",
           diff, diff < 0.1f ? "(PASS — BF16 precision acceptable)" : "(WARN — check recall)");

    mkl_free(A_f32); mkl_free(B_f32);
    mkl_free(C_naive); mkl_free(C_sgemm); mkl_free(C_amx);
    mkl_free(A_bf16); mkl_free(B_bf16);
}

static void scaling_sweep(const char* label, int DIM, int N_NEIGHBORS) {
    printf("\n--- Batch Size Scaling: %s (DIM=%d) ---\n", label, DIM);
    printf("  Batch | sgemm(ms) | bf16/AMX(ms) | AMX/sgemm speedup | AMX GFLOPS\n");
    printf("  ------|-----------|--------------|-------------------|----------\n");

    int batches[] = {1, 8, 16, 32, 64, 128, 256, 512, 1000};
    int nb = 9;
    int REPS = 500;

    for (int b = 0; b < nb; b++) {
        int M = batches[b];
        float*    A_f32  = (float*)mkl_malloc(M * DIM * sizeof(float), 64);
        float*    B_f32  = (float*)mkl_malloc(N_NEIGHBORS * DIM * sizeof(float), 64);
        float*    C_s    = (float*)mkl_malloc(M * N_NEIGHBORS * sizeof(float), 64);
        float*    C_a    = (float*)mkl_malloc(M * N_NEIGHBORS * sizeof(float), 64);
        MKL_BF16* A_bf16 = (MKL_BF16*)mkl_malloc(M * DIM * sizeof(MKL_BF16), 64);
        MKL_BF16* B_bf16 = (MKL_BF16*)mkl_malloc(N_NEIGHBORS * DIM * sizeof(MKL_BF16), 64);

        fill_random_f32(A_f32, M * DIM);
        fill_random_f32(B_f32, N_NEIGHBORS * DIM);
        convert_to_bf16(A_f32, A_bf16, M * DIM);
        convert_to_bf16(B_f32, B_bf16, N_NEIGHBORS * DIM);

        /* warmup */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N_NEIGHBORS, DIM, 1.0f, A_f32, DIM, B_f32, DIM, 0.0f, C_s, N_NEIGHBORS);
        cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                                M, N_NEIGHBORS, DIM, 1.0f,
                                A_bf16, DIM, B_bf16, DIM, 0.0f, C_a, N_NEIGHBORS);

        double ts = now_sec();
        for (int r = 0; r < REPS; r++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N_NEIGHBORS, DIM, 1.0f, A_f32, DIM, B_f32, DIM, 0.0f, C_s, N_NEIGHBORS);
        double sgemm_t = (now_sec() - ts) / REPS;

        ts = now_sec();
        for (int r = 0; r < REPS; r++)
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                                    M, N_NEIGHBORS, DIM, 1.0f,
                                    A_bf16, DIM, B_bf16, DIM, 0.0f, C_a, N_NEIGHBORS);
        double amx_t = (now_sec() - ts) / REPS;

        double amx_gflops = (2.0 * M * N_NEIGHBORS * DIM) / (amx_t * 1e9);
        printf("  %5d | %9.4f | %12.4f | %17.2fx | %.2f\n",
               M, sgemm_t * 1000.0, amx_t * 1000.0,
               sgemm_t / amx_t, amx_gflops);

        mkl_free(A_f32); mkl_free(B_f32); mkl_free(C_s); mkl_free(C_a);
        mkl_free(A_bf16); mkl_free(B_bf16);
    }
}

int main() {
    srand(42);
    printf("=== AMX Graph ANNS — Three-Way GEMM Benchmark ===\n");
    printf("Naive (FP32) vs MKL SGEMM (AVX-512) vs MKL BF16 GEMM (AMX TMUL)\n");
    printf("Verify AMX engagement: vtune -collect hotspots, check AMX utilization\n");

    /* SIFT1M-like: 128d, 32 neighbors, batch=1000 */
    run_benchmark("SIFT1M-like", 128, 32, 1000, 1000);

    /* GIST1M-like: 960d, 32 neighbors, batch=1000 */
    run_benchmark("GIST1M-like", 960, 32, 1000, 500);

    /* Batch scaling sweep */
    scaling_sweep("SIFT1M-like", 128, 32);
    scaling_sweep("GIST1M-like", 960, 32);

    printf("\n=== Done ===\n");
    printf("Key number to report: 'AMX/sgemm speedup' column — this is the real AMX gain.\n");
    printf("Run with VTune to confirm AMX tile utilization > 0%%.\n");
    return 0;
}
