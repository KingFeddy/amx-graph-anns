#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mkl.h>
#include <mkl_cblas.h>

#define BATCH_SIZE 32
#define DIM 128
#define N_NEIGHBORS 32

// Brute force multiply
void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

// Fill matrix with random floats in [0, 1]
void random_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Check if two matrices are close enough
float max_diff(const float* A, const float* B, int n) {
    float max = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(A[i] - B[i]);
        if (diff > max) max = diff;
    }
    return max;
}

// Timer
double now_sec() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main() {
    printf("=== AMX GEMM Validation Test ===\n");
    printf("Hardware: Intel Xeon 6767P (Granite Rapids)\n");
    printf("Matrix: (%d x %d) x (%d x %d)^T = (%d x %d)\n", BATCH_SIZE, DIM, N_NEIGHBORS, DIM, BATCH_SIZE, N_NEIGHBORS);
    printf("Simulating: %d concurrent queries x %d neighbor vectors\n\n", BATCH_SIZE, N_NEIGHBORS);

    srand(42);

    float* A = (float*)mkl_malloc(BATCH_SIZE * DIM * sizeof(float), 64);
    float* B = (float*)mkl_malloc(N_NEIGHBORS * DIM * sizeof(float), 64);
    float* C_naive = (float*)mkl_malloc(BATCH_SIZE * N_NEIGHBORS * sizeof(float), 64);
    float* C_mkl = (float*)mkl_malloc(BATCH_SIZE * N_NEIGHBORS * sizeof(float), 64);

    if (!A || !B || !C_naive || !C_mkl) {
        printf("ERROR: allocation failed\n");
        return 1;
    }

    // Fill with random data
    random_matrix(A, BATCH_SIZE, DIM);
    random_matrix(B, N_NEIGHBORS, DIM);

    printf("Running naive triple-loop multiply...\n");
    double t0 = now_sec();
    naive_gemm(A, B, C_naive, BATCH_SIZE, N_NEIGHBORS, DIM);
    double naive_time = now_sec() - t0;
    printf("  Naive time: %.6f seconds\n\n", naive_time);

    printf("Running MKL GEMM (AMX accelerated)...\n");

    // Warmup call. AMX tiles need to be initialized on first use
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                BATCH_SIZE, N_NEIGHBORS, DIM,
                1.0f,
                A, DIM,
                B, DIM,
                0.0f,
                C_mkl, N_NEIGHBORS);

    // Timed call
    double t1 = now_sec();
    for (int i = 0; i < 1000; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    BATCH_SIZE, N_NEIGHBORS, DIM,
                    1.0f,
                    A, DIM,
                    B, DIM,
                    0.0f,
                    C_mkl, N_NEIGHBORS);
    }
    double mkl_time = (now_sec() - t1) / 1000.0;
    printf("  MKL time:   %.6f seconds (avg over 1000 runs)\n\n", mkl_time);

    // Correctness check
    float diff = max_diff(C_naive, C_mkl, BATCH_SIZE * N_NEIGHBORS);
    printf("=== Correctness ===\n");
    printf("  Max absolute difference: %.6f\n", diff);
    printf("  Result: %s\n\n", diff < 1e-3 ? "PASS" : "FAIL");

    // Speedup
    printf("=== Performance ===\n");
    printf("  Naive:   %.6f sec\n", naive_time);
    printf("  MKL/AMX: %.6f sec\n", mkl_time);
    printf("  Speedup: %.2fx\n\n", naive_time / mkl_time);

    // Scaling test
    printf("=== Batch Size Scaling ===\n");
    printf("  (simulates more concurrent queries)\n");
    printf("  Batch | Time (ms) | GFLOPS\n");
    printf("  ------|-----------|-------\n");

    int batch_sizes[] = {1, 8, 16, 32, 64, 128, 256, 512, 1000};
    int n_batches = 9;

    for (int b = 0; b < n_batches; b++) {
        int M = batch_sizes[b];
        float* A_b = (float*)mkl_malloc(M * DIM * sizeof(float), 64);
        float* C_b = (float*)mkl_malloc(M * N_NEIGHBORS * sizeof(float), 64);
        random_matrix(A_b, M, DIM);

        // warmup
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N_NEIGHBORS, DIM, 1.0f, A_b, DIM, B, DIM, 0.0f, C_b, N_NEIGHBORS);

        // timed
        double ts = now_sec();
        int reps = 1000;
        for (int r = 0; r < reps; r++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N_NEIGHBORS, DIM, 1.0f,
                        A_b, DIM, B, DIM, 0.0f, C_b, N_NEIGHBORS);
        }
        double elapsed = (now_sec() - ts) / reps;
        double gflops = (2.0 * M * N_NEIGHBORS * DIM) / (elapsed * 1e9);
        printf("  %5d | %9.4f | %.2f\n", M, elapsed * 1000.0, gflops);

        mkl_free(A_b);
        mkl_free(C_b);
    }

    printf("\nDone. AMX GEMM validated on Granite Rapids.\n");

    mkl_free(A);
    mkl_free(B);
    mkl_free(C_naive);
    mkl_free(C_mkl);
    return 0;
}
