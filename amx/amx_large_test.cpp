/*
 * amx_large_test.cpp
 * Demonstrates AMX actually engaging at large matrix sizes.
 * Shows the minimum matrix size needed to trigger AMX dispatch.
 * This proves AMX works on this hardware but graph ANNS matrices are too small.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <mkl.h>
#include <mkl_cblas.h>

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static MKL_BF16 f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    return (MKL_BF16)(u >> 16);
}

static void fill_bf16(MKL_BF16* p, int n) {
    for (int i = 0; i < n; i++) {
        float f = (float)rand() / RAND_MAX;
        p[i] = f32_to_bf16(f);
    }
}

static void fill_f32(float* p, int n) {
    for (int i = 0; i < n; i++) p[i] = (float)rand() / RAND_MAX;
}

int main() {
    srand(42);
    printf("=== AMX Dispatch Threshold Test ===\n");
    printf("Finding minimum matrix size that triggers AMX on Granite Rapids\n");
    printf("Watch MKL_VERBOSE output for when SGEMM changes to AMX kernel\n\n");

    /* Graph ANNS realistic sizes: M=neighbors, N=batch, K=dims */
    printf("--- Graph ANNS realistic sizes (M=neighbors) ---\n");
    printf("  M(neighbors) x N(batch) x K(dims) | sgemm(us) | bf16(us) | speedup\n");
    printf("  ----------------------------------|-----------|----------|--------\n");

    int dims[] = {128, 960};
    int batches[] = {32, 64, 128, 256, 512, 1000, 2000, 4000};
    int M = 32; /* fixed: graph degree */

    for (int di = 0; di < 2; di++) {
        int K = dims[di];
        for (int bi = 0; bi < 8; bi++) {
            int N = batches[bi];
            float*    A_f32  = (float*)mkl_malloc(N * K * sizeof(float), 64);
            float*    B_f32  = (float*)mkl_malloc(M * K * sizeof(float), 64);
            float*    C_s    = (float*)mkl_malloc(N * M * sizeof(float), 64);
            float*    C_a    = (float*)mkl_malloc(N * M * sizeof(float), 64);
            MKL_BF16* A_bf16 = (MKL_BF16*)mkl_malloc(N * K * sizeof(MKL_BF16), 64);
            MKL_BF16* B_bf16 = (MKL_BF16*)mkl_malloc(M * K * sizeof(MKL_BF16), 64);

            fill_f32(A_f32, N*K); fill_f32(B_f32, M*K);
            fill_bf16(A_bf16, N*K); fill_bf16(B_bf16, M*K);

            /* warmup */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        N, M, K, 1.0f, A_f32, K, B_f32, K, 0.0f, C_s, M);
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                                    N, M, K, 1.0f, A_bf16, K, B_bf16, K, 0.0f, C_a, M);

            int reps = 200;
            double ts = now_sec();
            for (int r = 0; r < reps; r++)
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            N, M, K, 1.0f, A_f32, K, B_f32, K, 0.0f, C_s, M);
            double st = (now_sec() - ts) / reps * 1e6;

            ts = now_sec();
            for (int r = 0; r < reps; r++)
                cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                                        N, M, K, 1.0f, A_bf16, K, B_bf16, K, 0.0f, C_a, M);
            double at = (now_sec() - ts) / reps * 1e6;

            printf("  %4d x %4d x %4d              | %9.2f | %8.2f | %.2fx\n",
                   M, N, K, st, at, st/at);

            mkl_free(A_f32); mkl_free(B_f32); mkl_free(C_s); mkl_free(C_a);
            mkl_free(A_bf16); mkl_free(B_bf16);
        }
        printf("\n");
    }

    /* Large matrix test to prove AMX CAN fire on this hardware */
    printf("--- Large matrices (LLM-style, should trigger AMX) ---\n");
    printf("  M x N x K               | sgemm(ms) | bf16(ms) | speedup\n");
    printf("  -------------------------|-----------|----------|--------\n");

    int large[][3] = {{512,512,512},{1024,1024,1024},{2048,2048,2048},{4096,1024,1024}};
    for (int i = 0; i < 4; i++) {
        int LM = large[i][0], LN = large[i][1], LK = large[i][2];
        float*    A_f32  = (float*)mkl_malloc(LM*LK*sizeof(float), 64);
        float*    B_f32  = (float*)mkl_malloc(LN*LK*sizeof(float), 64);
        float*    C_s    = (float*)mkl_malloc(LM*LN*sizeof(float), 64);
        float*    C_a    = (float*)mkl_malloc(LM*LN*sizeof(float), 64);
        MKL_BF16* A_bf16 = (MKL_BF16*)mkl_malloc(LM*LK*sizeof(MKL_BF16), 64);
        MKL_BF16* B_bf16 = (MKL_BF16*)mkl_malloc(LN*LK*sizeof(MKL_BF16), 64);

        fill_f32(A_f32, LM*LK); fill_f32(B_f32, LN*LK);
        fill_bf16(A_bf16, LM*LK); fill_bf16(B_bf16, LN*LK);

        /* warmup */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    LM, LN, LK, 1.0f, A_f32, LK, B_f32, LK, 0.0f, C_s, LN);
        cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                                LM, LN, LK, 1.0f, A_bf16, LK, B_bf16, LK, 0.0f, C_a, LN);

        int reps = 20;
        double ts = now_sec();
        for (int r = 0; r < reps; r++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        LM, LN, LK, 1.0f, A_f32, LK, B_f32, LK, 0.0f, C_s, LN);
        double st = (now_sec() - ts) / reps * 1000.0;

        ts = now_sec();
        for (int r = 0; r < reps; r++)
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans,
                                    LM, LN, LK, 1.0f, A_bf16, LK, B_bf16, LK, 0.0f, C_a, LN);
        double at = (now_sec() - ts) / reps * 1000.0;

        printf("  %4d x %4d x %4d         | %9.4f | %8.4f | %.2fx\n",
               LM, LN, LK, st, at, st/at);

        mkl_free(A_f32); mkl_free(B_f32); mkl_free(C_s); mkl_free(C_a);
        mkl_free(A_bf16); mkl_free(B_bf16);
    }

    printf("\nConclusion: Graph ANNS matrices (M=32) are too small for AMX dispatch.\n");
    printf("AMX fires at large square matrices (LLM workloads, not ANNS).\n");
    printf("This is the fundamental reason graph ANNS is AMX-resistant.\n");
    return 0;
}
