#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <mkl.h>
#include <mkl_cblas.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA  18

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static MKL_BF16 f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    return (MKL_BF16)(u >> 16);
}

int main() {
    syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    mkl_set_num_threads(1);
    srand(42);

    printf("=== Approach: Higher R AMX Speedup Test ===\n");
    printf("Graph degree R determines hop-0 matrix M dimension\n\n");

    int R_vals[] = {32, 64, 128};
    int batch = 1000;
    int dims[] = {128, 960};
    int REPS = 200;

    for (int di = 0; di < 2; di++) {
        int K = dims[di];
        printf("--- DIM=%d (batch=%d) ---\n", K, batch);
        printf("  R  | Matrix(RxBxD)      | sgemm(ms) | bf16/AMX(ms) | speedup | Amdahl GIST\n");
        printf("-----|--------------------|-----------|--------------|---------|-----------\n");

        for (int ri = 0; ri < 3; ri++) {
            int R = R_vals[ri];
            int M = R, N = batch;

            float*    A_f = (float*)mkl_malloc(N*K*sizeof(float), 64);
            float*    B_f = (float*)mkl_malloc(M*K*sizeof(float), 64);
            float*    C_s = (float*)mkl_malloc(N*M*sizeof(float), 64);
            float*    C_a = (float*)mkl_malloc(N*M*sizeof(float), 64);
            MKL_BF16* A_b = (MKL_BF16*)mkl_malloc(N*K*sizeof(MKL_BF16), 64);
            MKL_BF16* B_b = (MKL_BF16*)mkl_malloc(M*K*sizeof(MKL_BF16), 64);

            for (int j = 0; j < N*K; j++) {
                A_f[j] = (float)rand()/RAND_MAX;
                uint32_t u; memcpy(&u, &A_f[j], 4);
                A_b[j] = (MKL_BF16)(u >> 16);
            }
            for (int j = 0; j < M*K; j++) {
                B_f[j] = (float)rand()/RAND_MAX;
                uint32_t u; memcpy(&u, &B_f[j], 4);
                B_b[j] = (MKL_BF16)(u >> 16);
            }

            /* warmup */
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,N,M,K,1.0f,A_f,K,B_f,K,0.0f,C_s,M);
            cblas_gemm_bf16bf16f32(CblasRowMajor,CblasNoTrans,CblasTrans,N,M,K,1.0f,A_b,K,B_b,K,0.0f,C_a,M);

            double ts = now_sec();
            for (int r = 0; r < REPS; r++)
                cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,N,M,K,1.0f,A_f,K,B_f,K,0.0f,C_s,M);
            double st = (now_sec()-ts)/REPS*1000.0;

            ts = now_sec();
            for (int r = 0; r < REPS; r++)
                cblas_gemm_bf16bf16f32(CblasRowMajor,CblasNoTrans,CblasTrans,N,M,K,1.0f,A_b,K,B_b,K,0.0f,C_a,M);
            double at = (now_sec()-ts)/REPS*1000.0;

            double spd = st/at;
            /* Amdahl for GIST1M: window grows with R */
            /* Approximate: window_pct ~ 27.6% * (R/32) but capped */
            double win_pct = (K == 960) ? (27.6 * R / 32.0) : (7.1 * R / 32.0);
            if (win_pct > 60.0) win_pct = 60.0; /* conservative cap */
            double amdahl = 1.0 / ((1.0 - win_pct/100.0) + (win_pct/100.0)/spd);

            char matrix[32];
            snprintf(matrix, sizeof(matrix), "%dx%dx%d", R, N, K);
            printf(" %3d | %-19s| %9.4f | %12.4f | %7.2fx | %9.3fx\n",
                   R, matrix, st, at, spd, amdahl);

            mkl_free(A_f); mkl_free(B_f); mkl_free(C_s); mkl_free(C_a);
            mkl_free(A_b); mkl_free(B_b);
        }
        printf("\n");
    }
    return 0;
}
