/*
 * amx_init_test.cpp
 * Explicitly requests AMX tile permissions via arch_prctl before GEMM.
 * Tests whether MKL routes to AMX after proper OS-level initialization.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <mkl.h>
#include <mkl_cblas.h>

/* Linux AMX permission constants */
#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILEDATA      18

static int request_amx_permission() {
    /* Request AMX tile state permission from the kernel */
    long ret = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (ret != 0) {
        printf("WARNING: arch_prctl(ARCH_REQ_XCOMP_PERM) failed: %ld\n", ret);
        printf("         AMX tiles may not be available\n");
        return -1;
    }
    /* Verify permission was granted */
    unsigned long bitmask = 0;
    ret = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (ret == 0 && (bitmask & (1UL << XFEATURE_XTILEDATA))) {
        printf("AMX tile permission: GRANTED (bitmask bit 18 set)\n");
        return 0;
    }
    printf("WARNING: AMX tile permission not confirmed in bitmask: 0x%lx\n", bitmask);
    return -1;
}

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
    printf("=== AMX Explicit Initialization Test ===\n\n");

    /* Step 1: Request AMX permission BEFORE any MKL call */
    printf("Requesting AMX tile permission from OS...\n");
    int amx_ok = request_amx_permission();
    printf("\n");

    /* Step 2: Force single thread to eliminate threading noise */
    mkl_set_num_threads(1);
    printf("MKL threads set to 1 (eliminates barrier overhead)\n\n");

    int REPS = 100;
    int sizes[][3] = {
        /* Graph ANNS sizes */
        {32, 128, 128},
        {32, 1000, 128},
        {32, 1000, 960},
        {32, 4000, 960},
        /* LLM sizes that should definitely hit AMX */
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
    };
    const char* labels[] = {
        "ANNS hop-0 (32x128x128)",
        "ANNS batch=1000 SIFT",
        "ANNS batch=1000 GIST",
        "ANNS batch=4000 GIST",
        "LLM-small  (512^3)",
        "LLM-medium (1024^3)",
        "LLM-large  (2048^3)",
    };
    int n = 7;

    printf("%-28s | sgemm(ms) | bf16(ms) | speedup\n", "Config");
    printf("%-28s-|-----------|----------|--------\n", "----------------------------");

    for (int i = 0; i < n; i++) {
        int M = sizes[i][0], N = sizes[i][1], K = sizes[i][2];
        int reps = (M >= 1024) ? 10 : REPS;

        float*    A_f = (float*)mkl_malloc(M*K*sizeof(float), 64);
        float*    B_f = (float*)mkl_malloc(N*K*sizeof(float), 64);
        float*    C_s = (float*)mkl_malloc(M*N*sizeof(float), 64);
        float*    C_a = (float*)mkl_malloc(M*N*sizeof(float), 64);
        MKL_BF16* A_b = (MKL_BF16*)mkl_malloc(M*K*sizeof(MKL_BF16), 64);
        MKL_BF16* B_b = (MKL_BF16*)mkl_malloc(N*K*sizeof(MKL_BF16), 64);

        for (int j = 0; j < M*K; j++) { A_f[j]=(float)rand()/RAND_MAX; uint32_t u; memcpy(&u,&A_f[j],4); A_b[j]=(MKL_BF16)(u>>16); }
        for (int j = 0; j < N*K; j++) { B_f[j]=(float)rand()/RAND_MAX; uint32_t u; memcpy(&u,&B_f[j],4); B_b[j]=(MKL_BF16)(u>>16); }

        /* warmup */
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,M,N,K,1.0f,A_f,K,B_f,K,0.0f,C_s,N);
        cblas_gemm_bf16bf16f32(CblasRowMajor,CblasNoTrans,CblasTrans,M,N,K,1.0f,A_b,K,B_b,K,0.0f,C_a,N);

        double ts = now_sec();
        for (int r = 0; r < reps; r++)
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,M,N,K,1.0f,A_f,K,B_f,K,0.0f,C_s,N);
        double st = (now_sec()-ts)/reps*1000.0;

        ts = now_sec();
        for (int r = 0; r < reps; r++)
            cblas_gemm_bf16bf16f32(CblasRowMajor,CblasNoTrans,CblasTrans,M,N,K,1.0f,A_b,K,B_b,K,0.0f,C_a,N);
        double at = (now_sec()-ts)/reps*1000.0;

        printf("%-28s | %9.4f | %8.4f | %.2fx\n", labels[i], st, at, st/at);

        mkl_free(A_f); mkl_free(B_f); mkl_free(C_s); mkl_free(C_a);
        mkl_free(A_b); mkl_free(B_b);
    }

    printf("\n");
    if (amx_ok == 0)
        printf("AMX permission was granted. If speedup at 2048^3 is >4x, AMX is firing.\n");
    else
        printf("AMX permission was NOT granted. All results are AVX-512 only.\n");
    printf("Verify with: MKL_VERBOSE=1 ./amx_init_test 2>&1 | grep -i 'amx\\|tmul\\|BF16'\n");
    return 0;
}
