#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mkl.h>
#include <mkl_cblas.h>

#define DIM 128
#define N_NEIGHBORS 32

// Timer
double now_sec() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}


// Fill matrix with random floats in [0, 1]
void random_matrix(float* mat, int n) {
    for (int i = 0; i < n; i++)
        mat[i] = (float)rand() / RAND_MAX;
}

int main() {
    printf("=== Extended AMX Batch Scaling ===\n");
    printf("Finding GFLOPS plateau point\n\n");
    printf("Batch | Time (us) | GFLOPS\n");
    printf("------|-----------|-------\n");

    srand(42);

    float* B = (float*)mkl_malloc(N_NEIGHBORS * DIM * sizeof(float), 64);
    random_matrix(B, N_NEIGHBORS * DIM);

    int batches[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 10000};
    int n = 15;

    for (int b = 0; b < n; b++) {
        int M = batches[b];
        float* A = (float*)mkl_malloc(M * DIM * sizeof(float), 64);
        float* C = (float*)mkl_malloc(M * N_NEIGHBORS * sizeof(float), 64);
        random_matrix(A, M * DIM);
	
	// Warmup
        for (int w = 0; w < 5; w++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N_NEIGHBORS, DIM, 1.0f,
                        A, DIM, B, DIM, 0.0f, C, N_NEIGHBORS);
	
	// More reps for smaller batches so timing is stabley
        int reps = (M < 100) ? 10000 : (M < 1000) ? 1000 : 100;
        double ts = now_sec();
        for (int r = 0; r < reps; r++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N_NEIGHBORS, DIM, 1.0f,
                        A, DIM, B, DIM, 0.0f, C, N_NEIGHBORS);
        double elapsed = (now_sec() - ts) / reps;
        double gflops = (2.0 * M * N_NEIGHBORS * DIM) / (elapsed * 1e9);

        printf("%5d | %9.2f | %.2f\n", M, elapsed * 1e6, gflops);

        mkl_free(A);
        mkl_free(C);
    }

    mkl_free(B);
    printf("\nDone.\n");
    return 0;
}
