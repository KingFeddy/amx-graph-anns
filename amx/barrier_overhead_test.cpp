#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=== Barrier Overhead Test for Epoch-Synchronous Search ===\n\n");

    int n_hops = 104;
    int n_reps = 1000;
    int thread_counts[] = {1, 8, 16, 32, 64, 128};
    int n_tc = 6;
    double search_times[] = {800.0, 120.0, 60.0, 243.0, 47.0, 35.0};

    printf("%-10s | %-18s | %-16s | %-16s | %-12s\n",
           "Threads", "Time/barrier(us)", "104 barriers(ms)",
           "Search time(ms)", "Overhead%");
    printf("----------------------------------------------------------------------\n");

    for (int ti = 0; ti < n_tc; ti++) {
        int T = thread_counts[ti];
        omp_set_num_threads(T);

        /* warmup */
        #pragma omp parallel
        {
            #pragma omp barrier
        }

        double t0 = now_sec();
        for (int r = 0; r < n_reps; r++) {
            for (int h = 0; h < n_hops; h++) {
                #pragma omp parallel
                {
                    volatile double x = (double)(omp_get_thread_num() + 1) * 1.001;
                    (void)x;
                    #pragma omp barrier
                }
            }
        }
        double elapsed = now_sec() - t0;

        double us_per_barrier   = elapsed / (n_reps * n_hops) * 1e6;
        double total_barrier_ms = us_per_barrier * n_hops / 1000.0;
        double search_ms        = search_times[ti];
        double overhead_pct     = total_barrier_ms / search_ms * 100.0;

        printf("%-10d | %-18.2f | %-16.3f | %-16.1f | %-11.1f%%\n",
               T, us_per_barrier, total_barrier_ms, search_ms, overhead_pct);
    }

    printf("\n=== Interpretation ===\n");
    printf("Idea A adds 104 barriers per 1000-query batch.\n");
    printf("AMX projected gain on GIST1M: ~34%% of search time saved.\n");
    printf("If overhead%% < 34%% -> Idea A viable.\n");
    printf("If overhead%% > 34%% -> sync kills the gain.\n");
    return 0;
}
