# Workload Perf Profiling — SIFT1M Vamana
## Hardware: Intel Xeon 6767P (Granite Rapids)
## Dataset: SIFT1M, 128d, L=100, K=10, 10K queries

| Threads | QPS | Cache Miss Rate | LLC Miss Rate | IPC |
|---------|-----|----------------|---------------|-----|
| 1 | 3,292 | 85.44% | 72.43% | 0.74 |
| 32 | 74,799 | 46.38% | 40.58% | 0.74 |
| 128 | 148,811 | 40.27% | 35.60% | 0.68 |

## Key finding
Cache miss rate drops 2x (85% → 40%) as concurrency increases from T=1 to T=128.
IPC remains flat (~0.74) confirming memory-bound bottleneck, not compute.
Concurrent queries share cache lines — each neighbor vector loaded once benefits multiple queries.
AMX GEMM makes this explicit: load each vector once, multiply against all concurrent queries simultaneously.
