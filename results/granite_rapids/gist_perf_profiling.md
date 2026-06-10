# GIST1M Perf Profiling
## Hardware: Intel Xeon 6767P (Granite Rapids)
## Dataset: GIST1M, 960d, L=100, K=10, 1K queries

| Threads | QPS | Cache Miss Rate | LLC Miss Rate | IPC |
|---------|-----|----------------|---------------|-----|
| 1 | 928.48 | 78.25% | 74.12% | 0.80 |
| 32 | 14,687.81 | 77.16% | 81.02% | 0.83 |
| 128 | 20,929.38 | 75.94% | 78.74% | 0.83 |

## Key finding
Cache miss rate stays flat (~77%) regardless of thread count — unlike SIFT1M which drops from 85% to 40%.
Each GIST1M vector is 3,840 bytes (60 cache lines) vs 512 bytes for SIFT1M.
At 32 neighbors per hop: 1,920 cache lines loaded per hop — too large for cache residency between queries.
Thread-level sharing cannot rescue the cache situation at 960d.
AMX batching is therefore more critical at high dimensionality: amortizes expensive vector loads across all concurrent queries.
