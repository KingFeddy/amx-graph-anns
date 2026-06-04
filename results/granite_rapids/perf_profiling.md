# Hardware Profiling — Granite Rapids (Intel Xeon 6767P)

## Tool: perf stat
## Dataset: SIFT1M, 1000 queries, single threaded

## Vamana (DiskANN, R=32, L=100)

| Metric | Value |
|--------|-------|
| Cache miss rate | 60.77% |
| Cache references | 158,213,453 |
| Cache misses | 96,140,883 |
| Instructions per cycle | 1.30 |
| Total cycles | 8,146,374,095 |
| LLC loads | 24,581,793 |
| LLC load miss rate | 24.49% |
| Wall time | 3.54s |

## IVF (Faiss, nlist=1024, all nprobes)

| Metric | Value |
|--------|-------|
| Cache miss rate | 21.07% |
| Cache references | 21,373,654,742 |
| Cache misses | 4,502,637,655 |
| Instructions per cycle | 0.48 |
| Total cycles | 1,790,917,342,159 |
| LLC loads | 2,916,800,573 |
| LLC load miss rate | 24.85% |
| Wall time | 4.08s (all nprobes combined) |

## Key Finding

Vamana has 3x higher cache miss rate than IVF (60.77% vs 21.07%),
confirming graph traversal is memory-bound due to random edge access
patterns. Vamana's higher IPC (1.30 vs 0.48) shows the CPU is
compute-efficient when data arrives — the bottleneck is purely data
supply, not compute. AMX batching attacks this by amortizing memory
loads across multiple queries, increasing compute intensity per load.
