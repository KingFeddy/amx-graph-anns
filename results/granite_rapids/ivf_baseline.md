# IVF Baseline — Granite Rapids (Intel Xeon 6767P)

## Hardware
- CPU: Intel Xeon 6767P (Granite Rapids)
- Cores: 256
- RAM: 251GB

## Dataset
- SIFT1M: 1M vectors, 128 dimensions, float32
- Index params: nlist=1024
- Search params: K=10, single threaded

## Results — nprobe Scaling
| nprobe | QPS | Recall@10 |
|--------|-----|-----------|
| 1 | 204,491 | 0.9490 |
| 4 | 363,840 | 1.0000 |
| 8 | 181,729 | 1.0000 |
| 16 | 94,536 | 1.0000 |
| 32 | 48,436 | 1.0000 |
| 64 | 25,146 | 1.0000 |
| 128 | 11,901 | 1.0000 |

## Key Finding
Peak throughput at nprobe=4 (363,840 QPS) with perfect recall.
nprobe=1 is faster at 204K QPS but recall drops to 94.9%.
IVF peak is ~2x faster than Vamana peak (176K QPS) confirming
cluster structure enables better hardware utilization than graph traversal.
