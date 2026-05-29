# Vamana Baseline

## Hardware
- CPU: Intel Xeon 6767P (Granite Rapids)
- Cores: 256
- RAM: 251GB
- AMX: amx_bf16, amx_tile, amx_int8 confirmed

## Dataset
- SIFT1M: 1M vectors, 128 dimensions, float32
- Index params: R=32, L=125, alpha=1.2
- Search params: K=10, L=100

## Results — Threading Scaling Curve
| Threads | QPS | Recall@10 | Mean Latency | p99.9 Latency |
|---------|-----|-----------|--------------|---------------|
| 1 | 4,589 | 99.14% | 218 µs | 361 µs |
| 2 | 8,566 | 99.14% | 233 µs | 1,074 µs |
| 4 | 12,915 | 99.14% | 309 µs | 1,049 µs |
| 8 | 23,093 | 99.14% | 345 µs | 1,249 µs |
| 16 | 43,377 | 99.14% | 366 µs | 2,174 µs |
| 32 | 80,211 | 99.14% | 387 µs | 4,215 µs |
| 64 | 142,097 | 99.14% | 410 µs | 7,874 µs |
| 128 | 176,259 | 99.14% | 564 µs | 12,244 µs |
| 256 | 109,964 | 99.14% | 1,537 µs | 29,430 µs |

## Key Finding
Peak throughput at 128 threads (176,259 QPS). Performance degrades at 256 threads
due to memory bandwidth saturation — confirms graph ANNS is memory-bound.
Scaling becomes sublinear after 64 threads. AMX batching targets this bottleneck
by increasing compute intensity per memory load rather than adding more threads.
