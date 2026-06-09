# GIST1M Vamana Threading Baseline
## Hardware: Intel Xeon 6767P (Granite Rapids), 256 cores, 251GB RAM
## Dataset: GIST1M, 960 dims, 1M vectors, L=100, K=10

| Threads | QPS | Recall@10 |
|---------|-----|-----------|
| 1 | 915.70 | 88.50% |
| 2 | 1803.22 | 88.50% |
| 4 | 2870.89 | 88.50% |
| 8 | 5146.12 | 88.50% |
| 16 | 9829.33 | 88.50% |
| 32 | 15644.56 | 88.50% |
| 64 | 21714.56 | 88.50% |
| 128 | 23005.18 | 88.50% |
| 256 | 12553.22 | 88.50% |

Peak: 23,005 QPS at 128 threads. Degrades sharply at 256 — memory bandwidth saturation.
Note: Lower recall vs SIFT1M (88.5% vs 99.13%) due to higher dimensionality with same L=100.
