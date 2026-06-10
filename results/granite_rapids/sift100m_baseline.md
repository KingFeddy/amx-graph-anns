# SIFT 100M Vamana Threading Baseline
## Hardware: Intel Xeon 6767P (Granite Rapids), 256 cores, 251GB RAM
## Dataset: SIFT 100M subset (first 100M from SIFT1B), 128 dims, uint8, L=100, K=10

| Threads | QPS | Recall@10 |
|---------|-----|-----------|
| 1 | 1456.87 | 93.58% |
| 2 | 2751.20 | 93.58% |
| 4 | 4656.55 | 93.58% |
| 8 | 8719.75 | 93.58% |
| 16 | 16156.76 | 93.58% |
| 32 | 30515.24 | 93.58% |
| 64 | 60392.78 | 93.58% |
| 128 | 109720.42 | 93.58% |
| 256 | 111069.03 | 93.58% |

Peak: 111,069 QPS at 256 threads.
Note: Lower recall vs SIFT1M (93.58% vs 99.13%) — larger graph requires higher L for equivalent recall.
Note: Ground truth computed against 100M subset using compute_groundtruth, not the 1B ground truth file.
