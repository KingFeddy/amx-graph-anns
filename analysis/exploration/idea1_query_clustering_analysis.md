# Idea 1: Query Clustering Before Graph Search

## Hypothesis
Random queries scatter across the graph producing only 8.8% node coverage
at >=4 visitors/node. Similar queries traverse similar graph regions.
K-means clustering before search converts coincidental into structural sharing.

## Method
K-means on GIST1M query vectors (1000 queries). Measured node coverage
(>=4 queries/node) within each cluster using existing hop logs.

## Coverage Results
| K  | Cluster size | Coverage >=4 queries/node | vs Random baseline |
|----|-------------|---------------------------|-------------------|
| 5  | 200         | 55.2%                     | 6.3x better       |
| 10 | 100         | 47.0%                     | 5.3x better       |
| 20 | 50          | 32.6%                     | 3.7x better       |
| 50 | 20          | 16.7%                     | 1.9x better       |

## Why It Still Fails: Average Batch Size
Coverage is high but average batch size per hot node is only 7.6 queries
even at K=5. AMX speedup at batch=7 is ~1.2x (from amx_init_test).

## Amdahl Results
| K  | Coverage | Avg batch | AMX spd | Amdahl | Beats Track A? |
|----|----------|-----------|---------|--------|----------------|
| 5  | 58.6%    | 7.6       | 1.2x    | 1.108x | NO (vs 1.192x) |
| 10 | 48.9%    | 7.0       | 1.2x    | 1.089x | NO             |

## Conclusion
DEAD END. Coverage improves dramatically with clustering but batch size
per node stays too small (~7) for AMX to fire effectively. High coverage
x low AMX speedup < Track A's low coverage x high AMX speedup.

Root cause: even clustered queries spread across hundreds of distinct nodes
within their local graph region. Structural divergence persists even among
similar queries.

POTENTIAL FUTURE DIRECTION: Combine clustering with epoch-synchronous search
(Track B). Clustered queries in epoch-sync would have both high coverage AND
large batch sizes (all cluster members active simultaneously).
