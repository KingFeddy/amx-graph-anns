# Track B: Node-Level Batching Analysis

## Hypothesis
Ding's FedEx idea: batch individual node-expansion operations across queries
dynamically at runtime. Even after queries diverge, some pairs coincidentally
visit the same node simultaneously and can share a GEMM.

## Method
Used GIST1M and SIFT1M hop logs (1000 queries each).
For each node, counted how many distinct queries visited it across all hops.
Measured what fraction of total evaluations fall in "hot" nodes (>=T visitors).

## Results: SIFT1M (128d)
| Min queries/node | Evals covered | % of total | Avg batch |
|-----------------|---------------|------------|-----------|
| 2               | 80.3%         | 80.3%      | 3.5       |
| 8               | 11.2%         | 11.2%      | 12.5      |
| 32              | 2.6%          | 2.6%       | 152.3     |
| 64              | 2.4%          | 2.4%       | 209.9     |

## Results: GIST1M (960d)
| Min queries/node | Evals covered | % of total | Avg batch |
|-----------------|---------------|------------|-----------|
| 2               | 87.0%         | 87.0%      | 5.1       |
| 8               | 40.0%         | 40.0%      | 14.4      |
| 32              | 6.7%          | 6.7%       | 57.3      |
| 64              | 2.6%          | 2.6%       | 148.2     |

## Scaling Test (simulating more queries)
Coverage plateaus at ~2-8% regardless of batch size (up to 10,000 simulated).
Root cause: 1000 queries visit 760,512 unique nodes on GIST1M. Graph search
is designed to guide each query to its own target — divergence is intentional.

## Amdahl Projection
At AMX-eligible threshold (>=32 queries/node):
- SIFT1M: 2.6% coverage → 1.01x end-to-end (vs Track A 1.028x)
- GIST1M: 6.7% coverage → 1.05x end-to-end (vs Track A 1.192x)

## Conclusion
DEAD END at 1000 queries. Coverage is too sparse because graph search
is structurally designed to diverge queries. The coincidental node overlap
is inherently low — not a solvable engineering problem at this query count.
Would potentially improve at very high traffic (100K+ concurrent queries)
but that is outside the scope of this project.
