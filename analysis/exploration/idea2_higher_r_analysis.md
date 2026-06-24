# Idea 2: Higher Graph Degree R for Better AMX Matrices

## Hypothesis
Track A's hop-0 GEMM matrix is R x batch x D. Larger R means larger M
dimension, which improves AMX tile utilization. Rebuilt indexes at R=64, R=128.

## AMX Speedup Results (amx_r_sweep, single thread, batch=1000)
| R   | DIM | Matrix          | sgemm(ms) | BF16/AMX(ms) | Speedup |
|-----|-----|-----------------|-----------|--------------|---------|
| 32  | 128 | 32x1000x128     | 0.046     | 0.015        | 3.19x   |
| 64  | 128 | 64x1000x128     | 0.080     | 0.022        | 3.73x   |
| 128 | 128 | 128x1000x128    | 0.153     | 0.036        | 4.32x   |
| 32  | 960 | 32x1000x960     | 0.359     | 0.106        | 3.40x   |
| 64  | 960 | 64x1000x960     | 0.659     | 0.149        | 4.44x   |
| 128 | 960 | 128x1000x960    | 1.276     | 0.253        | 5.05x   |

## Window Fraction Results
| Config      | H_AMX | Window% | AMX speedup | Amdahl  |
|-------------|-------|---------|-------------|---------|
| SIFT R=32   | 4     | 7.1%    | 1.62x       | 1.028x  |
| SIFT R=64   | 4     | 8.4%    | 3.73x       | 1.066x  |
| GIST R=32   | 19    | 22.5%   | 3.52x       | 1.192x  |
| GIST R=64   | 34    | 39.5%   | 4.44x       | 1.441x  |

## QPS vs Recall (GIST1M, baseline without AMX)
| Config        | QPS    | Recall@10 |
|---------------|--------|-----------|
| R=32 L=100    | 21,099 | 88.31%    |
| R=64 L=50     | 5,033  | 89.01%    |
| R=64 L=100    | 4,323  | 94.80%    |
| R=128 L=100   | 3,757  | 97.37%    |

## Projected QPS With AMX (at matched recall ~88%)
- R=32 L=100: 21,099 x 1.192 = 25,150 QPS
- R=64 L=50:   5,033 x 1.441 =  7,253 QPS

## Conclusion
DEAD END for absolute QPS improvement. Higher R dramatically drops baseline
QPS (more neighbors per hop = more total work) faster than AMX recovers it.

KEY FINDING: The AMX benefit SCALES with R. At R=64 GIST1M, the Amdahl
ceiling rises to 1.441x (vs 1.290x at R=32). For users who ALREADY want
high recall requiring R=64+, AMX is significantly more valuable.
This is a publishable characterization finding even though it does not
improve absolute throughput.
