# Idea 2: Higher Graph Degree R for Better AMX Matrices

## Hypothesis
Track A's hop-0 GEMM matrix is R x batch x D. Larger R means larger M
dimension, improving AMX tile utilization. Rebuilt indexes at R=64, R=128.

## AMX Speedup Results (amx_r_sweep, single thread, batch=1000, arch_prctl enabled)
| R   | DIM | Matrix          | sgemm(ms) | BF16/AMX(ms) | Speedup |
|-----|-----|-----------------|-----------|--------------|---------|
| 32  | 128 | 32x1000x128     | 0.046     | 0.015        | 3.19x   |
| 64  | 128 | 64x1000x128     | 0.080     | 0.022        | 3.73x   |
| 128 | 128 | 128x1000x128    | 0.153     | 0.036        | 4.32x   |
| 32  | 960 | 32x1000x960     | 0.359     | 0.106        | 3.40x   |
| 64  | 960 | 64x1000x960     | 0.659     | 0.149        | 4.44x   |
| 128 | 960 | 128x1000x960    | 1.276     | 0.253        | 5.05x   |

## Window Fraction and Corrected Amdahl (search-only profile)
GIST1M search-only: distance compute ~74%, sync ~18%, other ~8%

| Config      | H_AMX | Window% | AMX speedup | Corrected Amdahl |
|-------------|-------|---------|-------------|-----------------|
| GIST R=32   | 19    | 22.5%   | 3.52x (h0)  | 1.034x (full)   |
| GIST R=64   | 34    | 39.5%   | 4.44x (h0)  | 1.060x (full)   |

Note: full window Amdahl uses weighted AMX across all hops (mostly 1.02x
after hop 0-1), not the hop-0 speedup alone.

## QPS vs Recall (GIST1M baseline, no AMX)
| Config      | QPS    | Recall@10 |
|-------------|--------|-----------|
| R=32 L=100  | 21,099 | 88.31%    |
| R=64 L=50   | 5,033  | 89.01%    |
| R=64 L=100  | 4,323  | 94.80%    |
| R=128 L=100 | 3,757  | 97.37%    |

## Conclusion
DEAD END for absolute QPS. Higher R drops baseline QPS 4x faster than
AMX recovers it. Net result: lower absolute throughput at matched recall.

KEY FINDING: AMX benefit scales with R. This is a characterization result —
for users who already need R=64+ for high recall requirements, AMX is
significantly more valuable (1.060x vs 1.034x). Published as finding, not
as a performance optimization.
