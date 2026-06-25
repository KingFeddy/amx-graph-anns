# Corrected Amdahl Analysis — Final Numbers

## Runtime Profile Corrections

### Error in Previous Analysis
Earlier VTune profiling on GIST1M showed std::istream::read at 29.4%.
This was misidentified as "graph read overhead during search."

Actual cause: the GIST1M binary loads the index (~2.4s) before searching
(~0.25s). With only 1000 queries, loading dominated the 2.69s total runtime.
The 29.4% "graph read overhead" was 90% index loading, not search.

### Corrected Search-Only Profiles

SIFT1M (10,000 queries, loading fraction negligible):
| Function              | CPU Time | % of Search |
|-----------------------|----------|-------------|
| DistanceL2Float       | 3.049s   | 39.9%       |
| kmp_fork_barrier      | 2.121s   | 27.7%       |
| iterate_to_fixed_point| 0.498s   | 6.5%        |
| std::vector copy      | 0.340s   | 4.4%        |
| Others                | 1.642s   | 21.5%       |

GIST1M (search-only estimate, loading excluded):
| Function              | % of Search |
|-----------------------|-------------|
| DistanceL2Float       | ~74%        |
| kmp_fork_barrier      | ~18%        |
| Others                | ~8%         |

Note: DiskANN already uses optimize_index_layout() with _mm_prefetch before
search — the memory layout is already optimized.

## All Corrected Amdahl Numbers

### Medoid hop-0 batching (hop-0 medoid GEMM only, AMX 3.52x at batch=1000)
| Dataset | Window% of evals | Window% of runtime | Amdahl  |
|---------|-----------------|-------------------|---------|
| SIFT1M  | 1.3%            | 0.52%             | 1.004x  |
| GIST1M  | 1.3%            | 0.96%             | 1.008x  |

### Medoid hop-0 batching, extended to hops 0-19 (weighted AMX 1.213x)
| Dataset | Window% of evals | Window% of runtime | Amdahl  |
|---------|-----------------|-------------------|---------|
| GIST1M  | 22.5%           | 16.7%             | 1.034x  |

### Neighbor List Copy Elimination (std::vector copy, 4.4% of SIFT runtime)
| Dataset | Fraction | Amdahl  |
|---------|----------|---------|
| SIFT1M  | 4.4%     | 1.046x  |
| GIST1M  | ~2-3%    | 1.020x  |

### Combined best case (copy elimination + medoid hop-0 batching, full window, GIST1M)
1.034x × 1.020x = **1.055x**

## Why Everything Is Bounded

The hard wall:
- Distance compute: 39.9% (SIFT) / 74% (GIST) — AMX can help
- Sync overhead: 27.7% (SIFT) / 18% (GIST) — AMX cannot help
- Bookkeeping: 6.5% / 8% — AMX cannot help

AMX-eligible fraction = window% × distance compute fraction
= 1.3% × 39.9% = 0.52% (SIFT, hop 0 only)
= 22.5% × 74% = 16.7% (GIST, full H_AMX window)

Even with infinite AMX speedup, GIST1M ceiling = 1/(1-0.167) = 1.20x.

## Key Research Insight
The contribution is not the speedup — it is the characterization of WHY
graph ANNS resists AMX:
1. AMX window is structurally small (medoid = only guaranteed fat batch)
2. Per-hop sharing drops from 1000 to 1.1 queries/node after hop 0-1
3. The remaining runtime (sync + bookkeeping) is AMX-impenetrable
4. arch_prctl must be called or AMX silently degrades to AVX-512
