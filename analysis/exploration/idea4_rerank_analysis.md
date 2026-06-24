# Idea 4: Re-rank Stage Batching

## Hypothesis
DiskANN SSD uses cheap PQ traversal then full-precision re-ranking.
Batch all queries' re-rank computations into one dense GEMM.

## VTune Hotspot Profile (GIST1M, T=32, L=100)
| Function                          | CPU Time | % of Total |
|-----------------------------------|----------|------------|
| diskann::DistanceL2Float::compare | 6.487s   | 43.3%      |
| std::istream::read                | 4.400s   | 29.4%      |
| __kmp_fork_barrier                | 3.282s   | 21.9%      |
| diskann::InMemDataStore::get_dist | 0.010s   | 0.07%      |

## Conclusion
DEAD END. In-memory DiskANN computes full-precision L2 throughout — there
is no cheap-then-expensive two-stage scoring. The explicit re-rank stage
(get_distance) accounts for only 0.07% of runtime. Nothing to batch.

The disk-based DiskANN variant DOES have a re-rank stage. This idea would
be valid for DiskANN-SSD but is out of scope for this project.

## Implication
Runtime breakdown sets a hard ceiling on any distance-compute optimization:
- Distance compute: 43.3% → AMX can help this
- Graph reads:      29.4% → AMX cannot help this
- Synchronization:  21.9% → AMX cannot help this
- Other:             5.4%
Maximum possible speedup (perfect AMX): 1 / (0.567) = 1.76x absolute ceiling.
