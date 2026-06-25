# Approach: Re-rank Stage Batching

## Hypothesis
DiskANN SSD uses cheap PQ traversal then full-precision re-ranking.
Batch all queries' re-rank computations into one dense GEMM.

## VTune Hotspot Profile

Two profiles are relevant. The first is the raw GIST1M run; the second
corrects for index-loading contamination (see corrected_amdahl_analysis.md).

Raw GIST1M (T=32, L=100, 1K queries) — CONTAMINATED by index-loading time:
| Function                          | CPU Time | % of Total |
|-----------------------------------|----------|------------|
| diskann::DistanceL2Float::compare | 6.487s   | 43.3%      |
| std::istream::read                | 4.400s   | 29.4%      |
| __kmp_fork_barrier                | 3.282s   | 21.9%      |
| diskann::InMemDataStore::get_dist | 0.010s   | 0.07%      |

The 29.4% std::istream::read is almost entirely index-loading, not search:
with only 1K GIST queries, the ~2.4s load dominates the ~0.25s search. The
corrected search-only fractions are distance ~74% / sync ~18% / other ~8%
(GIST1M) and distance 39.9% / sync 27.7% / other 32.4% (SIFT1M, 10K queries).
The get_distance (re-rank) figure of 0.07% is unaffected by this correction —
it is a search-time measurement and stands.

## Conclusion
DEAD END. In-memory DiskANN computes full-precision L2 throughout — there
is no cheap-then-expensive two-stage scoring. The explicit re-rank stage
(get_distance) accounts for only 0.07% of runtime. Nothing to batch.

The disk-based DiskANN variant DOES have a re-rank stage. This idea would
be valid for DiskANN-SSD but is out of scope for this project.

## Implication
The corrected search-only breakdown sets the hard ceiling on any
distance-compute optimization. Using the GIST1M search-only profile:
- Distance compute: ~74% → AMX can help this
- Synchronization:  ~18% → AMX cannot help this
- Other:             ~8% → AMX cannot help this
Maximum possible speedup with perfect AMX on the full distance fraction:
1 / (1 - 0.74) = 3.8x absolute ceiling (GIST1M). For SIFT1M, distance is
only 39.9% of search, giving 1 / (1 - 0.399) = 1.66x. These are theoretical
ceilings assuming AMX accelerates 100% of distance compute; the realized
the medoid-batch window is far smaller (hop-0 only), which is why measured end-to-end
is ~1.0x. See corrected_amdahl_analysis.md.
