# Idea A: Epoch-Synchronous Search

## Hypothesis
Keep all queries synchronized at hop boundaries. Each epoch: all active
queries expand one hop simultaneously → one BF16 GEMM per epoch → covers
100% of distance compute with AMX-eligible matrices.

## Why It Seemed Promising
- GIST1M convergence is remarkably uniform (min=99, max=138 hops, median=104)
- Barrier overhead at T=32: only 1.5% (measured)
- Batch size stays 998-1000 for hops 0-109

## The Fatal Flaw
"Batch size stays 998-1000" means 998-1000 queries are ACTIVE.
It does NOT mean they are computing distances to the SAME nodes.

Actual matrix at each epoch = (queries sharing that node) × 32 × 960.
From hop log analysis:
- Hop 0: 1000 queries, same medoid → 1000×32×960 ✓ AMX fires at 3.52x
- Hop 1: 1000 queries, ~30 unique nodes → avg 33 per node → 2.00x
- Hop 2+: 1000 queries, hundreds of unique nodes → avg 1.1 per node → 1.02x

Weighted AMX speedup: 1.063x. End-to-end after sync overhead: 1.011x.

## Conclusion
RETRACTED. Worse than Track A (1.034x). Epoch-sync does not help because
the divergence problem is in the node dimension, not the query dimension.
Adding barriers solves nothing — queries were already computing different
things, synchronizing them just adds overhead.
