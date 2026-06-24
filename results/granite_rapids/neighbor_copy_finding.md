# Unexpected Finding: Neighbor List Copy Overhead

## Discovery
VTune search-only profile (SIFT1M, 10K queries, T=32) revealed:
std::vector<unsigned int>::vector (copy constructor) at 4.4% of CPU time.

This is LARGER than the AMX-addressable window on SIFT1M (0.52%).

## Root Cause
In DiskANN's iterate_to_fixed_point, when expanding a node's neighbors,
the neighbor list is fetched via _graph_store->get_neighbours(n) which
returns a const reference to a std::vector<uint32_t>.

However, downstream code copies this vector into scratch space, triggering
the copy constructor for potentially R=32 uint32_t values per hop,
repeated for every hop of every query.

## Scale
- 2,141 avg distance comparisons per SIFT1M query (= node expansions)
- Each expansion copies ~32 uint32_t values = 128 bytes
- 10,000 queries × 2,141 copies × 128 bytes = ~2.74GB of copies per run
- At memory bandwidth, this is pure overhead

## Potential Fix
Replace copy with const reference or span where the neighbor list is only
read, not modified. This is a pure correctness-preserving refactor.
Estimated speedup: 4-5% on SIFT1M (larger AMX improvement than Track A).

## Status
Identified but not implemented. Implementation risk: low (pure refactor).
Recommended as Track C in future work.
