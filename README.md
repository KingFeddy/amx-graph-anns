# AMX Graph ANNS

Characterizing and accelerating graph-based approximate nearest
neighbor search with Intel AMX.

NJIT Honors Summer Research Initiative 2026
Advisor: Prof. Xiaoning Ding

---

## Motivation

Vector databases power modern AI — RAG pipelines, recommendation
systems, semantic search. The core operation is Approximate Nearest
Neighbor Search (ANNS): given a query vector, find the K most similar
vectors among millions. The bottleneck is distance computation:
each query computes distances to hundreds of database vectors, one
small matrix-vector multiply (GEMV) at a time. These operations are
memory-bound and leave modern CPU compute idle.

Intel AMX (Advanced Matrix Extensions) on Granite Rapids can execute
large matrix-matrix multiplies (GEMM) at extremely high throughput —
but only when many queries compute distances to the same database
vectors simultaneously, so the work can be fused into one GEMM.

CABANA (IEEE CAL 2025) proved this works for IVF-based ANNS, where
the cluster structure guarantees queries share vectors. Whether it
works for graph-based ANNS — the dominant index type in production —
was an open question. Graph traversal is query-dependent: each query
walks its own path. Do concurrent queries ever touch the same nodes
at the same time?

## Research Question

Can the AMX query-batching approach proven for IVF-based ANNS in
CABANA be extended to graph-based ANNS, and what structural properties
of graph algorithms determine the feasibility and magnitude of that
acceleration?

---

## Key Concepts

**Sharing rate at hop h** — the fraction of distance computations at
traversal depth h that are redundant across concurrent queries:
sharing_rate(h) = 1 - unique_nodes_evaluated / total_evaluations

High sharing means multiple queries evaluate the same node at the
same depth — those computations can be fused into one AMX GEMM.

**H_AMX** — the last hop where sharing rate exceeds the 5% usefulness
threshold. The batch controller uses AMX GEMM for hops 0 through
H_AMX and falls back to per-query GEMV beyond that depth. H_AMX is
the size of the AMX opportunity window.

**Methodology** — we instrumented the distance-computation hot path
of DiskANN (`iterate_to_fixed_point` in index.cpp) and Faiss
(`search_from_candidates` in HNSW.cpp) at C++ source level. Every
distance computation is logged with (query_id, hop_depth, node_id)
using thread-local query tagging. 1000 concurrent queries per run.
Index parameters held constant across all datasets (R=32, L_build=125,
alpha=1.2) so dimensionality and scale are isolated variables.

---

## Results at a Glance

| Algorithm | Dataset   | Dims | Hop 0 Sharing | H_AMX | AMX Opportunity      |
|-----------|-----------|------|--------------|-------|----------------------|
| IVF       | SIFT1M    | 128  | 87-99%       | N/A   | Extremely high       |
| Vamana    | SIFT1M    | 128  | 99.9%        | 4     | Strong in early hops |
| Vamana    | GIST1M    | 960  | 99.9%        | 24    | Extends to 24 hops   |
| Vamana    | SIFT 100M | 128  | 99.9%        | 3     | Scale-invariant      |
| HNSW      | SIFT1M    | 128  | 6.9%         | 0     | None                 |
| HNSW      | GIST1M    | 960  | 21.7%        | 17    | Emerges at high dims |

Three headline findings:

1. **Dimensionality is the dominant factor.** H_AMX grows 6x from
   128d to 960d. Modern AI embeddings (768d-1536d) sit exactly where
   AMX batching matters most.
2. **Scale barely matters.** 100x more vectors moves H_AMX by one hop.
   The opportunity is architectural, not data-dependent.
3. **Vamana structurally dominates HNSW at every dimensionality**
   (4 vs 0 at 128d, 24 vs 17 at 960d) because its fixed medoid entry
   point guarantees hop-0 sharing.

---

## Phase 1: Faiss IndexHNSWFlat

**Algorithm:** HNSW

**Library:** Faiss v1.7.4

**Dataset:** SIFT1M, d=128

**Parameters:** M=16, ef_construction=200, ef_search=100

**Queries:** 1000

**Result:**

Sharing rate at hop 0: 6.47%

H_AMX at 5% threshold: 0

Max hop depth: 103

Sharing never exceeds the AMX efficiency threshold at any depth.
HNSW's upper layer greedy descent disperses queries to different
entry points before layer 0 search begins, eliminating sharing
immediately. Each query's descent path depends on its own vector,
so by the time queries reach layer 0 they are scattered across
hundreds of different graph regions. AMX GEMM batching provides
negligible benefit for Faiss HNSW at 128 dimensions.

![Faiss HNSW Decay Curve](results/faiss_decay.png)

---

## Phase 2: DiskANN Vamana

**Algorithm:** Vamana

**Library:** DiskANN 0.7.0

**Dataset:** SIFT1M, d=128

**Parameters:** R=32, L_build=125, alpha=1.2, L_search=100

**Queries:** 1000

**Result:**

| Hop | Sharing Rate |
|-----|-------------|
| 0   | 99.9%       |
| 1   | 97.4%       |
| 2   | 76.9%       |
| 3   | 30.3%       |
| 4   | 8.5%        |
| 5+  | <5%         |

H_AMX at 5% threshold: 4

Max hop depth: 107

All 1000 queries start from the same fixed medoid node and evaluate
the exact same 32 neighbors at hop 0, giving 99.9% sharing. Paths
diverge rapidly, dropping below 5% by hop 5. A strong AMX batching
window exists for hops 0 through 4.

**Analytical result:** hop-0 sharing follows (N-1)/N exactly. With
N concurrent queries all evaluating the same medoid's 32 neighbors:
total evaluations = 32N, unique nodes = 32, so
sharing = (32N - 32)/32N = (N-1)/N. Verified empirically at every
batch size (N=10 gives exactly 90.0%, N=100 gives exactly 99.0%).
The hop-0 AMX opportunity is derivable from first principles.

![Vamana Decay Curve](results/vamana_decay.png)

---

## Phase 3: Faiss IVF (Baseline Validation)

**Algorithm:** IVF

**Library:** Faiss

**Dataset:** SIFT1M, d=128

**Parameters:** nlist=1024, nprobe sweep [8, 16, 32, 64, 128]

**Queries:** 1000

**Result:**

| nprobe | Sharing Rate | Clusters shared by 2+ | Clusters shared by 8+ |
|--------|-------------|----------------------|----------------------|
| 8      | 87.3%       | 97.3%                | 47.6%                |
| 16     | 93.6%       | 99.7%                | 85.3%                |
| 32     | 96.8%       | 99.9%                | 96.1%                |
| 64     | 98.4%       | 100%                 | 99.4%                |
| 128    | 99.2%       | 100%                 | 100%                 |

IVF shows extremely high cluster sharing across all nprobe values.
This reproduces CABANA's findings on SIFT1M and validates our
measurement methodology — when the same instrumentation applied to
graph algorithms gives different answers, those answers are credible.

![IVF Cluster Sharing](results/ivf_sharing.png)

---

## Phase 4: Granite Rapids Hardware Validation

**Hardware:** Intel Xeon 6767P (Granite Rapids), 256 cores, 251GB RAM
**AMX:** amx_bf16, amx_tile, amx_int8 confirmed

### Vamana Threading Baseline (SIFT1M)

| Threads | QPS     | Recall@10 |
|---------|---------|-----------|
| 1       | 4,589   | 99.14%    |
| 2       | 8,566   | 99.14%    |
| 4       | 12,915  | 99.14%    |
| 8       | 23,093  | 99.14%    |
| 16      | 43,377  | 99.14%    |
| 32      | 80,211  | 99.14%    |
| 64      | 142,097 | 99.14%    |
| 128     | 176,259 | 99.14%    |
| 256     | 109,964 | 99.14%    |

Peak throughput at 128 threads. Performance degrades at 256 threads
due to memory bandwidth saturation, confirming graph ANNS is
memory-bound — more compute does not help; the memory system is the
wall. AMX batching targets this bottleneck by increasing compute
intensity per memory load.

### IVF Threading Baseline (SIFT1M)

| nprobe | QPS     | Recall@10 |
|--------|---------|-----------|
| 1      | 204,491 | 94.90%    |
| 4      | 363,840 | 100%      |
| 8      | 181,729 | 100%      |

IVF upper bound: 363,840 QPS at nprobe=4 with perfect recall — the
proof of what this hardware can do when memory access patterns are
favorable, and the target the batch controller is chasing.

### Additional Threading Baselines

| Dataset   | Peak QPS | Threads | Recall@10 | Notes                       |
|-----------|----------|---------|-----------|-----------------------------|
| GIST1M    | 23,005   | 128     | 88.50%    | 960d, avg degree 24.3       |
| SIFT 100M | 111,069  | 256     | 93.58%    | GT computed on 100M subset  |

GIST1M recall is lower at the same L=100 because the 960d graph is
sparser (avg degree 24.3 vs 30.2) and search is harder; recall is
tunable with higher L and does not affect sharing structure.

### AMX GEMM Validation

Standalone GEMM benchmark confirms AMX tile instructions are
operational on Granite Rapids:

- 47x speedup over naive triple-loop at batch size 32
- Correctness verified: max absolute difference 0.000008 (PASS)
- Cache boundary at batch 64: sub-32 batches are cache-resident at
  ~2us, batches 64+ pay a fixed ~12us memory cost but GFLOPS scales
  continuously to 3441 at batch 10000 with no plateau visible

**Implication for batch controller:** two operating regimes exist.
Fire at batch <=32 for latency-sensitive workloads. Accumulate
1000+ queries for maximum throughput.

### H_AMX Confirmation

Vamana decay curves regenerated on Granite Rapids confirm H_AMX=4
matches earlier results exactly. Sharing rate is a property of the
algorithm and dataset, not the hardware.

---

## Phase 5: Hardware Profiling

This phase identifies the mechanism behind every number above.

### perf stat — Cache Behavior vs Concurrency

| Dataset | Threads | QPS     | Cache Miss Rate | LLC Miss Rate | IPC  |
|---------|---------|---------|----------------|---------------|------|
| SIFT1M  | 1       | 3,292   | 85.44%         | 72.43%        | 0.74 |
| SIFT1M  | 32      | 74,799  | 46.38%         | 40.58%        | 0.74 |
| SIFT1M  | 128     | 148,811 | 40.27%         | 35.60%        | 0.68 |
| GIST1M  | 1       | 928     | 78.25%         | 74.12%        | 0.80 |
| GIST1M  | 32      | 14,688  | 77.16%         | 81.02%        | 0.83 |
| GIST1M  | 128     | 20,929  | 75.94%         | 78.74%        | 0.83 |

Two qualitatively different behaviors:

**SIFT1M (128d):** cache miss rate drops from 85% to 40% as
concurrency increases. Concurrent queries accidentally share cache
lines — query B finds the vector query A just loaded still resident.
IPC stays flat, confirming the bottleneck is memory latency, not
compute.

**GIST1M (960d):** cache miss rate stays flat at ~77% regardless of
thread count. Each 960d vector is 3,840 bytes = 60 cache lines; one
hop evaluation touches 1,920 cache lines. Vectors are evicted before
another query can reuse them — accidental cache sharing physically
cannot happen. **At high dimensionality, explicit AMX batching is
the only mechanism that can amortize vector loads.**

### VTune Hotspots — The Bottleneck Shift

| Threads | DistanceL2Float | kmp_fork_barrier | Dominant Bottleneck     |
|---------|-----------------|------------------|-------------------------|
| T=1     | 60.1%           | 0%               | Distance computation    |
| T=32    | 44.4%           | 25.6%            | Compute (transitioning) |
| T=128   | ~35%            | ~62%             | Thread imbalance        |

The bottleneck flips from compute to synchronization as thread count
grows. Barrier time is threads finishing queries at different speeds
and waiting for each other. AMX accelerates distance computation, so
its leverage is highest where compute still dominates: **T=32-64 is
the batch controller's optimal operating zone.** At T=128, Amdahl's
law caps gains — accelerating 35% of runtime cannot exceed 1.5x.
This is also a stated limitation: AMX does not fix thread imbalance,
which becomes the next bottleneck.

### VTune Memory Access (T=32)

- Memory Bound: 36.5% of pipeline slots
- L3 Bound: 15.2% + DRAM Bound: 16.5% = **31.7% of clockticks
  stalled on cache misses**
- DRAM bandwidth: 21.1 GB/s average of 684 GB/s platform peak (3%)

High stalls plus low bandwidth utilization is the signature of
**latency-bound random access**, not streaming. Each miss is a random
pointer chase to a neighbor vector. AMX batching converts N random
loads of the same vector into 1 load reused N times — it attacks
exactly the 31.7% of stalled clockticks.

---

## Phase 6: Dimensionality, Scale, and Workload Analysis

### Decay Curves Across Datasets and Algorithms

| Algorithm | Dataset   | Dims | H_AMX |
|-----------|-----------|------|-------|
| Vamana    | SIFT1M    | 128  | 4     |
| Vamana    | GIST1M    | 960  | 24    |
| Vamana    | SIFT 100M | 128  | 3     |
| HNSW      | SIFT1M    | 128  | 0     |
| HNSW      | GIST1M    | 960  | 17    |

**Dimensionality effect (controlled scale):** SIFT1M vs GIST1M, both
1M vectors, only dims change. H_AMX grows 4 -> 24. The mechanism is
distance concentration: at 960d all pairwise distances become more
uniform, so the pull toward each query's individual target is weaker
relative to graph structure — paths stay correlated for many more
hops before diverging.

**Scale effect (controlled dimensionality):** SIFT1M vs SIFT 100M,
both 128d, only scale changes. H_AMX moves 4 -> 3. A larger graph
offers more distinct routes, so paths diverge marginally faster, but
the effect is one hop versus twenty. Scale is secondary.

**HNSW at high dims:** H_AMX goes 0 -> 17 from 128d to 960d. The
hierarchical dispersal that kills sharing at 128d weakens at 960d
because the upper-layer greedy descent converges to similar regions
when all distances look similar. Dimensionality drives AMX
opportunity for all graph algorithms — Vamana's fixed medoid just
gives it a structural lead at every dimensionality.

![Vamana GIST1M Decay Curve](results/granite_rapids/vamana_decay_gist.png)

![HNSW GIST1M Decay Curve](results/granite_rapids/hnsw_decay_gist.png)

### Query Distribution — H_AMX vs Batch Size

| N    | SIFT1M H_AMX | GIST1M H_AMX |
|------|-------------|-------------|
| 1    | 0           | 0           |
| 10   | 1           | 1           |
| 50   | 2           | 2           |
| 100  | 2           | 4           |
| 500  | 3           | 7           |
| 1000 | 4           | 24          |

H_AMX itself scales with batch size. A single query has zero AMX
opportunity — sharing requires concurrency to exist. N>=10 is the
floor of usefulness; full benefit arrives at N=1000. At every batch
size, GIST1M >= SIFT1M, reconfirming the dimensionality effect.

This directly maps to workload scenarios: sequential query arrival
(N=1) gets no AMX benefit; bursty arrival gets partial benefit
proportional to burst size; high-concurrency serving gets the full
window.

### Batch Size End-to-End Sensitivity

**SIFT1M:**

| Batch  | Threads | QPS     | Recall@10 |
|--------|---------|---------|-----------|
| 1      | 1       | 1,173   | 100.0%    |
| 10     | 10      | 3,246   | 99.0%     |
| 50     | 50      | 5,741   | 99.2%     |
| 500    | 128     | 25,082  | 98.8%     |
| 1,000  | 128     | 49,890  | 98.8%     |
| 10,000 | 128     | 161,315 | 99.1%     |

**GIST1M:**

| Batch  | Threads | QPS    | Recall@10 |
|--------|---------|--------|-----------|
| 1      | 1       | 598    | 100.0%    |
| 10     | 10      | 1,358  | 92.0%     |
| 500    | 128     | 11,628 | 88.5%     |
| 1,000  | 128     | 22,041 | 88.5%     |

QPS spans 137x (SIFT1M) and 37x (GIST1M) from concurrency alone,
before any AMX. This is the no-AMX baseline curve the batch
controller must beat. GIST1M scales less because each query is
already expensive at 960d, limiting parallelism gains.

---

## Synthesis: When Does AMX Help Graph ANNS?

**AMX helps when:**
- The algorithm has a shared entry structure (Vamana's fixed medoid)
- Concurrent query count is >= 10, with full benefit at 1000+
- Operating at moderate thread counts (T=32-64) where distance
  computation still dominates runtime
- Especially at high dimensionality (768d+), where vectors are too
  large for accidental cache sharing and explicit batching is the
  only amortization mechanism

**AMX does not help when:**
- Queries arrive one at a time (no sharing possible)
- The algorithm disperses entry points (HNSW at low dims)
- Thread count is high enough that synchronization, not compute,
  dominates (T=128+)

**Why it helps (the mechanism):**
- Graph ANNS wastes 31.7% of clockticks on latency-bound random
  memory access
- Batching converts N redundant loads of each neighbor vector into
  1 load reused across all N queries
- AMX GEMM then executes the fused computation at up to 47x the
  throughput of independent GEMVs

---

## Repository Structure

- `faiss_instrumented/`     Faiss HNSW instrumentation patch and C++ driver
- `vamana_instrumented/`    DiskANN Vamana instrumentation patch and C++ driver
- `ivf_baseline/`           IVF baseline driver
- `amx/`                    AMX GEMM validation and batch scaling tests
- `analysis/`               Decay curve analysis, IVF sharing analysis, plotting scripts
- `results/`                Output CSVs and figures
- `results/granite_rapids/` Baseline QPS, decay curves, and profiling on Granite Rapids
- `notes/`                  Paper drafts and meeting notes

## Reproducibility

All instrumentation is captured as git patches that apply cleanly to
fresh clones of DiskANN 0.7.0 and Faiss v1.7.4. Index build
parameters are constant everywhere (R=32, L_build=125, alpha=1.2,
T=64). Datasets come from ann-benchmarks.com (SIFT1M, GIST1M HDF5)
and dl.fbaipublicfiles.com (SIFT1B u8bin); conversion scripts are in
`analysis/`. SIFT 100M ground truth is computed against the exact
100M subset using DiskANN's `compute_groundtruth`. Drivers infer
vector dimensionality from input files and take explicit index paths
to prevent cross-dataset contamination.
