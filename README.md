# AMX Graph ANNS

Characterizing and accelerating graph-based approximate nearest
neighbor search with Intel AMX.

NJIT Honors Summer Research Initiative 2026
Advisor: Prof. Xiaoning Ding

---

## Research Question

Can the AMX query-batching approach proven for IVF-based ANNS in
CABANA be extended to graph-based ANNS, and what structural properties
of graph algorithms determine the feasibility and magnitude of that
acceleration?

---

## Phase 1: Faiss IndexHNSWFlat

**Algorithm:** HNSW

**Library:** Faiss

**Dataset:** SIFT1M, 1M vectors, d=128

**Parameters:** M=16, ef_construction=200, ef_search=100

**Queries:** 1000

**Result:**

Sharing rate at hop 0: 6.47%

H_AMX at 5% threshold: 0

Max hop depth: 103

Sharing never exceeds the AMX efficiency threshold at any depth.
HNSW's upper layer greedy descent disperses queries to different
entry points before layer 0 search begins, eliminating sharing
immediately. AMX GEMM batching provides negligible benefit for
Faiss HNSW at 128 dimensions.

![Faiss HNSW Decay Curve](results/faiss_decay.png)

---

## Phase 2: DiskANN Vamana

**Algorithm:** Vamana

**Library:** DiskANN

**Dataset:** SIFT1M, 1M vectors, d=128

**Parameters:** R=32, L_build=125, alpha=1.2, L_search=100

**Queries:** 1000

**Result:**

Sharing rate at hop 0: 99.9%

Sharing rate at hop 1: 97.4%

Sharing rate at hop 2: 76.9%

Sharing rate at hop 3: 30.3%

Sharing rate at hop 4: 8.5%

H_AMX at 5% threshold: 4

Max hop depth: 107

All 1000 queries start from the same fixed medoid node and evaluate
the exact same 32 neighbors at hop 0, giving 99.9% sharing. Paths
diverge rapidly, dropping below 5% by hop 5. A strong AMX batching
window exists for hops 0 through 4.

![Vamana Decay Curve](results/vamana_decay.png)

---

## Phase 3: Faiss IVF (Baseline Validation)

**Algorithm:** IVF

**Library:** Faiss

**Dataset:** SIFT1M, 1M vectors, d=128

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
Confirms CABANA methodology and validates our measurement approach.
Peak: 363,840 QPS at nprobe=4 with 100% recall.

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

Peak 176,259 QPS at 128 threads. Degrades at 256 — memory bandwidth
saturation.

### IVF Threading Baseline (SIFT1M)

| nprobe | QPS     | Recall@10 |
|--------|---------|-----------|
| 1      | 204,491 | 94.90%    |
| 4      | 363,840 | 100%      |
| 8      | 181,729 | 100%      |

IVF upper bound: 363,840 QPS at nprobe=4.

### AMX GEMM Validation

Standalone GEMM benchmark confirms AMX tile instructions are
operational on Granite Rapids:

- 47x speedup over naive triple-loop at batch size 32
- Correctness verified: max absolute difference 0.000008 (PASS)
- Cache boundary at batch 64: sub-32 batches cache-resident ~2us,
  batches 64+ pay fixed ~12us but GFLOPS scales to 3440 at batch
  10000 with no plateau

### H_AMX Confirmation

Vamana decay curves regenerated on Granite Rapids confirm H_AMX=4
matches earlier results exactly. Sharing rate is a property of the
algorithm and dataset, not the hardware.

---

## Phase 5: Hardware Profiling

### perf stat — SIFT1M vs GIST1M

| Dataset | Threads | QPS     | Cache Miss Rate | LLC Miss Rate | IPC  |
|---------|---------|---------|----------------|---------------|------|
| SIFT1M  | 1       | 3,292   | 85.44%         | 72.43%        | 0.74 |
| SIFT1M  | 32      | 74,799  | 46.38%         | 40.58%        | 0.74 |
| SIFT1M  | 128     | 148,811 | 40.27%         | 35.60%        | 0.68 |
| GIST1M  | 1       | 928     | 78.25%         | 74.12%        | 0.80 |
| GIST1M  | 32      | 14,688  | 77.16%         | 81.02%        | 0.83 |
| GIST1M  | 128     | 20,929  | 75.94%         | 78.74%        | 0.83 |

SIFT1M cache miss rate drops from 85% to 40% as concurrency
increases — concurrent queries share cache lines. GIST1M stays flat
at ~77% because 960d vectors (3,840 bytes = 60 cache lines) are too
large for cache residency between accesses.

### VTune Hotspots — SIFT1M

| Threads | DistanceL2Float | kmp_fork_barrier | Bottleneck            |
|---------|-----------------|------------------|-----------------------|
| T=1     | 60.1%           | 0%               | Distance computation  |
| T=32    | 44.4%           | 25.6%            | Compute (transitioning)|
| T=128   | ~35%            | ~62%             | Thread imbalance      |

VTune memory access (T=32): L3+DRAM bound = 31.7% of clockticks.
Latency-bound from random neighbor vector access, not bandwidth-bound.
Average DRAM bandwidth: 21.1 GB/s of 684 GB/s peak (3%).

---

## Phase 6: Dimensionality and Scale Analysis

### Decay Curves Across Datasets and Algorithms

| Algorithm | Dataset   | Dims | Hop 0 Sharing | H_AMX |
|-----------|-----------|------|--------------|-------|
| Vamana    | SIFT1M    | 128  | 99.9%        | 4     |
| Vamana    | GIST1M    | 960  | 99.9%        | 24    |
| Vamana    | SIFT 100M | 128  | 99.9%        | 3     |
| HNSW      | SIFT1M    | 128  | 6.9%         | 0     |
| HNSW      | GIST1M    | 960  | 21.7%        | 17    |

H_AMX scales with dimensionality for both algorithms. At 960d, the
curse of dimensionality causes slower query path divergence, extending
the AMX-eligible window. Vamana maintains structural advantage at all
dimensionalities due to its fixed medoid entry point.

Scale effect: SIFT1M (1M) H_AMX=4 vs SIFT 100M (100M) H_AMX=3.
Scale has minor secondary influence; dimensionality dominates.

![Vamana GIST1M Decay Curve](results/granite_rapids/vamana_decay_gist.png)

![HNSW GIST1M Decay Curve](results/granite_rapids/hnsw_decay_gist.png)

### Query Distribution — H_AMX vs Batch Size

Hop-0 sharing follows (N-1)/N analytically. H_AMX by batch size:

| N    | SIFT1M H_AMX | GIST1M H_AMX |
|------|-------------|-------------|
| 1    | 0           | 0           |
| 10   | 1           | 1           |
| 50   | 2           | 2           |
| 100  | 2           | 4           |
| 500  | 3           | 7           |
| 1000 | 4           | 24          |

### Batch Size End-to-End Sensitivity (SIFT1M)

| Batch  | Threads | QPS     | Recall@10 |
|--------|---------|---------|-----------|
| 1      | 1       | 1,173   | 100.0%    |
| 10     | 10      | 3,246   | 99.0%     |
| 50     | 50      | 5,741   | 99.2%     |
| 500    | 128     | 25,082  | 98.8%     |
| 1,000  | 128     | 49,890  | 98.8%     |
| 10,000 | 128     | 161,315 | 99.1%     |

---

## Key Findings

| Algorithm | Dataset   | Max Sharing    | H_AMX | AMX Opportunity       |
|-----------|-----------|----------------|-------|-----------------------|
| IVF       | SIFT1M    | 87-99%         | N/A   | Extremely high        |
| Vamana    | SIFT1M    | 99.9% at hop 0 | 4     | Strong in early hops  |
| Vamana    | GIST1M    | 99.9% at hop 0 | 24    | Extends to 24 hops    |
| Vamana    | SIFT 100M | 99.9% at hop 0 | 3     | Scale effect minor    |
| HNSW      | SIFT1M    | 6.9% at hop 0  | 0     | Negligible at 128d    |
| HNSW      | GIST1M    | 21.7% at hop 0 | 17    | Moderate at 960d      |

**Dimensionality is the dominant factor.** H_AMX=4 at 128d,
H_AMX=24 at 960d for Vamana. Scale is secondary (H_AMX=4 at 1M,
H_AMX=3 at 100M).

**Hop-0 sharing is analytically derivable.** With N concurrent
queries all evaluating the same medoid's R neighbors:
sharing = (N-1)/N. This is exact, not empirical.

**Bottleneck shifts with thread count.** At T=1: distance
computation = 60%. At T=128: thread imbalance = 62%. Optimal
batch controller operating point: T=32.

**Vamana is the primary AMX target.** Fixed medoid guarantees
99.9% hop-0 sharing regardless of dataset. HNSW hierarchical
dispersal limits sharing at all dimensionalities.

**Implication:** The batch controller will exploit the H_AMX
window by fusing distance computations across concurrent queries
as GEMM for hops 0-H_AMX, falling back to per-query GEMV beyond.

---

## Repository Structure

- `faiss_instrumented/`   Faiss HNSW instrumentation patch and C++ driver
- `vamana_instrumented/`  DiskANN Vamana instrumentation patch and C++ driver
- `ivf_baseline/`         IVF baseline driver
- `amx/`                  AMX GEMM validation and batch scaling tests
- `analysis/`             Decay curve analysis, IVF sharing analysis, plotting scripts
- `results/`              Output CSVs and figures
- `results/granite_rapids/` Baseline QPS, decay curves, profiling on Granite Rapids
- `notes/`                Paper drafts and meeting notes
