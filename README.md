# AMX Graph ANNS

Characterizing and accelerating graph-based approximate nearest
neighbor search with Intel AMX.

NJIT Honors Summer Research Initiative 2026
Advisor: Prof. Xiaoning Ding

---

## Research Question

When multiple queries run simultaneously through a graph-based ANNS
index, do they evaluate the same neighbor nodes at the same traversal
depth? If so, those redundant distance computations can be fused into
a single AMX GEMM operation instead of computed independently.

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
Faiss HNSW in its current form.

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
At nprobe=32, essentially all cluster accesses are shared by 2+
queries, enabling near-perfect GEMM batching. This confirms CABANA's
findings on SIFT1M and validates our measurement methodology.

![IVF Cluster Sharing](results/ivf_sharing.png)

---

## Key Finding

Three algorithms show fundamentally different batching opportunity
profiles:

| Algorithm | Max Sharing | AMX Opportunity |
|-----------|-------------|-----------------|
| IVF | 87-99% | Extremely high |
| Vamana | 99.9% at hop 0, H_AMX=4 | Strong in early hops |
| HNSW | 6.5% at hop 0, H_AMX=0 | Negligible |

IVF has near-perfect sharing because its cluster structure guarantees
multiple queries probe the same vectors. This confirms CABANA and
validates our measurement methodology.

Vamana has strong sharing in early hops because all queries start
from the same fixed medoid node. At hop 0, all 1000 queries evaluate
the exact same 32 neighbors giving 99.9% sharing. Paths diverge
rapidly, dropping below 5% by hop 5.

HNSW has negligible sharing because its hierarchical design runs a
greedy descent through upper layers before layer 0 search begins.
This disperses queries to different entry points, eliminating sharing
before the main search even starts.

**Implication:** Vamana is the primary target for AMX acceleration.
The batch controller will exploit the H_AMX=4 window by fusing
distance computations across queries as GEMM for hops 0-4, falling
back to per-query GEMV beyond that depth.

![Phase 1 vs Phase 2 Comparison](results/phase1_phase2_decay.png)

---

## Repository Structure

- `faiss_instrumented/`   Faiss HNSW instrumentation patch and C++ driver
- `vamana_instrumented/`  DiskANN Vamana instrumentation patch and C++ driver
- `analysis/`             Decay curve analysis, IVF sharing analysis, and plotting scripts
- `results/`              Output CSVs and figures for all three algorithms
- `notes/`                Paper drafts and meeting notes
