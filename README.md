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

**Algorithm:** HNSW (Malkov & Yashunin 2016)
**Library:** Faiss (Meta)
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

**Algorithm:** Vamana (Subramanya et al. 2019)
**Library:** DiskANN (Microsoft)
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

## Key Finding

The two algorithms show fundamentally different batching opportunity
profiles driven by their entry point mechanisms.

Vamana uses a single fixed medoid as the entry point for all queries.
Every query starts from the same node, creating near-perfect sharing
in early hops before paths diverge.

HNSW uses a hierarchical design where queries descend through upper
layers before reaching layer 0. This descent sends each query to a
different entry point, eliminating sharing before the main search
even begins.

**Implication:** Vamana is the primary target for AMX acceleration.
The batch controller will exploit the H_AMX=4 window by fusing
distance computations across queries as GEMM for hops 0-4, falling
back to per-query GEMV beyond that depth.

![Phase 1 vs Phase 2 Comparison](results/phase1_phase2_decay.png)

---

## Repository Structure

- `faiss_instrumented/`   Phase 1 instrumentation and driver
- `vamana_instrumented/`  Phase 2 instrumentation and driver
- `analysis/`             Decay curve analysis and plotting scripts
- `results/`              Output CSVs and figures
- `notes/`                Paper drafts and meeting notes
