# AMX Graph ANNS

Characterizing and accelerating graph-based approximate nearest
neighbor search with Intel AMX.

Phase 1: Faiss IndexHNSWFlat → AMX / GPU
Phase 2: DiskANN Vamana     → AMX / GPU

NJIT Honors Summer Research Initiative 2025
Advisor: Prof. Xiaoning Ding

## Structure
- faiss_instrumented/   Phase 1 hop-level instrumentation
- vamana_instrumented/  Phase 2 hop-level instrumentation
- analysis/             shared Python analysis and plotting
- results/              output figures and CSVs
- notes/                paper drafts and meeting notes
