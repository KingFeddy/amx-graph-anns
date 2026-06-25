import sys
import pandas as pd
import matplotlib.pyplot as plt

faiss_csv = sys.argv[1] if len(sys.argv) > 1 else "results/faiss_decay.csv"
vamana_csv = sys.argv[2] if len(sys.argv) > 2 else "results/vamana_decay.csv"
out = sys.argv[3] if len(sys.argv) > 3 else "results/phase1_phase2_decay.png"

fa = pd.read_csv(faiss_csv)
va = pd.read_csv(vamana_csv)

AMX_THRESHOLD = 0.05

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(fa["hop"], fa["sharing_rate"] * 100, "b-o", markersize=5, linewidth=2, label="Phase 1: Faiss IndexHNSWFlat")
ax.plot(va["hop"], va["sharing_rate"] * 100, "r-s", markersize=5, linewidth=2, label="Phase 2: DiskANN Vamana")
ax.axhline(AMX_THRESHOLD * 100, color="gray", linestyle="--", linewidth=1.5, label=f"AMX threshold ({AMX_THRESHOLD*100:.0f}%)")

for df, color, name in [(fa, "blue", "HNSW"), (va, "red", "Vamana")]:
    above = df[df["sharing_rate"] >= AMX_THRESHOLD]
    h_amx = int(above["hop"].max()) if len(above) else 0
    ax.axvline(h_amx, color=color, linestyle=":", linewidth=1.5, label=f"H_AMX ({name}) = {h_amx}")

ax.set_xlabel("Hop Depth", fontsize=12)
ax.set_ylabel("Neighbor Sharing Rate (%)", fontsize=12)
ax.set_title(
    "Batching Opportunity Decay: Faiss HNSW vs DiskANN Vamana\n"
    "SIFT1M, 1000 queries, ef/L_search = 100",
    fontsize=13
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
