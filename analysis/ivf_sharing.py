import numpy as np
import faiss
import pandas as pd
import matplotlib.pyplot as plt
import sys

def load_fvecs(path, max_n=-1):
    with open(path, 'rb') as f:
        data = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            data.append(vec)
            if max_n > 0 and len(data) >= max_n:
                break
    return np.array(data)

def main():
    base_path  = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/kingfeddy/repos/anns_research/sift/sift_base.fvecs"
    query_path = sys.argv[2] if len(sys.argv) > 2 else \
        "/home/kingfeddy/repos/anns_research/sift/sift_query.fvecs"

    DIM      = 128
    NLIST    = 1024
    NPROBES  = [8, 16, 32, 64, 128]
    N_QUERIES = 1000

    print("Loading base vectors...")
    base = load_fvecs(base_path)
    print(f"  Loaded {len(base)} vectors")

    print("Loading queries...")
    queries = load_fvecs(query_path, N_QUERIES)
    print(f"  Loaded {len(queries)} queries")

    print(f"Building IVF index (nlist={NLIST})...")
    quantizer = faiss.IndexFlatL2(DIM)
    index = faiss.IndexIVFFlat(quantizer, DIM, NLIST)
    index.train(base)
    index.add(base)
    print("  Index built")

    results = []

    for nprobe in NPROBES:
        index.nprobe = nprobe

        # get cluster assignments for all queries
        # quantizer.search returns top nprobe cluster IDs per query
        D, I = quantizer.search(queries, nprobe)

        # count how many queries probe each cluster
        cluster_counts = np.zeros(NLIST, dtype=int)
        for q in range(N_QUERIES):
            for c in I[q]:
                cluster_counts[c] += 1

	# cluster sharing analysis
        probed_clusters = cluster_counts[cluster_counts > 0]
        total_probings = N_QUERIES * nprobe
        unique_clusters = len(probed_clusters)

        # sharing rate: fraction of cluster probings that are redundant
        sharing_rate = 1.0 - unique_clusters / total_probings

        # avg queries per probed cluster
        avg_queries_per_cluster = probed_clusters.mean()

        # fraction of clusters shared by N+ queries
        pct_shared_2plus = (probed_clusters >= 2).mean() * 100
        pct_shared_4plus = (probed_clusters >= 4).mean() * 100
        pct_shared_8plus = (probed_clusters >= 8).mean() * 100

	results.append({
            "nprobe": nprobe,
            "probed_clusters": unique_clusters,
            "avg_queries_per_cluster": avg_queries_per_cluster,
            "sharing_rate": sharing_rate,
            "pct_shared_2plus": pct_shared_2plus,
            "pct_shared_4plus": pct_shared_4plus,
            "pct_shared_8plus": pct_shared_8plus,
        })

        print(f"  nprobe={nprobe}: {len(probed_clusters)} clusters probed, "
              f"avg {avg_queries_per_cluster:.1f} queries/cluster, "
              f"sharing={sharing_rate:.3f}")

    res = pd.DataFrame(results)
    print("\n", res.to_string(index=False))
    res.to_csv("results/ivf_sharing.csv", index=False)

    # plot sharing rate vs nprobe
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "IVF Cluster Sharing Across 1000 Queries — SIFT1M, nlist=1024",
        fontsize=13
    )

    ax1.plot(res["nprobe"], res["sharing_rate"] * 100, "b-o", markersize=8, linewidth=2)
    ax1.set_xlabel("nprobe", fontsize=11)
    ax1.set_ylabel("Cluster Sharing Rate (%)", fontsize=11)
    ax1.set_title("Sharing Rate vs nprobe", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.plot(res["nprobe"], res["avg_queries_per_cluster"], "r-o", markersize=8, linewidth=2)
    ax2.set_xlabel("nprobe", fontsize=11)
    ax2.set_ylabel("Avg Queries per Probed Cluster", fontsize=11)
    ax2.set_title("Avg Queries per Cluster vs nprobe", fontsize=12)
    ax2.grid(True, alpha=0.3)

    ax3.plot(res["nprobe"], res["pct_shared_2plus"],
             "g-o", markersize=8, linewidth=2, label="2+ queries")
    ax3.plot(res["nprobe"], res["pct_shared_4plus"],
             "b-o", markersize=8, linewidth=2, label="4+ queries")
    ax3.plot(res["nprobe"], res["pct_shared_8plus"],
             "r-o", markersize=8, linewidth=2, label="8+ queries")
    ax3.set_xlabel("nprobe", fontsize=11)
    ax3.set_ylabel("% of Probed Clusters", fontsize=11)
    ax3.set_title("Clusters Shared by N+ Queries", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig("results/ivf_sharing.png", dpi=150, bbox_inches="tight")
    print("Saved results/ivf_sharing.png")

if __name__ == "__main__":
    main()
