import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_decay(df, min_queries=50):
    results = []
    for h in range(df["hop"].max() + 1):
        hop_df = df[df["hop"] == h]
        n_queries = hop_df["query_id"].nunique()
        if n_queries < min_queries:
            break
        total_evals = len(hop_df)
        unique_nodes = hop_df["neighbor_id"].nunique()
        sharing_rate = 1.0 - unique_nodes / total_evals
        results.append({
            "hop": h,
            "n_queries": n_queries,
            "total_evals": total_evals,
            "unique_nodes": unique_nodes,
            "sharing_rate": sharing_rate,
            "avg_evals": total_evals / n_queries,
        })
    return pd.DataFrame(results)

def find_h_amx(res, threshold=0.10):
    above = res[res["sharing_rate"] >= threshold]
    return int(above["hop"].max()) if len(above) else 0

def plot_single(res, label, h_amx, threshold, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        f"Batching Opportunity Decay — {label}\nSIFT1M, 1000 queries, ef/L=100",
        fontsize=13
    )

    ax1.plot(res["hop"], res["sharing_rate"] * 100, "b-o", markersize=5, linewidth=2)
    ax1.fill_between(res["hop"], res["sharing_rate"] * 100, alpha=0.12, color="blue")
    ax1.axhline(threshold * 100, color="red", linestyle="--", linewidth=1.5, label=f"AMX threshold ({threshold*100:.0f}%)")
    ax1.axvline(h_amx, color="orange", linestyle="--", linewidth=2, label=f"H_AMX = {h_amx}")
    ax1.set_ylabel("Sharing Rate (%)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.plot(res["hop"], res["n_queries"], "g-o", markersize=5)
    ax2.set_xlabel("Hop Depth", fontsize=11)
    ax2.set_ylabel("Queries Still Active", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--label", default="HNSW")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.10)
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  {len(df):,} entries | {df['query_id'].nunique()} queries | max hop {df['hop'].max()}")

    res = compute_decay(df)
    print(res.to_string(index=False))
    res.to_csv(args.output, index=False)

    h_amx = find_h_amx(res, args.threshold)
    print(f"\nH_AMX = {h_amx} (sharing >= {args.threshold*100:.0f}%)")

    plot_path = args.output.replace(".csv", ".png")
    plot_single(res, args.label, h_amx, args.threshold, plot_path)

if __name__ == "__main__":
    main()
