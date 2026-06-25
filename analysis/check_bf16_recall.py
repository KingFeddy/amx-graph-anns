"""
check_bf16_recall.py
Tests whether BF16 quantization of query/neighbor vectors changes Recall@10.
Method: convert queries to BF16 and back to FP32 (simulating quantization noise),
then compare DiskANN search results against FP32 baseline.
"""
import numpy as np
import subprocess
import struct
import os
import sys

def read_fvecs(path):
    with open(path, 'rb') as f:
        data = f.read()
    offset = 0
    vecs = []
    while offset < len(data):
        dim = struct.unpack_from('i', data, offset)[0]
        offset += 4
        vec = struct.unpack_from(f'{dim}f', data, offset)
        offset += dim * 4
        vecs.append(vec)
    return np.array(vecs, dtype=np.float32)

def read_ivecs(path):
    with open(path, 'rb') as f:
        data = f.read()
    offset = 0
    vecs = []
    while offset < len(data):
        dim = struct.unpack_from('i', data, offset)[0]
        offset += 4
        vec = struct.unpack_from(f'{dim}i', data, offset)
        offset += dim * 4
        vecs.append(vec)
    return np.array(vecs, dtype=np.int32)

def to_fvecs(arr, path):
    with open(path, 'wb') as f:
        for v in arr:
            f.write(struct.pack('i', len(v)))
            f.write(v.astype(np.float32).tobytes())

def f32_to_bf16_to_f32(arr):
    """Simulate BF16 quantization: truncate lower 16 bits of float32 mantissa."""
    as_int = arr.view(np.uint32)
    bf16_int = (as_int >> 16).astype(np.uint16)
    back_int = bf16_int.astype(np.uint32) << 16
    return back_int.view(np.float32)

def run_diskann_search(index_prefix, query_file, gt_file, result_path, K=10, L=100, T=64):
    cmd = [
        os.path.expanduser('~/DiskANN/build/apps/search_memory_index'),
        '--data_type', 'float',
        '--dist_fn', 'l2',
        '--index_path_prefix', index_prefix,
        '--query_file', query_file,
        '--gt_file', gt_file,
        '--result_path', result_path,
        '-K', str(K),
        '-L', str(L),
        '-T', str(T)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse recall from output
    for line in result.stdout.split('\n'):
        if 'Recall@' in line or 'recall' in line.lower():
            print(f"  DiskANN: {line.strip()}")
    return result.stdout

def compute_recall(result_file_prefix, gt, K=10):
    """Compute Recall@K by reading DiskANN result files."""
    result_file = f"{result_file_prefix}_idx_{K}.bin"
    if not os.path.exists(result_file):
        return None
    with open(result_file, 'rb') as f:
        nq, k = struct.unpack('ii', f.read(8))
        results = np.frombuffer(f.read(nq * k * 4), dtype=np.uint32).reshape(nq, k)
    
    correct = 0
    total = 0
    for i in range(min(nq, len(gt))):
        gt_set = set(gt[i][:K])
        pred_set = set(results[i])
        correct += len(gt_set & pred_set)
        total += K
    return correct / total

print("=== BF16 Recall Check on GIST1M ===")
print("Comparing Recall@10: FP32 baseline vs BF16-quantized queries\n")

GIST_DIR = os.path.expanduser('~/data/gist')
INDEX_PREFIX = f'{GIST_DIR}/gist_index'
QUERY_FILE = f'{GIST_DIR}/gist_query.fvecs'
GT_FILE = f'{GIST_DIR}/gist_groundtruth.ivecs'
BF16_QUERY_FILE = f'{GIST_DIR}/gist_query_bf16sim.fvecs'

# Load data
print("Loading GIST1M queries and ground truth...")
queries_f32 = read_fvecs(QUERY_FILE)
gt = read_ivecs(GT_FILE)
print(f"  Queries: {queries_f32.shape} (float32)")
print(f"  Ground truth: {gt.shape}\n")

# Simulate BF16 quantization
queries_bf16sim = f32_to_bf16_to_f32(queries_f32)
max_diff = np.abs(queries_f32 - queries_bf16sim).max()
mean_diff = np.abs(queries_f32 - queries_bf16sim).mean()
print(f"BF16 quantization error on queries:")
print(f"  Max absolute diff: {max_diff:.6f}")
print(f"  Mean absolute diff: {mean_diff:.8f}\n")

# Save BF16-simulated queries
to_fvecs(queries_bf16sim, BF16_QUERY_FILE)
print(f"Saved BF16-simulated queries to {BF16_QUERY_FILE}\n")

# Run FP32 baseline search
print("Running DiskANN search with FP32 queries (baseline)...")
run_diskann_search(INDEX_PREFIX, QUERY_FILE, GT_FILE, '/tmp/gist_fp32_results')

# Run BF16-simulated search
print("\nRunning DiskANN search with BF16-simulated queries...")
run_diskann_search(INDEX_PREFIX, BF16_QUERY_FILE, GT_FILE, '/tmp/gist_bf16_results')

# Compute recall from result files
recall_fp32 = compute_recall('/tmp/gist_fp32_results', gt, K=10)
recall_bf16 = compute_recall('/tmp/gist_bf16_results', gt, K=10)

print("\n=== Results ===")
if recall_fp32 and recall_bf16:
    print(f"  Recall@10 FP32:          {recall_fp32*100:.2f}%")
    print(f"  Recall@10 BF16-simulated: {recall_bf16*100:.2f}%")
    print(f"  Recall drop:             {(recall_fp32 - recall_bf16)*100:.3f}%")
    if abs(recall_fp32 - recall_bf16) < 0.005:
        print("\n  PASS: BF16 quantization has negligible impact on recall.")
        print("  Safe to use BF16 in the batch controller for GIST1M.")
    else:
        print(f"\n  WARN: Recall drops by {(recall_fp32-recall_bf16)*100:.2f}% under BF16.")
        print("  Consider measuring impact at different L values.")
else:
    print("  Could not read result files directly.")
    print("  Check DiskANN output above for recall numbers.")

# Clean up
os.remove(BF16_QUERY_FILE)
print("\nDone.")
