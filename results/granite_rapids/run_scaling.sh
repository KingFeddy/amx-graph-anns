#!/bin/bash
# Reproduces the Vamana threading scaling curve on Granite Rapids
# Run from this directory after building DiskANN and downloading SIFT1M

DISKANN=~/DiskANN/build/apps/search_memory_index
INDEX=~/data/sift/sift_index
QUERY=~/data/sift/sift_query.bin
GT=~/data/sift/sift_groundtruth.bin
RESULTS=~/data/sift/results

for T in 1 2 4 8 16 32 64 128 256; do
    echo -n "T=$T: "
    $DISKANN \
        --data_type float \
        --dist_fn l2 \
        --index_path_prefix $INDEX \
        --query_file $QUERY \
        --gt_file $GT \
        --result_path $RESULTS \
        -K 10 -L 100 -T $T 2>/dev/null | grep "100 "
done
