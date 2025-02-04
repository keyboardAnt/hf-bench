#! /bin/bash

timestamp=$(date +\%Y\%m\%d_\%H\%M\%S) && bsub \
    -N \
    -q "long-gpu short-gpu risk-gpu" \
    -gpu "num=1:j_exclusive=yes:gmem=80GB" \
    -R "rusage[mem=20GB]" \
    -R "affinity[core(8)]" \
    -R "span[hosts=1]" \
    -n 1 \
    -M 20GB \
    -o /home/projects/dharel/nadavt/repos/hf-bench/logs/cluster/${timestamp}_jobid_%J_benchmark.log \
    "module load miniconda/24.11_environmentally && module load CUDA/12.4.0 && conda activate hf-bench-env && python -m hf_bench.benchmark  ${@}"