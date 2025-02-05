#!/bin/bash
#SBATCH -p g80
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --constraint="ampere"

export HF_HOME=/export/work/$USER/.cache/huggingface

nvidia-smi

python -m hf_bench.benchmark --experiment_config $1
