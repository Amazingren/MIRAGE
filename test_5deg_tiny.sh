#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

# micromamba environment
source ~/.bashrc
conda activate /share_5/users/bin_ren/envs/anyir

# Project directory
cd /share_5/users/bin_ren/project/ll/MIRAGE

### ===== [2] Configurations =====
DENOISE_DIR="/share_5/users/bin_ren/datasets/ll/test/denoise/"
DERAIN_DIR="/share_5/users/bin_ren/datasets/ll/test/derain/"
DEHAZE_DIR="/share_5/users/bin_ren/datasets/ll/test/dehaze/"
GOPRO_DIR="/share_5/users/bin_ren/datasets/ll/test/deblur/"
ENHANCE_DIR="/share_5/users/bin_ren/datasets/ll/test/enhance/"

LOG_DIR="./outputs/5deg_tiny_ep100"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test_$(date +%Y%m%d_%H%M%S).log"

# Save everything to a log file while keeping console output.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"

### ===== [3] Launch Training =====
CUDA_VISIBLE_DEVICES=0 python test_tiny.py \
    --trainset AnyIR \
    --mode 6 \
    --avg_denoise_sigma 25 \
    --denoise_path "$DENOISE_DIR" \
    --derain_path "$DERAIN_DIR" \
    --dehaze_path "$DEHAZE_DIR" \
    --gopro_path "$GOPRO_DIR" \
    --enhance_path "$ENHANCE_DIR" \
    --ckpt_name 5deg_tiny/epoch=100.ckpt \
    --output_path ./outputs/5deg_tiny_ep100