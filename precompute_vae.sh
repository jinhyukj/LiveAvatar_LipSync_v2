#!/bin/bash
# Precompute VAE latents for all training videos.
# Auto-detects all visible GPUs and distributes work via mp.spawn.
#
# Usage:
#   bash precompute_vae.sh                                           # All GPUs, batch_size=4
#   bash precompute_vae.sh configs/lipsync_train.yaml 4              # batch_size=4 (explicit)
#   CUDA_VISIBLE_DEVICES=0,1 bash precompute_vae.sh configs/lipsync_train.yaml 2

set -e

CONFIG=${1:-"configs/lipsync_train.yaml"}
BATCH_SIZE=${2:-4}
OUTPUT_DIR=${3:-"/home/work/liveavatar_data"}

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate hb_liveavatar

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${SCRIPT_DIR}/LiveAvatar:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== Precompute VAE Latents ==="
echo "Config: ${CONFIG}"
echo "Batch size: ${BATCH_SIZE}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================="

python precompute_vae_latents.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE"
