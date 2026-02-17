#!/bin/bash
# Precompute T5 text embeddings for all unique prompts.
# Single GPU only (fast: ~8750 forward passes takes minutes).
# Usage:
#   bash precompute_text.sh
#   bash precompute_text.sh configs/lipsync_train.yaml /home/work/liveavatar_data

set -e

CONFIG=${1:-"configs/lipsync_train.yaml"}
OUTPUT_DIR=${2:-"/home/work/liveavatar_data"}

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate hb_liveavatar

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${SCRIPT_DIR}/LiveAvatar:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== Precompute Text Embeddings ==="
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "==================================="

python precompute_text_embeddings.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR"
