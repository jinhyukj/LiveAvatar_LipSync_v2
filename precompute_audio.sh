#!/bin/bash
# Precompute audio embeddings for all training videos.
# Usage:
#   bash precompute_audio.sh                                           # Single GPU
#   bash precompute_audio.sh configs/lipsync_train.yaml 4              # 4 GPUs

set -e

CONFIG=${1:-"configs/lipsync_train.yaml"}
NUM_GPUS=${2:-1}
OUTPUT_DIR=${3:-"/home/work/liveavatar_data"}

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate hb_liveavatar

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${SCRIPT_DIR}/LiveAvatar:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== Precompute Audio Embeddings ==="
echo "Config: ${CONFIG}"
echo "GPUs: ${NUM_GPUS}"
echo "Output: ${OUTPUT_DIR}"
echo "===================================="

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$NUM_GPUS" \
        precompute_audio_embeddings.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR"
else
    python precompute_audio_embeddings.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR"
fi
