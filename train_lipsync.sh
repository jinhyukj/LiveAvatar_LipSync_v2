#!/bin/bash
# LiveAvatar LipSync Training Launch Script
# Usage: bash train_lipsync.sh [CONFIG_PATH] [ACCELERATE_CONFIG]
#
# Training modes (set training_mode in config YAML):
#   training_mode: "v2v"  — V2V inpainting (49ch: noise+mask+masked+ref)
#   training_mode: "i2v"  — I2V generation (16ch noise only, ref via sink conditioning)
#
# Examples:
#   bash train_lipsync.sh                                          # Single GPU, default config
#   bash train_lipsync.sh configs/lipsync_train.yaml               # Single GPU, custom config
#   bash train_lipsync.sh configs/lipsync_train.yaml configs/accelerate_config_multi_gpu.yaml  # Multi GPU

set -e

CONFIG=${1:-"configs/lipsync_train.yaml"}
ACCEL_CONFIG=${2:-"configs/accelerate_config.yaml"}

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate hb_liveavatar

echo "=== LiveAvatar LipSync Training ==="
echo "Config: ${CONFIG}"
echo "Accelerate config: ${ACCEL_CONFIG}"
echo "===================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${SCRIPT_DIR}/LiveAvatar:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export ENABLE_COMPILE=false  # Disable torch.compile for training (saves ~40-50GB GPU memory)

accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    train_lipsync.py \
    --config "$CONFIG"
