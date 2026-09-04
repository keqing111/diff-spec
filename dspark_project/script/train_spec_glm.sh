#!/bin/bash
# Online DSpark Training Script for Qwen3-8B on Ascend NPU
#
# Runs the full online DSpark training pipeline on Ascend: data preparation,
# vLLM server launch, and training with hidden states generated on-the-fly.
# DSpark extends DFlash with a Markov head and a confidence head.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/dspark_qwen3_8b_sharegpt_online_ascend.sh
#
# Note: This assumes your environment has torch_npu and an Ascend-compatible
# vLLM installation that supports hidden-state extraction.

set -euo pipefail
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=1
export NO_PROXY=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188 
export no_proxy=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188
# # HCCL port range to avoid conflicts when using torchrun multi-process
# # Each rank binds a different port in this range
# export HCCL_NPU_SOCKET_PORT_RANGE="16667"
# ============ Configuration ============
MODEL="/home/y50063564/dspark_project/glm_config"
DATASET="sharegpt"                # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="/home/y50063564/dspark_project/outputs"

VLLM_PORT=1123
SEQ_LENGTH=4096
EPOCHS=3
LR=6e-4
LOGGER="tensorboard"

# DSpark-specific parameters
SPECULATOR_TYPE="dspark"
DRAFT_ATTN_IMPL="sdpa"   # simple_flex_attention | sdpa | eager (use sdpa/eager on Ascend NPU)
BLOCK_SIZE=8
MAX_ANCHORS=256
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=32000
# Draft fc aux layers (must match vLLM planes [:, :-1] after data.py split)
TARGET_LAYER_IDS="2 33"

# Markov + confidence head settings
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"   # vanilla | gated | rnn
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# Ascend NPU assignments (online training needs separate devices for vLLM/training)
TRAIN_NPUS="0,1"
NUM_TRAIN_NPUS=2


# Step 3: Train DSpark against the live vLLM server
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train.pid"

echo "=== Step 3: Training on Ascend NPU(s): $TRAIN_NPUS ==="
nohup env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    /home/y50063564/dspark_project/speculators/scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "/home/y50063564/dspark_project/input/processed_data" \
    --vllm-endpoint "http://80.48.17.178:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --draft-attn-impl "$DRAFT_ATTN_IMPL" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --on-missing generate \
    --on-generate delete \
    --request-timeout 900 \
    --max-retries 5 \
    --num-workers 4 \
    > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

    # --logger "$LOGGER" \
    # --hidden-states-path "/home/z00841464/dspark_project/outputs/hidden_states" \

echo "Log file: $LOG_FILE"
# echo "TensorBoard: tensorboard --logdir ./logs --host 0.0.0.0 --port 6006"
echo "View log with: tail -f $LOG_FILE"
echo "Stop with: kill \$(cat $PID_FILE)"