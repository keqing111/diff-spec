#!/bin/bash
# 8:2(=4:1) 生产配置训练:1 个 torchrun,dp=8(卡 6-13),连 1123 端口的 2 副本 vLLM。
# 全新 save_path(dspark_8dp),不续 dp=2 旧 checkpoint(mid-epoch 换 DP 会对不齐数据)。
# 用法: 先起 vllm_serve_8to2.sh,再起本脚本。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
set -euo pipefail
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=1
export NO_PROXY=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178
export no_proxy=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178

# ============ Configuration ============
MODEL="/home/y50063564/Qwen3-4B"
DATASET="sharegpt"
OUTPUT_DIR="/home/y50063564/processed_data/dspark_data/dspark_8dp"

VLLM_PORT=1123
SEQ_LENGTH=4096
EPOCHS=3
LR=6e-4
LOGGER="tensorboard"

# DSpark-specific parameters
SPECULATOR_TYPE="dspark"
DRAFT_ATTN_IMPL="sdpa"
BLOCK_SIZE=8
MAX_ANCHORS=256
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=32000
# Draft fc aux layers (must match vLLM planes [:, :-1] after data.py split)
TARGET_LAYER_IDS="1 9 17 25 33"

# Markov + confidence head settings
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# Ascend NPU assignments (8:2 生产配置: 8 trainer rank, 2 副本 vLLM on cards 14,15)
TRAIN_NPUS="6,7,8,9,10,11,12,13"
NUM_TRAIN_NPUS=8

# Step 3: Train DSpark against the live vLLM server
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train.pid"

echo "=== Step 3 (8:2 = 4:1): Training on Ascend NPU(s): $TRAIN_NPUS ==="
nohup env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    /home/y50063564/dspark_project/speculators/scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "/home/y50063564/data/open_perfectblend_qwen3_4b_700k" \
    --vllm-endpoint "http://80.48.17.178:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --checkpoint-freq 0.1 \
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

echo "Log file: $LOG_FILE"
echo "View log with: tail -f $LOG_FILE"
