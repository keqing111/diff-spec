#!/bin/bash
# 16 卡满配训练(另一台服务器 80.48.17.185,无共用约束):
#   单次训练 dp=12(卡 0-11),连 1123 端口的 4 副本 vLLM(卡 12-15)。
#   全新 save_path(dspark_16card),不续任何旧 checkpoint。
# 前提: docker 环境与路径和 80.48.17.178 一致;模型/数据目录存在;先起 vllm_serve_16card.sh。
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
OUTPUT_DIR="/home/y50063564/processed_data/dspark_data/dspark_16card"

# 注意:另一台服务器 IP 是 80.48.17.185
VLLM_IP=80.48.17.185
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
# 最后一层(索引4,共5层)用 Diff-Transformer(diffv1) 注意力;置空则全部用 GQA
DIFF_ATTENTION_LAYER_INDICES="4"

# Markov + confidence head settings
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# Ascend NPU assignments (16 卡满配: 12 trainer 用卡 0-11,vLLM 4 副本在卡 12-15)
TRAIN_NPUS="0,1,2,3,4,5,6,7,8,9,10,11"
NUM_TRAIN_NPUS=12

# Step 3: Train DSpark against the live vLLM server
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train.pid"

echo "=== Step 3 (16-card): Training on Ascend NPU(s): $TRAIN_NPUS ==="
nohup env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    /home/y50063564/dspark_project/speculators/scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "/home/y50063564/data/open_perfectblend_qwen3_4b_700k" \
    --vllm-endpoint "http://${VLLM_IP}:${VLLM_PORT}/v1" \
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
    --diff-attention-layer-indices $DIFF_ATTENTION_LAYER_INDICES \
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
