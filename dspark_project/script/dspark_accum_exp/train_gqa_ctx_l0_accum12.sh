#!/bin/bash
# Ablation: PURE GQA accum12 + ctx-only restriction on DRAFT LAYER 0 only.
# No diff branch anywhere (--diff-attention-* absent). Everything else identical
# to train_accum12.sh so it isolates the effect of "ctx-only" alone vs
# diff-ctx-only @layer0.
# Trainer on card 11, hidden states from the card-15 vLLM server.
set -e

export ASCEND_RT_VISIBLE_DEVICES=11
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178
export NO_PROXY="$no_proxy"

PYTHON=/usr/local/python3.12.13/bin/python3
TORCHRUN=/usr/local/python3.12.13/bin/torchrun

RUN_NAME=gqa_ctx_l0_a12
RUN_ROOT=/home/y50063564/processed_data/dspark_data/dspark_accum/gqa_ctx_l0_a12
LOG_DIR=$RUN_ROOT/logs
CKPT_DIR=$RUN_ROOT/checkpoints
mkdir -p "$LOG_DIR" "$CKPT_DIR"
LOG_FILE="$LOG_DIR/train_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train.pid"

nohup "$TORCHRUN" --standalone --nproc_per_node 1 \
    /home/y50063564/dspark_project/speculators/scripts/train.py \
    --verifier-name-or-path /home/y50063564/Qwen3-4B \
    --data-path /home/y50063564/data/open_perfectblend_qwen3_4b_50k \
    --vllm-endpoint http://80.48.17.178:1123/v1 \
    --save-path "$CKPT_DIR" \
    --draft-vocab-size 32000 \
    --epochs 3 \
    --lr 6e-4 \
    --grad-accum 12 \
    --checkpoint-freq 1 \
    --log-freq 20 \
    --total-seq-len 4096 \
    --speculator-type dspark \
    --draft-attn-impl sdpa \
    --block-size 8 \
    --max-anchors 256 \
    --num-layers 5 \
    --target-layer-ids 1 9 17 25 33 \
    --gqa-context-only-layer-indices 0 \
    --markov-rank 256 \
    --markov-head-type vanilla \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' \
    --confidence-head-alpha 1.0 \
    --on-missing generate \
    --on-generate delete \
    --request-timeout 900 \
    --max-retries 5 \
    --num-workers 4 \
    > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "gqa-ctx-l0-accum12 trainer log : $LOG_FILE"
echo "gqa-ctx-l0-accum12 trainer pid : $(cat "$PID_FILE")"
