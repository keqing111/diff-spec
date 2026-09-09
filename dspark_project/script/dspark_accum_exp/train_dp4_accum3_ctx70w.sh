#!/bin/bash
# Real run (dp12-approx): pure-GQA ctx-only @layer0, FULL 700k, dp=4 + accum=3
# (each optimizer step averages 4 ranks x 3 micro-batches = 12 packs ~ dp12),
# 3 epochs, on cards 11-14; hidden states from the single card-15 vLLM server.
# Config matches the dp12 GQA baseline except ctx-only added on draft layer 0.
set -e

export ASCEND_RT_VISIBLE_DEVICES="11,12,13,14"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178
export NO_PROXY="$no_proxy"

PYTHON=/usr/local/python3.12.13/bin/python3
TORCHRUN=/usr/local/python3.12.13/bin/torchrun

RUN_NAME=dp4_accum3_ctx70w
RUN_ROOT=/home/y50063564/processed_data/dspark_data/dspark_accum/$RUN_NAME
LOG_DIR=$RUN_ROOT/logs
CKPT_DIR=$RUN_ROOT/checkpoints
mkdir -p "$LOG_DIR" "$CKPT_DIR"
LOG_FILE="$LOG_DIR/train_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train.pid"

nohup "$TORCHRUN" --standalone --nproc_per_node 4 \
    /home/y50063564/dspark_project/speculators/scripts/train.py \
    --verifier-name-or-path /home/y50063564/Qwen3-4B \
    --data-path /home/y50063564/data/open_perfectblend_qwen3_4b_700k \
    --vllm-endpoint http://80.48.17.178:1123/v1 \
    --save-path "$CKPT_DIR" \
    --draft-vocab-size 32000 \
    --epochs 3 \
    --lr 6e-4 \
    --grad-accum 3 \
    --checkpoint-freq 1 \
    --log-freq 1 \
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
    --num-workers 2 \
    > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "dp4-accum3-ctx70w log : $LOG_FILE"
echo "dp4-accum3-ctx70w pid : $(cat "$PID_FILE")"
