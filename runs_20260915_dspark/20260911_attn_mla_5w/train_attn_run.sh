#!/bin/bash
# Draft-attention comparison for DSpark (target = Qwen3-4B):
#   mha = 5-layer MHA        (num_key_value_heads 8 -> 32, Q heads unchanged)
#   gqa = Qwen3-4B native GQA (32 Q / 8 KV / head_dim 128)  <- the reference
#   mla = light MLA          (native GQA head layout + K/V routed through a
#                             shared rank-512 latent, --mla-kv-lora-rank 512)
#
# Single card, grad-accum 12, 3 epochs on data/open_perfectblend_qwen3_4b_50k.
# Trainer card is chosen per arm so all three can run concurrently against one
# server; nothing else depends on the card.
#
# Usage: bash train_attn_run.sh {mha|gqa|mla} <card>
#
# EXACT diff vs the reference script train_accum12.sh (3 flags, nothing else):
#   1. --full-attention-indices 0 1 2 3 4
#        baseline: dspark defaults every draft layer to sliding_attention with
#        window 2048. Qwen3-4B itself is full attention everywhere
#        (sliding_window=None), so "native" here means dropping the window.
#   2. --draft-attention-type <mha|gqa|mla>   (default is gqa)
#   3. --mla-kv-lora-rank 512                 (mla only)
# Output dir / run root differ, as do the log/pid paths. That is all.
set -euo pipefail

# NOTE: keep the :? messages free of braces -- bash ends the parameter
# expansion at the first unescaped "}", which would silently corrupt "$1".
ARM="${1:?usage: train_attn_run.sh ARM CARD, where ARM is mha, gqa or mla}"
CARD="${2:?usage: train_attn_run.sh ARM CARD, where CARD is the NPU id}"

case "$ARM" in
    mha|gqa) ATTN_FLAGS=(--draft-attention-type "$ARM") ;;
    mla)     ATTN_FLAGS=(--draft-attention-type mla --mla-kv-lora-rank 512) ;;
    *) echo "unknown arm: $ARM (expected mha|gqa|mla)" >&2; exit 2 ;;
esac

export ASCEND_RT_VISIBLE_DEVICES="$CARD"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178
export NO_PROXY="$no_proxy"

PYTHON=/usr/local/python3.12.13/bin/python3
TORCHRUN=/usr/local/python3.12.13/bin/torchrun

RUN_NAME="attn_${ARM}_a12"
# ATTN_RUN_ROOT lets the smoke test write somewhere disposable; unset (the
# normal case) means the real per-arm directory.
RUN_ROOT="${ATTN_RUN_ROOT:-/home/y50063564/processed_data/dspark_data/dspark_attn/$ARM}"
LOG_DIR="$RUN_ROOT/logs"
CKPT_DIR="$RUN_ROOT/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"
LOG_FILE="$LOG_DIR/train_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train.pid"

EXTRA_STEPS="${DL_MAX_STEPS:-0}"

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
    --log-freq 1 \
    --total-seq-len 4096 \
    --speculator-type dspark \
    --draft-attn-impl sdpa \
    --block-size 8 \
    --max-anchors 256 \
    --num-layers 5 \
    --target-layer-ids 1 9 17 25 33 \
    --full-attention-indices 0 1 2 3 4 \
    "${ATTN_FLAGS[@]}" \
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

{
    echo "arm            : $ARM"
    echo "attn flags     : ${ATTN_FLAGS[*]}"
    echo "card           : $CARD"
    echo "run root       : $RUN_ROOT"
    echo "data           : /home/y50063564/data/open_perfectblend_qwen3_4b_50k"
    echo "verifier       : /home/y50063564/Qwen3-4B"
    echo "started        : $(date -Is)"
    echo "log            : $LOG_FILE"
    echo "pid            : $(cat "$PID_FILE")"
    echo "DL_MAX_STEPS   : $EXTRA_STEPS"
    if git -C /home/y50063564/dspark_project/speculators rev-parse HEAD 2>/dev/null; then
        git -C /home/y50063564/dspark_project/speculators diff \
            > "$RUN_ROOT/diff.patch" 2>/dev/null || true
        echo "diff.patch     : $RUN_ROOT/diff.patch"
    else
        echo "git            : unavailable"
    fi
} | tee "$RUN_ROOT/config.txt"

echo "attn-$ARM trainer log : $LOG_FILE"
echo "attn-$ARM trainer pid : $(cat "$PID_FILE")"
