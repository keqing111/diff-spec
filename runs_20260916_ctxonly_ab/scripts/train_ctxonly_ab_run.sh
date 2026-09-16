#!/bin/bash
# A/B for the ctx-only anchor leak on the 50k crop (5w), 3 epochs, accum=12.
#
#   fix    : ctx-only @layer0, anchor_in_context=OFF (default).
#            Queries see base[0 .. anchor-1] only -- exactly what inference gives
#            (the draft's context K/V stops at the last position the target has
#            processed). Anchors with no same-document predecessor are dropped so
#            the visible set is never empty.
#   legacy : ctx-only @layer0, --ctx-only-anchor-in-context (old behaviour).
#            Queries additionally see base[anchor], i.e. the verifier's hidden
#            state at the anchor token -- a training-only signal. Expected to
#            reproduce the earlier inflated accept_len.
#
# Usage: bash train_ctxonly_ab_run.sh {fix|legacy} <card>
#
# Everything else is byte-identical to the pre-existing 5w ctx-only run
# (train_gqa_ctx_l0_accum12.sh): 50k data, 3 epochs, lr 6e-4, --grad-accum 12,
# --gqa-context-only-layer-indices 0, sliding window 2048 (ctx-only requires
# sliding layers), --num-workers 4, single card. Only the two A/B arms differ.
set -euo pipefail

ARM="${1:?usage: train_ctxonly_ab_run.sh ARM CARD, ARM is fix or legacy}"
CARD="${2:?usage: train_ctxonly_ab_run.sh ARM CARD, CARD is the NPU id}"

case "$ARM" in
    fix)    ATTN_FLAGS=() ;;
    legacy) ATTN_FLAGS=(--ctx-only-anchor-in-context) ;;
    *) echo "unknown arm: $ARM (expected fix|legacy)" >&2; exit 2 ;;
esac

export ASCEND_RT_VISIBLE_DEVICES="$CARD"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178
export NO_PROXY="$no_proxy"

PYTHON=/usr/local/python3.12.13/bin/python3
TORCHRUN=/usr/local/python3.12.13/bin/torchrun

RUN_NAME="ctxonly_${ARM}_a12"
RUN_ROOT="${CTX_RUN_ROOT:-/home/y50063564/processed_data/dspark_data/dspark_ctxonly_ab/$ARM}"
LOG_DIR="$RUN_ROOT/logs"
CKPT_DIR="$RUN_ROOT/checkpoints"
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
    --log-freq 1 \
    --total-seq-len 4096 \
    --speculator-type dspark \
    --draft-attn-impl sdpa \
    --block-size 8 \
    --max-anchors 256 \
    --num-layers 5 \
    --target-layer-ids 1 9 17 25 33 \
    --gqa-context-only-layer-indices 0 \
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
    echo "attn flags     : ${ATTN_FLAGS[*]:-<none>}"
    echo "card           : $CARD"
    echo "data           : /home/y50063564/data/open_perfectblend_qwen3_4b_50k"
    echo "run root       : $RUN_ROOT"
    echo "started        : $(date -Is)"
    echo "log            : $LOG_FILE"
    echo "pid            : $(cat "$PID_FILE")"
    echo "git HEAD       : $(git -C /home/y50063564/dspark_project/speculators rev-parse HEAD 2>/dev/null || echo unknown)"
} | tee "$RUN_ROOT/config.txt"

git -C /home/y50063564/dspark_project/speculators diff \
    > "$RUN_ROOT/diff.patch" 2>/dev/null || true

echo "ctxonly-$ARM log : $LOG_FILE"
echo "ctxonly-$ARM pid : $(cat "$PID_FILE")"
