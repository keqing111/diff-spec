#!/bin/bash
# Smoke test for the draft-attention comparison. Two stages per arm, both on
# card 11 (sequential, so the stages never overlap):
#
#   stage 1  --dry-run    build the real draft model through build_draft_model,
#                         save a checkpoint, exit. Verifies that the CLI flags
#                         reach core.py's class selection and that the draft
#                         config round-trips into config.json. Needs no server.
#   stage 2  DL_MAX_STEPS=30  30 real micro-batches against the live vLLM
#                         server. Verifies forward/backward/optimizer and that
#                         train/accept_len + loss are produced.
#
# Usage: bash smoke_attn.sh          (stage 1 only)
#        bash smoke_attn.sh --with-server   (stages 1 and 2)
set -uo pipefail

CARD=11
ROOT=/home/y50063564/processed_data/dspark_data/dspark_attn/_smoke
SPECDIR=/home/y50063564/dspark_project/speculators
PYTHON=/usr/local/python3.12.13/bin/python3
RUNNER=/home/y50063564/dspark_project/script/dspark_accum_exp/train_attn_run.sh
CKPTCHECK=/home/y50063564/dspark_project/script/dspark_accum_exp/check_ckpt_attn.py

WITH_SERVER=0
[ "${1:-}" = "--with-server" ] && WITH_SERVER=1

mkdir -p "$ROOT"

for ARM in gqa mha mla; do
    case "$ARM" in
        mha|gqa) ATTN_FLAGS=(--draft-attention-type "$ARM") ;;
        mla)     ATTN_FLAGS=(--draft-attention-type mla --mla-kv-lora-rank 512) ;;
    esac
    DEST="$ROOT/$ARM"
    rm -rf "$DEST"; mkdir -p "$DEST"

    echo "==================================================================="
    echo "stage 1: dry-run   arm=$ARM"
    echo "==================================================================="
    ASCEND_RT_VISIBLE_DEVICES=$CARD \
    PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
    TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 \
    "$PYTHON" "$SPECDIR/scripts/train.py" \
        --verifier-name-or-path /home/y50063564/Qwen3-4B \
        --data-path /home/y50063564/data/open_perfectblend_qwen3_4b_50k \
        --save-path "$DEST" \
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
        --full-attention-indices 0 1 2 3 4 \
        "${ATTN_FLAGS[@]}" \
        --markov-rank 256 \
        --markov-head-type vanilla \
        --enable-confidence-head \
        --confidence-head-with-markov \
        --loss-fn '{"ce": 0.1, "tv": 0.9}' \
        --confidence-head-alpha 1.0 \
        --dry-run 2>&1 | tail -25
    echo "dry-run exit: ${PIPESTATUS[0]}"
done

echo
echo "==================================================================="
echo "stage 1 verification: what config.json / weights actually contain"
echo "==================================================================="
CKPT_DIRS=()
for ARM in gqa mha mla; do
    d=$(find "$ROOT/$ARM" -maxdepth 3 -name config.json -printf '%h\n' 2>/dev/null | sort | tail -1)
    [ -n "$d" ] && CKPT_DIRS+=("$d")
done
if [ ${#CKPT_DIRS[@]} -gt 0 ]; then
    "$PYTHON" "$CKPTCHECK" "${CKPT_DIRS[@]}" || true
else
    echo "!! no checkpoint produced by any dry-run -- inspect the logs above"
fi

if [ "$WITH_SERVER" -eq 0 ]; then
    echo
    echo "stage 1 done. Re-run with --with-server once the vLLM server is up."
    exit 0
fi

echo
echo "==================================================================="
echo "stage 2: 30 real micro-batches per arm (needs the server on :1123)"
echo "==================================================================="
for ARM in gqa mha mla; do
    DEST="$ROOT/$ARM/train"
    rm -rf "$DEST"
    echo "--- $ARM ---"
    ATTN_RUN_ROOT="$DEST" DL_MAX_STEPS=30 bash "$RUNNER" "$ARM" "$CARD"
    LOG=$(ls -t "$DEST"/logs/train_*.log 2>/dev/null | head -1)
    echo "log: $LOG"
    sleep 2
    # torchrun leaves a zombie behind, and `kill -0` succeeds on zombies, so
    # poll /proc state (skip Z = defunct) instead. Hard cap at 30 min.
    PID=$(cat "$DEST"/logs/train.pid 2>/dev/null || echo "")
    deadline=$(( $(date +%s) + 1800 ))
    while [ -n "$PID" ]; do
        st=$(ps -o stat= -p "$PID" 2>/dev/null) || break
        case "$st" in *Z*) break ;; esac
        if [ "$(date +%s)" -gt "$deadline" ]; then
            echo "!! $ARM still alive after 30 min; moving on"
            break
        fi
        sleep 5
    done
    echo "--- $ARM tail ---"
    grep -E "global_step=|accept|loss" "$LOG" 2>/dev/null | tail -6
done
echo "SMOKE_DONE"
