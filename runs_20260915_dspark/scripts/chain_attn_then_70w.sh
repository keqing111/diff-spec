#!/bin/bash
# Chain: wait for the three draft-attention runs (MHA / GQA / MLA) to exit,
# report how each finished, then resume the paused 700k ctx-only run.
#
# The 700k run needs cards 11-14 (dp4) and the attention runs occupy 11-13, so
# it must not start until all three are gone. The vLLM server on :1123 is left
# running and is reused as-is: the client learns the hidden-states path from
# kv_transfer_params in the response, so the server's scratch dir does not have
# to match whatever the 700k run used before.
#
# Zombie-safe: polls /proc state (a finished torchrun stays as <defunct>, and
# `kill -0` succeeds on zombies).
#
# Usage: nohup bash chain_attn_then_70w.sh > chain.log 2>&1 &
set -uo pipefail

ATTN_ROOT=/home/y50063564/processed_data/dspark_data/dspark_attn
RESUME=/home/y50063564/dspark_project/script/dspark_accum_exp/train_dp4_accum3_ctx70w.sh
WATCH=/home/y50063564/dspark_project/script/dspark_accum_exp/dp4_ctx70w_watch.sh
DEADLINE_HOURS=14

log() { echo "[$(date -Is)] chain: $*"; }

deadline=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))

log "waiting for the three attention runs"
pids=()
for ARM in gqa mha mla; do
    p=$(cat "$ATTN_ROOT/$ARM/logs/train.pid" 2>/dev/null || echo "")
    log "  $ARM pid=${p:-<none>}"
    [ -n "$p" ] && pids+=("$p")
done

for p in "${pids[@]}"; do
    while :; do
        st=$(ps -o stat= -p "$p" 2>/dev/null) || break
        case "$st" in *Z*) break ;; esac
        if [ "$(date +%s)" -gt "$deadline" ]; then
            log "TIMEOUT after ${DEADLINE_HOURS}h waiting on pid $p; resuming 700k anyway"
            break 2
        fi
        sleep 60
    done
done

log "all attention runs exited; per-arm status:"
for ARM in gqa mha mla; do
    L=$(ls -t "$ATTN_ROOT/$ARM/logs"/train_attn_${ARM}_a12_*.log 2>/dev/null | head -1)
    if [ -z "$L" ]; then
        log "  $ARM: NO LOG"
        continue
    fi
    log "  $ARM: epoch3_done=$(grep -c 'Training epoch 3/3 completed' "$L") \
traceback=$(grep -cE 'Traceback|ChildFailedError' "$L") \
last_step=$(grep -oE 'global_step=[0-9]+' "$L" | tail -1)"
done

log "resuming 700k ctx-only run (will re-enter epoch 1, global_step 28910)"
nohup bash "$RESUME" > "$ATTN_ROOT/logs/resume_70w.log" 2>&1 &
log "700k launcher pid $!"
sleep 60
nohup bash "$WATCH" > "$ATTN_ROOT/logs/watch_70w.log" 2>&1 &
log "700k watcher pid $! (log: $ATTN_ROOT/logs/watch_70w.log)"
log "CHAIN_DONE"
