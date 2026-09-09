#!/bin/bash
# Zombie-safe completion watcher for the single ablation run (gqa_ctx_l0_a12).
# A process is considered finished when it no longer exists OR is a zombie (Z).
# On finish it runs compare_ablation_l0.py and prints the per-epoch numbers.
ROOT=/home/y50063564/processed_data/dspark_data/dspark_accum/gqa_ctx_l0_a12
P=$(cat "$ROOT/logs/train.pid" 2>/dev/null)
log() { echo "[ablation] $*"; }
log "watching pid=$P"
alive() {
    [ -n "$P" ] || return 1
    stat=$(ps -o stat= -p "$P" 2>/dev/null) || return 1
    case "$stat" in *Z*) return 1 ;; esac   # zombie = finished
    return 0
}
while alive; do sleep 60; done
log "training process finished (pid $P)."
sleep 10
L=$(ls -t "$ROOT/logs"/train_gqa_ctx_l0_a12_*.log | head -1)
log "log: $L"
tail -n 8 "$L"
log "errors: $(grep -cE 'Traceback|Error' "$L")  val3/3: $(grep -c 'Validation epoch 3/3 completed' "$L")"
cd /home/y50063564/dspark_project/script/dspark_accum_exp || exit 1
/usr/local/python3.12.13/bin/python3 compare_ablation_l0.py
log "done."
