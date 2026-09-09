#!/bin/bash
# Zombie-safe watcher for the 700k dp4+accum3 ctx run.
#  - polls the torchrun PID state (zombie/absent = finished)
#  - flags a STALL if the log is unchanged for >30 min while the process is alive
#    (the epoch-end dp deadlock we could not fully de-risk in the smoke)
# Writes markers to stdout; run in background.
ROOT=/home/y50063564/processed_data/dspark_data/dspark_accum/dp4_accum3_ctx70w
P=$(cat "$ROOT/logs/train.pid" 2>/dev/null)
LOG=$(ls -t "$ROOT/logs"/train_dp4_accum3_ctx70w_*.log | head -1)
log(){ echo "[70w] $*"; }

alive(){
  [ -n "$P" ] || return 1
  st=$(ps -o stat= -p "$P" 2>/dev/null) || return 1
  case "$st" in *Z*) return 1;; esac
  return 0
}

log "watching pid=$P log=$LOG"
stall_since=""
while alive; do
  mt=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
  now=$(date +%s)
  age=$(( now - mt ))
  if [ "$age" -gt 1800 ]; then
    if [ -z "$stall_since" ]; then
      stall_since=$now
      log "STALL_DETECTED: log unchanged ${age}s; possible epoch-end dp deadlock"
    elif [ $(( now - stall_since )) -gt 2700 ]; then
      log "STALLED_FINAL after $(( now - stall_since ))s unchanged; exiting watcher"
      exit 3
    fi
  else
    stall_since=""
  fi
  sleep 300
done
log "process pid=$P finished."
sleep 15
log "val3/3=$(grep -c 'Validation epoch 3/3 completed' "$LOG") destroy=$(grep -c 'Destroyed distributed process group' "$LOG") errors=$(grep -cE 'ChildFailedError|Traceback' "$LOG")"
log "tail:"; tail -n 15 "$LOG"
log "done"
