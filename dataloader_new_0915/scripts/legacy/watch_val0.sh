#!/bin/bash
# Wait for the (re-run) epoch0 validation to finish, then gracefully stop the
# training so it does NOT proceed into epoch1, and print the val0 metrics.
R=/home/y50063564/processed_data/dspark_data/dspark_accum/dp4_accum3_ctx70w
LOG=$(ls -t $R/logs/train_dp4_accum3_ctx70w_2026091[02]*.log | head -1)
echo "[val0] watching $LOG"
for i in $(seq 1 480); do
  grep -qa "Validation epoch 1/3 completed" "$LOG" && { echo "[val0] val0 completed after ~$((i/2)) min"; break; }
  sleep 30
done
sleep 10   # let it write val_metrics.json / checkpoint
echo "[val0] stopping training (SIGTERM, graceful)"
pgrep -f '[s]peculators/scripts/train\.py' | xargs -r kill -TERM
for i in $(seq 1 40); do [ "$(pgrep -f '[s]peculators/scripts/train\.py' | wc -l)" -eq 0 ] && break; sleep 5; done
pgrep -f '[s]peculators/scripts/train\.py' | xargs -r kill -9 2>/dev/null
echo "[val0] ---- val0 metrics (log) ----"
grep -a -A 22 "Validation epoch 1/3 completed" "$LOG" | grep -aE "val/|epoch=" | head -20
echo "[val0] ---- val_metrics.json ----"
cat $R/checkpoints/0/val_metrics.json 2>/dev/null; echo
echo "[val0] ---- checkpoint state ----"
cat $R/checkpoints/0/training_state.json 2>/dev/null; echo
echo "[val0] DONE"
