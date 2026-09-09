#!/bin/bash
# Waits for both accum1/accum12 trainer processes to exit, then runs the
# comparison/plot script and prints a short status. Designed to run in the
# background while the (multi-hour) training runs.
ROOT=/home/y50063564/processed_data/dspark_data/dspark_accum
P1=$(cat "$ROOT/accum1/logs/train.pid" 2>/dev/null)
P2=$(cat "$ROOT/accum12/logs/train.pid" 2>/dev/null)

echo "[watcher] waiting for accum1 pid=$P1 and accum12 pid=$P2 ..."
while kill -0 "$P1" 2>/dev/null || kill -0 "$P2" 2>/dev/null; do
    sleep 60
done
echo "[watcher] both trainers have exited."
sleep 15

L1=$(ls -t "$ROOT/accum1/logs"/train_accum1_*.log | head -1)
L2=$(ls -t "$ROOT/accum12/logs"/train_accum12_*.log | head -1)
echo "[watcher] accum1 log: $L1"
echo "[watcher] accum12 log: $L2"

echo "=== accum1 tail ==="; tail -n 12 "$L1"
echo "=== accum12 tail ==="; tail -n 12 "$L2"

cd "$(dirname "$0")" || exit 1
/usr/local/python3.12.13/bin/python3 compare_accum.py \
    --accum1 "$L1" --accum12 "$L2" \
    -o "$ROOT/analysis" 2>&1 | tail -n 40
echo "[watcher] done. figures under $ROOT/analysis/"
