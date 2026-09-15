#!/bin/bash
# After the val0 run finishes (its watcher stops the training), free card10/15
# (stop the dp2 server) and then run the buffer-depletion experiment.
set -u
D=/home/y50063564/dataloader_diag
O="$D/logs/after_val0.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$O"; }
say "waiting for val0 to finish ..."
for i in $(seq 1 240); do
  grep -q "\[val0\] DONE" "$D/logs/val0.out" 2>/dev/null && { say "val0 done"; break; }
  sleep 30
done
grep -q "\[val0\] DONE" "$D/logs/val0.out" 2>/dev/null || { say "timeout waiting val0"; exit 1; }

say "stopping dp2 server (frees cards 10 & 15)"
[ -f /home/y50063564/processed_data/dspark_data/dspark_accum/logs/vllm_dp2.pid ] && \
  kill "$(cat /home/y50063564/processed_data/dspark_data/dspark_accum/logs/vllm_dp2.pid)" 2>/dev/null
sleep 8
pkill -f "vllm serve /home/y50063564/Qwen3-4B" 2>/dev/null
sleep 4
say "vllm left: $(pgrep -af '[v]llm serve' | wc -l); trainers left: $(pgrep -f '[s]peculators/scripts/train\.py' | wc -l)"

say "starting buffer-depletion experiment (run_tl.sh)"
bash "$D/run_tl.sh" >> "$O" 2>&1
say "analysing buffer dynamics"
DL_RAW="$D/raw_tl_w2" DL_OUT="$D/analysis_buffer" /usr/local/python3.12.13/bin/python3 \
    "$D/analyze_buffer.py" > "$D/logs/analysis_buffer.txt" 2>&1
say "buffer analysis -> logs/analysis_buffer.txt"
say "AFTER_VAL0 DONE"
