#!/bin/bash
# Orchestrates the dataloader/rank diagnostic end-to-end:
#   1. wait until the 70w run finishes epoch0 (checkpoint 28910 saved)
#   2. gracefully stop 70w + its dp2 server
#   3. run the diagnostic ALONE (dp1 server card4:1130 + dp4 trainer cards0-3, W2)
#   4. stop the diagnostic
#   5. resume 70w from its checkpoint (dp2 server back, same script)
# Everything is appended to /home/y50063564/dataloader_diag/logs/orchestrator.log
set -u
D=/home/y50063564/dataloader_diag
R70=/home/y50063564/processed_data/dspark_data/dspark_accum/dp4_accum3_ctx70w
S=/home/y50063564/dspark_project/script/dspark_accum_exp
mkdir -p "$D/logs"
O="$D/logs/orchestrator.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$O"; }

# stop the old stall-watcher for the 70w run (it would fire on our intentional stop)
for p in $(pgrep -f dp4_ctx70w_watch.sh); do kill "$p" 2>/dev/null; done

say "waiting for 70w epoch0 completion ..."
LOG70=$(ls -t $R70/logs/train_dp4_accum3_ctx70w_20260910_0825*.log | head -1)
for i in $(seq 1 480); do
  grep -qa "Validation epoch 1/3 started" "$LOG70" && { say "epoch0 done (checkpoint saved)"; break; }
  sleep 30
done
grep -qa "Validation epoch 1/3 started" "$LOG70" || { say "TIMEOUT waiting epoch0; aborting"; exit 1; }

say "stopping 70w training"
bash "$S/graceful_stop_70w.sh" >> "$O" 2>&1
say "stopping dp2 server"
[ -f /home/y50063564/processed_data/dspark_data/dspark_accum/logs/vllm_dp2.pid ] && kill "$(cat /home/y50063564/processed_data/dspark_data/dspark_accum/logs/vllm_dp2.pid)" 2>/dev/null
sleep 8
pkill -f "vllm serve /home/y50063564/Qwen3-4B" 2>/dev/null; sleep 4
say "trainers left: $(pgrep -f '[s]peculators/scripts/train\.py' | wc -l), vllm left: $(pgrep -af '[v]llm serve' | wc -l)"

say "starting DIAG server (card4:1130)"
bash "$D/serve_diag_dp1.sh" >> "$O" 2>&1
for i in $(seq 1 90); do curl -sf --noproxy '*' -m 3 http://80.48.17.178:1130/v1/models >/dev/null 2>&1 && { say "diag server ready"; break; }; sleep 10; done

say "starting DIAG trainer (dp4 cards0-3, W2, DL_PROFILE=1)"
rm -rf "$D/raw"; mkdir -p "$D/raw"
bash "$D/train_diag_dp4_w2.sh" >> "$O" 2>&1
DPID=$(cat "$D/logs/train_diag.pid")
say "diag trainer pid=$DPID; waiting for it to finish (1 epoch) ..."
for i in $(seq 1 360); do   # up to 3h
  st=$(ps -o stat= -p "$DPID" 2>/dev/null)
  [ -z "$st" ] && { say "diag trainer exited"; break; }
  case "$st" in *Z*) say "diag trainer zombie (done)"; break;; esac
  sleep 30
done
pgrep -f '[s]peculators/scripts/train\.py' | xargs -r kill -TERM; sleep 5
pgrep -f '[s]peculators/scripts/train\.py' | xargs -r kill -9 2>/dev/null
say "diag profiling files: $(find "$D/raw" -name '*.jsonl' | wc -l)"
say "stopping diag server"
[ -f "$D/logs/vllm_diag.pid" ] && kill "$(cat "$D/logs/vllm_diag.pid")" 2>/dev/null
sleep 6; pkill -f "vllm serve /home/y50063564/Qwen3-4B" 2>/dev/null; sleep 4

say "resuming 70w: restarting dp2 server"
bash "$S/serve_vllm_dp2.sh" >> "$O" 2>&1
for i in $(seq 1 90); do curl -sf --noproxy '*' -m 3 http://80.48.17.178:1123/v1/models >/dev/null 2>&1 && { say "dp2 server ready"; break; }; sleep 10; done
say "resuming 70w training"
bash "$S/train_dp4_accum3_ctx70w.sh" >> "$O" 2>&1
sleep 120
NEWLOG=$(ls -t $R70/logs/train_dp4_accum3_ctx70w_*.log | head -1)
say "70w resumed from: $(grep -a 'Fast-skipping' "$NEWLOG" | tail -1 | tr -s ' ')"
say "DIAG SEQUENCE DONE"
