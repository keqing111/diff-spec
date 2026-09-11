#!/bin/bash
# A/B dataloader diagnosis on cards 10-15 ONLY (per instruction):
#   server: single replica on card10, port 1131
#   A: dp4 cards 11-14, W2, in_order=True  (default)  -> raw_w2_inorder1
#   B: dp4 cards 11-14, W2, in_order=False            -> raw_w2_inorder0
# Both stop after DL_MAX_STEPS=1000 per rank; the analyzer runs after each.
set -u
D=/home/y50063564/dataloader_diag
O="$D/logs/ab.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$O"; }
mkdir -p "$D/logs"

say "starting single-replica server on card10:1131"
bash "$D/serve_diag_dp1_card10.sh" >> "$O" 2>&1
for i in $(seq 1 90); do curl -sf --noproxy '*' -m 3 http://80.48.17.178:1131/v1/models >/dev/null 2>&1 && { say "1131 ready"; break; }; sleep 10; done

run_case () {  # $1=name  $2=script  $3=rawdir
  local name="$1" script="$2" raw="$3"
  say "=== running case $name ==="
  rm -rf "$raw"; mkdir -p "$raw"
  bash "$script" >> "$O" 2>&1
  local pid; pid=$(cat "$D/logs/train_diag.pid")
  say "case $name pid=$pid; waiting (cap 1000 steps)"
  for i in $(seq 1 120); do
    st=$(ps -o stat= -p "$pid" 2>/dev/null)
    [ -z "$st" ] && break
    case "$st" in *Z*) break;; esac
    sleep 20
  done
  pgrep -f '[s]peculators/scripts/train\.py' | xargs -r kill -TERM; sleep 5
  pgrep -f '[s]peculators/scripts/train\.py' | xargs -r kill -9 2>/dev/null
  say "case $name stopped; analysing"
  DL_RAW="$raw" DL_OUT="$D/analysis_$name" /usr/local/python3.12.13/bin/python3 \
      "$D/analyze_dlprof.py" > "$D/logs/analysis_$name.txt" 2>&1
  say "case $name analysis -> logs/analysis_$name.txt"
}

run_case inorder1 "$D/train_diag_dp4_w2_inorder1.sh" "$D/raw_w2_inorder1"
run_case inorder0 "$D/train_diag_dp4_w2_inorder0.sh" "$D/raw_w2_inorder0"

say "stopping diag server (card10)"
[ -f "$D/logs/vllm_diag_inorder0.pid" ] && kill "$(cat $D/logs/vllm_diag_inorder0.pid)" 2>/dev/null
sleep 5; pkill -f "vllm serve /home/y50063564/Qwen3-4B" 2>/dev/null
say "AB DONE"
