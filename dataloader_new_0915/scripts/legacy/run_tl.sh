#!/bin/bash
# Buffer-depletion experiment: dp4 (cards 11-14) + W2 + single-replica server
# (card10:1131), 1000 steps, recording per-step docs/batch + load_ms + phase
# timestamps in ONE aligned file per rank (raw_tl_w2).
set -u
D=/home/y50063564/dataloader_diag
O="$D/logs/tl.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$O"; }
mkdir -p "$D/logs"
rm -rf "$D/raw_tl_w2"; mkdir -p "$D/raw_tl_w2"

say "starting server card10:1131"
bash "$D/serve_diag_dp1_card10.sh" >> "$O" 2>&1
for i in $(seq 1 90); do curl -sf --noproxy '*' -m 3 http://80.48.17.178:1131/v1/models >/dev/null 2>&1 && { say "1131 ready"; break; }; sleep 10; done

export ASCEND_RT_VISIBLE_DEVICES="11,12,13,14"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy=localhost,127.0.0.1,80.48.17.178
export NO_PROXY="$no_proxy"
export DL_PROFILE=1 DL_PROFILE_DIR="$D/raw_tl_w2" DL_MAX_STEPS=1000

LOG="$D/logs/train_tl_w2_$(date +%Y%m%d_%H%M%S).log"
say "launching trainer (W2, 1000 steps cap)"
nohup /usr/local/python3.12.13/bin/torchrun --standalone --nproc_per_node 4 \
  /home/y50063564/dspark_project/speculators/scripts/train.py \
  --verifier-name-or-path /home/y50063564/Qwen3-4B \
  --data-path /home/y50063564/data/open_perfectblend_qwen3_4b_50k \
  --vllm-endpoint http://80.48.17.178:1131/v1 \
  --save-path "$D/checkpoints_tl_w2" \
  --draft-vocab-size 32000 --epochs 1 --lr 6e-4 --grad-accum 3 \
  --checkpoint-freq 1 --log-freq 20 --total-seq-len 4096 \
  --speculator-type dspark --draft-attn-impl sdpa --block-size 8 --max-anchors 256 \
  --num-layers 5 --target-layer-ids 1 9 17 25 33 --gqa-context-only-layer-indices 0 \
  --markov-rank 256 --markov-head-type vanilla --enable-confidence-head \
  --confidence-head-with-markov --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --confidence-head-alpha 1.0 --on-missing generate --on-generate delete \
  --request-timeout 900 --max-retries 5 --num-workers 2 > "$LOG" 2>&1 &
PID=$!
say "pid=$PID log=$LOG"
for i in $(seq 1 120); do
  st=$(ps -o stat= -p "$PID" 2>/dev/null)
  [ -z "$st" ] && break
  case "$st" in *Z*) break;; esac
  sleep 20
done
pgrep -f '[s]peculators/scripts/train\.py' | xargs -r kill -9 2>/dev/null
say "trainer done; stopping server"
[ -f "$D/logs/vllm_diag_inorder0.pid" ] && kill "$(cat $D/logs/vllm_diag_inorder0.pid)" 2>/dev/null
sleep 5; pkill -f "vllm serve /home/y50063564/Qwen3-4B" 2>/dev/null
say "TL DONE"
