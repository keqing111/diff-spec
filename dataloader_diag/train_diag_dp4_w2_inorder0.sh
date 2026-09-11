#!/bin/bash
# Diagnostic trainer: dp4 (cards 0-3) + num_workers=2/rank + accum=3, 50k crop,
# 1 epoch, against the SINGLE-replica diag server (card4:1130). Reproduces the
# "server dp1 + trainer dp4 + W2" slow-rank combo with all 4 profiling layers ON.
set -e
export ASCEND_RT_VISIBLE_DEVICES="11,12,13,14"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178
export NO_PROXY="$no_proxy"
# --- profiling (all 4 layers, full logging, per rank/worker files) ---
export DL_PROFILE=1
export DL_MAX_STEPS=1000
export DL_PROFILE_DIR=/home/y50063564/dataloader_diag/raw_w2_inorder0
export DL_IN_ORDER=0

D=/home/y50063564/dataloader_diag
mkdir -p "$D/logs" "$DL_PROFILE_DIR"
LOG="$D/logs/train_diag_inorder0_$(date +%Y%m%d_%H%M%S).log"
nohup /usr/local/python3.12.13/bin/torchrun --standalone --nproc_per_node 4 \
  /home/y50063564/dspark_project/speculators/scripts/train.py \
  --verifier-name-or-path /home/y50063564/Qwen3-4B \
  --data-path /home/y50063564/data/open_perfectblend_qwen3_4b_50k \
  --vllm-endpoint http://80.48.17.178:1131/v1 \
  --save-path "$D/checkpoints_inorder0" \
  --draft-vocab-size 32000 --epochs 1 --lr 6e-4 --grad-accum 3 \
  --checkpoint-freq 1 --log-freq 20 --total-seq-len 4096 \
  --speculator-type dspark --draft-attn-impl sdpa --block-size 8 --max-anchors 256 \
  --num-layers 5 --target-layer-ids 1 9 17 25 33 --gqa-context-only-layer-indices 0 \
  --markov-rank 256 --markov-head-type vanilla --enable-confidence-head \
  --confidence-head-with-markov --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --confidence-head-alpha 1.0 --on-missing generate --on-generate delete \
  --request-timeout 900 --max-retries 5 --num-workers 2 \
  > "$LOG" 2>&1 &
echo $! > "$D/logs/train_diag.pid"
echo "diag trainer log: $LOG  pid: $(cat $D/logs/train_diag.pid)"
