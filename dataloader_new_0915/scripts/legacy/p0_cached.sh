#!/bin/bash
# Deterministic neutrality check: train on a dataset whose hidden states are ALL
# cached, so the inputs are bit-reproducible (no vLLM in the loop). Running the
# same command with and without the instrumentation must give identical per-step
# metrics; that is what proves the instrumentation itself is neutral.
#
#   p0_cached.sh <tag>        # tag: fp0 | fp1 | ag1
set -u
TAG="${1:?usage: p0_cached.sh <tag>}"
D=/home/y50063564/dataloader_diag
R="$D/p0c/$TAG"; mkdir -p "$R"
DATA=/home/y50063564/processed_data/dspark_data/dspark_eval/val   # 5000 docs, hs cached
LOG="$R/train.log"
echo "[$(date +%H:%M:%S)] tag=$TAG data=$DATA" | tee -a "$D/p0c/p0c.log"

export ASCEND_RT_VISIBLE_DEVICES="11,12,13,14"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy=localhost,127.0.0.1,80.48.17.178
export NO_PROXY="$no_proxy"
export SAMP_SEED=42
unset DL_PROFILE AG_PROFILE DL_PROFILE_DIR DL_IN_ORDER SAMP_BALANCE SAMP_RANK_PERM
export DL_MAX_STEPS=40
case "$TAG" in
  ag*) export DL_PROFILE=1 DL_PROFILE_DIR="$R" AG_PROFILE=1 AG_MAX_PER_FWD=6 ;;
  p*)  export DL_PROFILE=1 DL_PROFILE_DIR="$R" ;;
  f*)  : ;;                        # no profiling
  *) echo "unknown tag $TAG" >&2; exit 2;;
esac

/usr/local/python3.12.13/bin/torchrun --standalone --nproc_per_node 4 \
  /home/y50063564/dspark_project/speculators/scripts/train.py \
  --verifier-name-or-path /home/y50063564/Qwen3-4B \
  --data-path "$DATA" \
  --vllm-endpoint http://127.0.0.1:9/v1 \
  --save-path "$R/checkpoints" \
  --draft-vocab-size 32000 --epochs 1 --lr 6e-4 --grad-accum 3 \
  --checkpoint-freq 1 --log-freq 1 --total-seq-len 4096 \
  --speculator-type dspark --draft-attn-impl sdpa --block-size 8 --max-anchors 256 \
  --num-layers 5 --target-layer-ids 1 9 17 25 33 --gqa-context-only-layer-indices 0 \
  --markov-rank 256 --markov-head-type vanilla --enable-confidence-head \
  --confidence-head-with-markov --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --confidence-head-alpha 1.0 --on-missing skip --on-generate delete \
  --request-timeout 60 --max-retries 1 --num-workers 2 --prefetch-factor 4 \
  > "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] tag=$TAG training exit=$?" | tee -a "$D/p0c/p0c.log"

/usr/local/python3.12.13/bin/python3 - "$LOG" "$R/steps.txt" <<'PY'
import re, sys
log, out = sys.argv[1], sys.argv[2]
START=re.compile(r'^\[\d{2}:\d{2}:\d{2}\] INFO\s+train/'); KV=re.compile(r'([\w./]+)=([^,\s]+)')
blocks=[]; cur=None
for line in open(log, encoding='utf-8', errors='replace'):
    if START.match(line):
        if cur: blocks.append(cur)
        cur=line
    elif cur is not None and re.match(r'^\s+\S+=', line): cur+=line
    elif cur is not None: blocks.append(cur); cur=None
if cur: blocks.append(cur)
recs=[]; seen=set()
for b in blocks:
    kv=dict(KV.findall(b))
    if 'global_step' not in kv or 'train/loss' not in kv: continue
    gs=int(kv['global_step'])
    if gs in seen: continue
    seen.add(gs)
    recs.append((gs, kv['train/loss'], kv.get('train/accept_len'), kv.get('train/ce_loss'), kv.get('train/tv_loss')))
recs.sort()
open(out,'w').write("\n".join("\t".join(map(str,r)) for r in recs)+"\n")
print(f"{out}: {len(recs)} steps; first={recs[0] if recs else None}")
PY
