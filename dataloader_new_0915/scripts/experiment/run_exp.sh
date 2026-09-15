#!/bin/bash
# Generic experiment runner for the root-cause validation program.
#
#   run_exp.sh <RUN_ID> [--workers N] [--prefetch N] [--max-steps N]
#              [--server-dp N] [--in-order 0|1] [--samp-balance off|docs|tokens|skew_docs]
#              [--samp-perm a,b,c,d] [--samp-seed N] [--ag-max N] [--data PATH]
#              [--no-scrape] [--no-server] [--no-stop-server]
#
# Writes everything under $EXP/<RUN_ID>/: per-rank profile jsonl, metrics.jsonl,
# sys.jsonl, config.json (+ diff.patch), train log.
set -u
EXP=/home/y50063564/dataloader_diag/exp
REPO=/home/y50063564/dspark_project/speculators
RUN_ID="${1:?usage: run_exp.sh RUN_ID [opts]}"; shift

WORKERS=2; PREFETCH=4; MAX_STEPS=1500; SERVER_DP=1; IN_ORDER=1
SAMP_BALANCE="off"; SAMP_PERM=""; SAMP_SEED=42; AG_MAX=1
DATA=/home/y50063564/data/open_perfectblend_qwen3_4b_700k
SCRAPE=1; START_SERVER=1; STOP_SERVER=1
while [ $# -gt 0 ]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2;;
    --prefetch) PREFETCH="$2"; shift 2;;
    --max-steps) MAX_STEPS="$2"; shift 2;;
    --server-dp) SERVER_DP="$2"; shift 2;;
    --in-order) IN_ORDER="$2"; shift 2;;
    --samp-balance) SAMP_BALANCE="$2"; shift 2;;
    --samp-perm) SAMP_PERM="$2"; shift 2;;
    --samp-seed) SAMP_SEED="$2"; shift 2;;
    --ag-max) AG_MAX="$2"; shift 2;;
    --data) DATA="$2"; shift 2;;
    --no-scrape) SCRAPE=0; shift;;
    --no-server) START_SERVER=0; shift;;
    --no-stop-server) STOP_SERVER=0; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done

R=$EXP/$RUN_ID
mkdir -p "$R"
LOG=$R/train.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$R/run.log"; }

PORT=1132
say "=== RUN $RUN_ID === workers=$WORKERS prefetch=$PREFETCH max_steps=$MAX_STEPS server_dp=$SERVER_DP in_order=$IN_ORDER balance=$SAMP_BALANCE perm='$SAMP_PERM' seed=$SAMP_SEED ag_max=$AG_MAX"
say "data=$DATA  server_port=$PORT"

# ---- config archive ----
{
  echo "run_id $RUN_ID"
  echo "git_head $(git -C "$REPO" rev-parse HEAD)"
  echo "workers $WORKERS"; echo "prefetch_factor $PREFETCH"
  echo "max_steps $MAX_STEPS"; echo "server_dp $SERVER_DP"; echo "in_order $IN_ORDER"
  echo "samp_balance $SAMP_BALANCE"; echo "samp_perm $SAMP_PERM"; echo "samp_seed $SAMP_SEED"
  echo "ag_max_per_fwd $AG_MAX"; echo "data_path $DATA"
  echo "train_dp 4"; echo "train_cards 11,12,13,14"; echo "server_cards 10[,15]"
  echo "start_wall $(date -Is)"
} > "$R/config.txt"
git -C "$REPO" diff > "$R/diff.patch" 2>/dev/null || true

# ---- server ----
if [ "$START_SERVER" = "1" ]; then
  if curl -sf --noproxy '*' -m 3 "http://80.48.17.178:$PORT/v1/models" >/dev/null 2>&1; then
    say "server already up on $PORT"
  else
    if [ "$SERVER_DP" = "2" ]; then bash "$EXP/../serve_e1_dp2_1132.sh" >> "$R/run.log" 2>&1
    else bash "$EXP/../serve_e1_dp1_1132.sh" >> "$R/run.log" 2>&1; fi
    for i in $(seq 1 60); do
      curl -sf --noproxy '*' -m 3 "http://80.48.17.178:$PORT/v1/models" >/dev/null 2>&1 && { say "server ready"; break; }
      sleep 10
    done
  fi
fi
curl -sf --noproxy '*' -m 3 "http://80.48.17.178:$PORT/v1/models" >/dev/null 2>&1 || { say "SERVER NOT READY - abort"; exit 1; }

# ---- scrapers ----
SPIDS=()
if [ "$SCRAPE" = "1" ]; then
  /usr/local/python3.12.13/bin/python3 /home/y50063564/dataloader_diag/scrape_metrics.py \
      --url "http://80.48.17.178:$PORT/metrics" --out "$R/metrics.jsonl" --interval 1 >> "$R/run.log" 2>&1 &
  SPIDS+=($!)
  /usr/local/python3.12.13/bin/python3 /home/y50063564/dataloader_diag/scrape_sys.py \
      --out "$R/sys.jsonl" --interval 1 --pattern "speculators/scripts/train.py" --pattern "vllm" >> "$R/run.log" 2>&1 &
  SPIDS+=($!)
  sleep 2
  say "scrapers started: ${SPIDS[*]}"
fi

# ---- training ----
export ASCEND_RT_VISIBLE_DEVICES="11,12,13,14"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy=localhost,127.0.0.1,80.48.17.178
export NO_PROXY="$no_proxy"
export DL_PROFILE=1 DL_PROFILE_DIR="$R" DL_MAX_STEPS="$MAX_STEPS" DL_IN_ORDER="$IN_ORDER"
export AG_PROFILE=1 AG_MAX_PER_FWD="$AG_MAX"
export SAMP_SEED="$SAMP_SEED" SAMP_BALANCE="$SAMP_BALANCE"
if [ -n "$SAMP_PERM" ]; then export SAMP_RANK_PERM="$SAMP_PERM"; else unset SAMP_RANK_PERM; fi

say "launching trainer"
nohup /usr/local/python3.12.13/bin/torchrun --standalone --nproc_per_node 4 \
  "$REPO/scripts/train.py" \
  --verifier-name-or-path /home/y50063564/Qwen3-4B \
  --data-path "$DATA" \
  --vllm-endpoint "http://80.48.17.178:$PORT/v1" \
  --save-path "$R/checkpoints" \
  --draft-vocab-size 32000 --epochs 1 --lr 6e-4 --grad-accum 3 \
  --checkpoint-freq 1 --log-freq 20 --total-seq-len 4096 \
  --speculator-type dspark --draft-attn-impl sdpa --block-size 8 --max-anchors 256 \
  --num-layers 5 --target-layer-ids 1 9 17 25 33 --gqa-context-only-layer-indices 0 \
  --markov-rank 256 --markov-head-type vanilla --enable-confidence-head \
  --confidence-head-with-markov --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --confidence-head-alpha 1.0 --on-missing generate --on-generate delete \
  --request-timeout 900 --max-retries 5 --num-workers "$WORKERS" \
  --prefetch-factor "$PREFETCH" > "$LOG" 2>&1 &
TPID=$!
say "trainer pid=$TPID"

for i in $(seq 1 240); do
  st=$(ps -o stat= -p "$TPID" 2>/dev/null)
  [ -z "$st" ] && { say "trainer exited"; break; }
  case "$st" in *Z*) say "trainer zombie (done)"; break;; esac
  sleep 15
done
pgrep -f '[s]peculators/scripts/train\.py' | xargs -r kill -9 2>/dev/null

for p in "${SPIDS[@]:-}"; do [ -n "${p:-}" ] && kill "$p" 2>/dev/null; done
if [ "$STOP_SERVER" = "1" ]; then
  [ -f /home/y50063564/dataloader_diag/logs/vllm_e1.pid ] && kill "$(cat /home/y50063564/dataloader_diag/logs/vllm_e1.pid)" 2>/dev/null
  sleep 5; pkill -f "vllm serve /home/y50063564/Qwen3-4B" 2>/dev/null
fi
echo "end_wall $(date -Is)" >> "$R/config.txt"
say "profile files: $(find "$R" -name '*.jsonl' | wc -l)  (rank dirs: $(ls -d $R/rank* 2>/dev/null | wc -l))"
say "RUN_DONE $RUN_ID"
