#!/bin/bash
# P0-1 neutrality baseline: run the CURRENT code from the 28910 checkpoint for
# N steps with profiling OFF, capture per-step metrics, then stop.
# Usage: p0_baseline.sh <tag>     (tag in {before, after})
set -u
TAG="${1:-before}"
D=/home/y50063564/dataloader_diag
P0="$D/p0"; mkdir -p "$P0" "$D/logs"
O="$P0/p0_${TAG}.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$O"; }
N_STEPS=${N_STEPS:-60}

say "=== P0 baseline run tag=$TAG (N=$N_STEPS) ==="
say "starting dp2 server (cards 10,15; port 1123)"
bash /home/y50063564/dspark_project/script/dspark_accum_exp/serve_vllm_dp2.sh >> "$O" 2>&1
for i in $(seq 1 60); do
  curl -sf --noproxy '*' -m 3 http://80.48.17.178:1123/v1/models >/dev/null 2>&1 && { say "1123 ready"; break; }
  sleep 10
done
curl -sf --noproxy '*' -m 3 http://80.48.17.178:1123/v1/models >/dev/null 2>&1 || { say "SERVER FAILED"; exit 1; }

say "launching training (resume from checkpoint)"
unset DL_PROFILE AG_PROFILE DL_IN_ORDER SAMP_RANK_PERM SAMP_BALANCE SAMP_SEED
bash /home/y50063564/dspark_project/script/dspark_accum_exp/train_dp4_accum3_ctx70w.sh >> "$O" 2>&1
R=/home/y50063564/processed_data/dspark_data/dspark_accum/dp4_accum3_ctx70w
LOG=$(ls -t $R/logs/train_dp4_accum3_ctx70w_*.log | head -1)
say "trainer log: $LOG"

START_GS=""
for i in $(seq 1 180); do
  gs=$(grep -oE "global_step=[0-9]+" "$LOG" 2>/dev/null | tail -1 | cut -d= -f2)
  if [ -n "${gs:-}" ]; then
    [ -z "$START_GS" ] && { START_GS=$gs; say "first gs=$START_GS"; }
    if [ $((gs - START_GS)) -ge "$N_STEPS" ]; then say "reached gs=$gs (+$((gs-START_GS)) steps)"; break; fi
  fi
  sleep 10
done
say "stopping training"
pgrep -f '[s]peculators/scripts/train\.py' | xargs -r kill -TERM
for i in $(seq 1 30); do [ "$(pgrep -f '[s]peculators/scripts/train\.py' | wc -l)" -eq 0 ] && break; sleep 5; done
pgrep -f '[s]peculators/scripts/train\.py' | xargs -r kill -9 2>/dev/null

# extract per-step (global_step, loss, accept_len, ce_loss, tv_loss) in order
/usr/local/python3.12.13/bin/python3 - "$LOG" "$P0/p0_${TAG}_steps.txt" <<'PY'
import re, sys
log, out = sys.argv[1], sys.argv[2]
START=re.compile(r'^\[\d{2}:\d{2}:\d{2}\] INFO\s+train/')
KV=re.compile(r'([\w./]+)=([^,\s]+)')
rows=[]; cur=None
for line in open(log, encoding='utf-8', errors='replace'):
    if START.match(line):
        if cur: rows.append(cur)
        cur=''
        if True: cur=line.lstrip('\x00')
        continue
    if cur is not None and re.match(r'^\s+\S+=', line):
        cur += line
    elif cur is not None:
        rows.append(cur); cur=None
if cur: rows.append(cur)
seen=set(); recs=[]
for b in rows:
    kv=dict(KV.findall(b))
    if 'global_step' not in kv or 'train/loss' not in kv: continue
    gs=int(kv['global_step'])
    if gs in seen: continue
    seen.add(gs)
    recs.append((gs, kv.get('train/loss'), kv.get('train/accept_len'), kv.get('train/ce_loss'), kv.get('train/tv_loss')))
recs.sort()
with open(out,'w') as f:
    for r in recs:
        f.write("\t".join(str(x) for x in r)+"\n")
print(f"wrote {len(recs)} steps -> {out}")
print("first 5:", recs[:5])
PY

say "stopping dp2 server"
[ -f /home/y50063564/processed_data/dspark_data/dspark_accum/logs/vllm_dp2.pid ] && \
  kill "$(cat /home/y50063564/processed_data/dspark_data/dspark_accum/logs/vllm_dp2.pid)" 2>/dev/null
sleep 6; pkill -f "vllm serve /home/y50063564/Qwen3-4B" 2>/dev/null
say "P0_${TAG}_DONE"
