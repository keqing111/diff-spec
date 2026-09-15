#!/bin/bash
# E4-D: is the server ceiling the effective concurrency (spec-decode slot
# accounting)? Fix W=8 and raise --max-num-seqs 8 -> 16 -> 32.
set -u
D=/home/y50063564/dataloader_diag
O="$D/logs/E4D.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$O"; }
an(){ /usr/local/python3.12.13/bin/python3 - "$D/exp/$1" "$1" <<'PY' | tee -a "$O"
import glob, json, re, sys, collections, numpy as np
R,label=sys.argv[1],sys.argv[2]
tb=None;te=None
for f in glob.glob(f'{R}/rank*/timeline_rank*.jsonl'):
    tl=[json.loads(l) for l in open(f)]; tb=min(x['t_batch'] for x in tl) if tb is None else min(tb,min(x['t_batch'] for x in tl)); te=max(x['t_end'] for x in tl) if te is None else max(te,max(x['t_end'] for x in tl))
dur=te-tb
n=0
for f in glob.glob(f'{R}/rank*/worker*.jsonl'):
    for l in open(f):
        if '"kind": "sample"' in l: n+=1
run=collections.defaultdict(list); wait=collections.defaultdict(list); it=[(1e18,0,0)]
for l in open(f'{R}/metrics.jsonl'):
    x=json.loads(l); m=x.get('m',{})
    for k,v in m.get('vllm:num_requests_running',{}).items(): run[k].append(v)
    for k,v in m.get('vllm:num_requests_waiting',{}).items(): wait[k].append(v)
    if 'vllm:iteration_tokens_total_count' in m:
        c=sum(m['vllm:iteration_tokens_total_count'].values()); s=sum(m['vllm:iteration_tokens_total_sum'].values())
        if it[0][0]==1e18: it[0]=(x['w'],c,s)
        it.append((x['w'],c,s))
d0,d1=it[0],it[-1]; steps=d1[1]-d0[1]; toks=d1[2]-d0[2]; dt=d1[0]-d0[0]
k=sorted(run)[0]; r=np.array(run[k]); w=np.array(wait[k])
print(f"\n=== {label} === {n/dur:.1f} docs/s | engine {steps/dt:.1f} steps/s, {toks/max(steps,1):.0f} tok/step, {toks/dt:.0f} tok/s")
print(f"  running p50={np.median(r):.1f} p90={np.percentile(r,90):.1f} max={r.max():.0f} | waiting p50={np.median(w):.1f} max={w.max():.0f}")
PY
}
for N in 16 32; do
  say "server MAX_NUM_SEQS=$N (token budget 32768)"
  pkill -f "vllm serve /home/y50063564/Qwen3-4B" 2>/dev/null; sleep 6
  MAX_NUM_SEQS=$N MAX_NUM_BATCHED_TOKENS=32768 bash "$D/serve_e1_dp1_1132_seqs.sh" >> "$O" 2>&1
  for i in $(seq 1 90); do curl -sf --noproxy '*' -m 3 http://80.48.17.178:1132/v1/models >/dev/null 2>&1 && break; sleep 10; done
  say "running W=8 with max_num_seqs=$N"
  bash "$D/run_exp.sh" E4D_w8_seqs${N} --workers 8 --prefetch 4 --max-steps 1200 --server-dp 1 \
       --samp-seed 42 --ag-max 1 --no-server --no-stop-server >> "$O" 2>&1
  an E4D_w8_seqs${N}
done
[ -f "$D/logs/vllm_e1_seqs.pid" ] && kill "$(cat $D/logs/vllm_e1_seqs.pid)" 2>/dev/null
sleep 5; pkill -f "vllm serve /home/y50063564/Qwen3-4B" 2>/dev/null
say "control (max_num_seqs=8, budget 4096):"; an E3_w8_pf4
say "E4D_CHAIN_DONE"
