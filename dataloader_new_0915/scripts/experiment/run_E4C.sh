#!/bin/bash
# E4-C: does the server ceiling come from the token budget?
# Fix W=8 (the saturated point), vary the server's --max-num-batched-tokens:
#   4096 (already E3_w8_pf4) vs 16384 vs 32768.
set -u
D=/home/y50063564/dataloader_diag
O="$D/logs/E4C.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$O"; }

analyse () {
  local R="$D/exp/$1"
  /usr/local/python3.12.13/bin/python3 - "$R" "$1" <<'PY' | tee -a "$O"
import glob, json, re, sys, collections, numpy as np
R,label=sys.argv[1],sys.argv[2]
ld=collections.defaultdict(list); tb=None; te=None
for f in glob.glob(f'{R}/rank*/load_rank*.jsonl'):
    r=int(re.search(r'/rank(\d+)/',f).group(1))
    for l in open(f): ld[r].append(json.loads(l)['load_ms'])
for f in glob.glob(f'{R}/rank*/timeline_rank*.jsonl'):
    tl=[json.loads(l) for l in open(f)]
    a=min(x['t_batch'] for x in tl); b=max(x['t_end'] for x in tl)
    tb=a if tb is None else min(tb,a); te=b if te is None else max(te,b)
dur=te-tb; steps=max(len(v) for v in ld.values())
agg=0.0; docs=collections.defaultdict(int)
for f in glob.glob(f'{R}/rank*/worker*.jsonl'):
    r=int(re.search(r'/rank(\d+)/',f).group(1))
    for l in open(f):
        x=json.loads(l)
        if x['kind']=='sample': docs[r]+=1
run=collections.defaultdict(list); wait=collections.defaultdict(list); pt=[]
for l in open(f'{R}/metrics.jsonl'):
    x=json.loads(l); m=x.get('m',{})
    for k,v in m.get('vllm:num_requests_running',{}).items(): run[k].append(v)
    for k,v in m.get('vllm:num_requests_waiting',{}).items(): wait[k].append(v)
    if 'vllm:prompt_tokens_total' in m: pt.append((x['w'],sum(m['vllm:prompt_tokens_total'].values())))
tok_s=(pt[-1][1]-pt[0][1])/(pt[-1][0]-pt[0][0]) if len(pt)>1 else float('nan')
k=sorted(run)[0]; r=np.array(run[k]); w=np.array(wait[k])
print(f"\n=== {label} === 时长={dur:.0f}s 步/s={steps/dur:.2f}")
print(f"  总供给={sum(docs.values())/dur:.1f} docs/s | server prompt tokens/s={tok_s:.0f}")
print(f"  running p50={np.median(r):.1f} p90={np.percentile(r,90):.1f} max={r.max():.0f} | waiting p50={np.median(w):.1f} max={w.max():.0f}")
print("  各 rank 慢步(>200ms)占比: " + ", ".join(f"r{r}:{100*np.mean(np.array(ld[r])>200):.1f}%" for r in sorted(ld)))
PY
}

for TOK in 16384 32768; do
  say "starting server with MAX_NUM_BATCHED_TOKENS=$TOK"
  pkill -f "vllm serve /home/y50063564/Qwen3-4B" 2>/dev/null; sleep 6
  MAX_NUM_BATCHED_TOKENS=$TOK bash "$D/serve_e1_dp1_1132_tok.sh" >> "$O" 2>&1
  for i in $(seq 1 90); do curl -sf --noproxy '*' -m 3 http://80.48.17.178:1132/v1/models >/dev/null 2>&1 && { say "server ready"; break; }; sleep 10; done
  grep -a "max_num_scheduled_tokens" "$(ls -t $D/logs/vllm_e1_tok_*.log | head -1)" | tail -1 | cut -c1-160
  say "running W=8 with token budget $TOK"
  bash "$D/run_exp.sh" E4C_w8_tok${TOK} --workers 8 --prefetch 4 --max-steps 1200 --server-dp 1 \
       --samp-seed 42 --ag-max 1 --no-server --no-stop-server >> "$O" 2>&1
  say "done token=$TOK"; analyse E4C_w8_tok${TOK}
done
say "stopping server"; [ -f "$D/logs/vllm_e1_tok.pid" ] && kill "$(cat $D/logs/vllm_e1_tok.pid)" 2>/dev/null
sleep 5; pkill -f "vllm serve /home/y50063564/Qwen3-4B" 2>/dev/null
say "E4C control (token=4096, W=8) for comparison:"; analyse E3_w8_pf4
say "E4C_CHAIN_DONE"
