#!/bin/bash
# E3: num_workers sweep at the reference config (dp4, server dp1 card10:1132,
# prefetch 4, seed 42, 700k data). Server is started once and reused.
# W=2 already exists as E1_ref_w2_pf4.
set -u
D=/home/y50063564/dataloader_diag
O="$D/logs/E3.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$O"; }

analyse () {  # $1=run_dir
  local R="$D/exp/$1"
  /usr/local/python3.12.13/bin/python3 - "$R" "$1" <<'PY' | tee -a "$O"
import glob, json, re, sys, collections, numpy as np, pandas as pd
R,label=sys.argv[1],sys.argv[2]
ld=collections.defaultdict(list); docs=collections.defaultdict(list); tb=None; te=None
for f in glob.glob(f'{R}/rank*/load_rank*.jsonl'):
    r=int(re.search(r'/rank(\d+)/',f).group(1))
    for l in open(f):
        x=json.loads(l); ld[r].append(x['load_ms']); docs[r].append(x.get('docs_in_batch',0))
for f in glob.glob(f'{R}/rank*/timeline_rank*.jsonl'):
    tl=[json.loads(l) for l in open(f)]
    a=min(x['t_batch'] for x in tl); b=max(x['t_end'] for x in tl)
    tb=a if tb is None else min(tb,a); te=b if te is None else max(te,b)
dur=te-tb if tb else float('nan'); steps=max(len(v) for v in ld.values())
samp=collections.Counter(); sw=collections.Counter()
for f in glob.glob(f'{R}/rank*/worker*.jsonl'):
    r=int(re.search(r'/rank(\d+)/',f).group(1))
    for l in open(f):
        x=json.loads(l)
        if x['kind']=='sample': samp[r]+=1; sw[r]+=x['wall_ms']
# server metrics
run=collections.defaultdict(list); wait=collections.defaultdict(list)
try:
    for l in open(f'{R}/metrics.jsonl'):
        x=json.loads(l); m=x.get('m',{})
        for k,v in m.get('vllm:num_requests_running',{}).items():
            e=k.split('engine="')[1].split('"')[0] if 'engine="' in k else '?'; run[e].append(v)
        for k,v in m.get('vllm:num_requests_waiting',{}).items():
            e=k.split('engine="')[1].split('"')[0] if 'engine="' in k else '?'; wait[e].append(v)
except FileNotFoundError:
    pass
R_=np.array(run[sorted(run)[0]]) if run else np.array([np.nan])
W_=np.array(wait[sorted(wait)[0]]) if wait else np.array([np.nan])
print(f"\n=== {label} ===  时长={dur:.0f}s  步数={steps}  step/s={steps/dur:.2f}")
print(f"  server: running p50={np.nanmedian(R_):.1f} p90={np.nanpercentile(R_,90):.1f} max={np.nanmax(R_):.0f} | waiting max={np.nanmax(W_):.0f}")
tot_docs=0
for r in sorted(ld):
    ds=np.mean(docs[r]); tot_docs+=sum(docs[r])
    print(f"   r{r}: load p50={np.median(ld[r]):7.1f} slow(>200ms)={100*np.mean(np.array(ld[r])>200):5.1f}% "
          f"| docs/batch={ds:5.2f} | 该rank供给={samp[r]/dur:5.2f} docs/s (Σ往返{sw[r]/1000:.0f}s)")
print(f"  合计：消费 {tot_docs/dur:.2f} docs/s；客户端供给 Σ={sum(samp.values())/dur:.2f} docs/s")
PY
}

say "starting server once"
bash "$D/run_exp.sh" E3_w1_pf4 --workers 1 --prefetch 4 --max-steps 1200 --server-dp 1 --samp-seed 42 --ag-max 1 --no-stop-server >> "$O" 2>&1
say "E3 W=1 done"; analyse E3_w1_pf4

for W in 4 6 8 12; do
  say "running W=$W"
  bash "$D/run_exp.sh" E3_w${W}_pf4 --workers $W --prefetch 4 --max-steps 1200 --server-dp 1 \
       --samp-seed 42 --ag-max 1 --no-server --no-stop-server >> "$O" 2>&1
  say "E3 W=$W done"; analyse E3_w${W}_pf4
done

say "stopping server"
[ -f "$D/logs/vllm_e1.pid" ] && kill "$(cat $D/logs/vllm_e1.pid)" 2>/dev/null
sleep 5; pkill -f "vllm serve /home/y50063564/Qwen3-4B" 2>/dev/null
say "E3 reference W=2 (E1) for comparison:"; analyse E1_ref_w2_pf4
say "E3_CHAIN_DONE"
