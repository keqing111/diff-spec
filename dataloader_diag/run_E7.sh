#!/bin/bash
# E7: uniform-request-size experiment (doc-length band 600-800) at W=8 and W=4.
# Matched-rho control: E3_w8_pf4 (same W=8, varied doc lengths, imbalanced docs).
set -u
D=/home/y50063564/dataloader_diag
BAND=/home/y50063564/data/open_perfectblend_qwen3_4b_band600_800
O="$D/logs/E7.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$O"; }

an(){ /usr/local/python3.12.13/bin/python3 - "$D/exp/$1" "$1" <<'PY' | tee -a "$O"
import glob, json, re, sys, collections, numpy as np
R,label=sys.argv[1],sys.argv[2]
ld=collections.defaultdict(list); docs=collections.defaultdict(list)
samp=collections.Counter(); sw=collections.Counter(); cpu=collections.Counter()
for f in glob.glob(f'{R}/rank*/load_rank*.jsonl'):
    r=int(re.search(r'/rank(\d+)/',f).group(1))
    for l in open(f):
        x=json.loads(l); ld[r].append(x['load_ms']); docs[r].append(x.get('docs_in_batch',0))
for f in glob.glob(f'{R}/rank*/worker*.jsonl'):
    r=int(re.search(r'/rank(\d+)/',f).group(1))
    for l in open(f):
        x=json.loads(l)
        if x['kind']=='sample': samp[r]+=1; sw[r]+=x['wall_ms']; cpu[r]+=x.get('cpu_ms',0)
tb=None;te=None
for f in glob.glob(f'{R}/rank*/timeline_rank*.jsonl'):
    tl=[json.loads(l) for l in open(f)]
    a=min(x['t_batch'] for x in tl); b=max(x['t_end'] for x in tl)
    tb=a if tb is None else min(tb,a); te=b if te is None else max(te,b)
dur=te-tb; steps=max(len(v) for v in ld.values()); step_t=dur/steps
print(f"\n=== {label} === 时长={dur:.0f}s 步/s={steps/dur:.2f} step_time={step_t:.2f}s")
run=collections.defaultdict(list); wait=collections.defaultdict(list)
try:
    for l in open(f'{R}/metrics.jsonl'):
        x=json.loads(l); m=x.get('m',{})
        for k,v in m.get('vllm:num_requests_running',{}).items(): run[k].append(v)
        for k,v in m.get('vllm:num_requests_waiting',{}).items(): wait[k].append(v)
    k=sorted(run)[0]; r1=np.array(run[k]); w1=np.array(wait[k])
    print(f"  server running p50={np.median(r1):.1f} max={r1.max():.0f} | waiting p50={np.median(w1):.1f} max={w1.max():.0f}")
except FileNotFoundError: pass
for r in sorted(ld):
    rtt=(sw[r]/samp[r])/1000.0 if samp[r] else float('nan')
    nw=len(glob.glob(f'{R}/rank{r}/worker*.jsonl'))
    cap=nw/rtt; dem=np.mean(docs[r])/step_t
    a=np.array(ld[r])
    print(f"   r{r}: docs/batch={np.mean(docs[r]):5.2f} RTT={rtt*1000:5.0f}ms CPU/doc={cpu[r]/max(samp[r],1):5.0f}ms "
          f"| 需求={dem:5.2f} 能力={cap:5.2f} **ρ={dem/cap:5.3f}** | 慢步>200ms={100*np.mean(a>200):5.1f}% fast<1ms={100*np.mean(a<1):5.1f}%")
PY
}

say "starting server (card10:1132, default seqs/budget)"
curl -sf --noproxy '*' -m 3 http://80.48.17.178:1132/v1/models >/dev/null 2>&1 || bash "$D/serve_e1_dp1_1132.sh" >> "$O" 2>&1
for i in $(seq 1 90); do curl -sf --noproxy '*' -m 3 http://80.48.17.178:1132/v1/models >/dev/null 2>&1 && { say "server ready"; break; }; sleep 10; done

for W in 8 4; do
  say "band(600-800) W=$W"
  bash "$D/run_exp.sh" E7_band_w${W} --workers $W --prefetch 4 --max-steps 1200 --server-dp 1 \
       --samp-seed 42 --ag-max 1 --data "$BAND" --no-server --no-stop-server >> "$O" 2>&1
  an E7_band_w${W}
done
[ -f "$D/logs/vllm_e1.pid" ] && kill "$(cat $D/logs/vllm_e1.pid)" 2>/dev/null
sleep 5; pkill -f "vllm serve /home/y50063564/Qwen3-4B" 2>/dev/null
say "controls (varied lengths):"; an E3_w8_pf4; an E1_ref_w2_pf4
say "E7_DONE"
