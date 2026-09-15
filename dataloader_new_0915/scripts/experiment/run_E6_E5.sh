#!/bin/bash
# Chain: analyse E6 (perm) -> run E5 balance -> run E5' skew -> analyse both.
set -u
D=/home/y50063564/dataloader_diag
O="$D/logs/E6_E5.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$O"; }

analyse () {  # $1=run dir  $2=label
  local R="$D/exp/$1" L="$2"
  /usr/local/python3.12.13/bin/python3 - "$R" "$L" <<'PY' | tee -a "$O"
import glob, json, re, sys, collections, numpy as np
R, label = sys.argv[1], sys.argv[2]
ld=collections.defaultdict(list); dp=collections.defaultdict(list)
for f in glob.glob(f'{R}/rank*/load_rank*.jsonl'):
    r=int(re.search(r'/rank(\d+)/',f).group(1))
    for l in open(f):
        d=json.loads(l); ld[r].append((d['global_step'], d['load_ms'], d.get('docs_in_batch',-1)))
for f in glob.glob(f'{R}/rank*/worker*.jsonl'):
    r=int(re.search(r'/rank(\d+)/',f).group(1))
    for l in open(f):
        if '"kind": "collate"' in l: dp[r].append(json.loads(l)['t_perf'])
print(f"\n=== {label} ({R.split('/')[-1]}) ===")
print("  rank | load_ms p50 / p90 / >200ms 次数 | 平均 docs/batch | 生产批次")
for r in sorted(ld):
    v=[x[1] for x in ld[r]]; docs=[x[2] for x in ld[r]]
    print(f"   r{r}  | {np.median(v):8.1f} / {np.percentile(v,90):8.1f} / {sum(1 for x in v if x>200):5d}"
          f" | {np.mean(docs):6.2f} | {len(dp.get(r,[]))}")
PY
}

say "waiting for E6 to finish"
for i in $(seq 1 90); do grep -q "RUN_DONE" "$D/exp/E6_perm_rev/run.log" 2>/dev/null && break; sleep 30; done
say "E6 done; analysing"
analyse E6_perm_rev "E6 流置换 perm=3,2,1,0"

say "launching E5 balance (SAMP_BALANCE=docs)"
bash "$D/run_exp.sh" E5_balance_docs --workers 2 --prefetch 4 --max-steps 1500 --server-dp 1 \
     --samp-seed 42 --ag-max 1 --samp-balance docs >> "$O" 2>&1
say "E5 balance done; analysing"
analyse E5_balance_docs "E5 均衡 SAMP_BALANCE=docs"

say "launching E5' skew (SAMP_BALANCE=skew_docs)"
bash "$D/run_exp.sh" E5b_skew_docs --workers 2 --prefetch 4 --max-steps 1000 --server-dp 1 \
     --samp-seed 42 --ag-max 1 --samp-balance skew_docs >> "$O" 2>&1
say "E5' skew done; analysing"
analyse E5b_skew_docs "E5' 人为不均衡 SAMP_BALANCE=skew_docs"

say "chain done; baseline E1 for reference:"
analyse E1_ref_w2_pf4 "E1 基线（自然不均，对照）"
say "E6_E5_CHAIN_DONE"
