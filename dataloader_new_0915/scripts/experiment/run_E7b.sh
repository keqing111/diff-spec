#!/bin/bash
set -u
D=/home/y50063564/dataloader_diag
BAND=/home/y50063564/data/open_perfectblend_qwen3_4b_band600_800
O="$D/logs/E7b.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$O"; }
curl -sf --noproxy '*' -m 3 http://80.48.17.178:1132/v1/models >/dev/null 2>&1 || bash "$D/serve_e1_dp1_1132.sh" >> "$O" 2>&1
for i in $(seq 1 90); do curl -sf --noproxy '*' -m 3 http://80.48.17.178:1132/v1/models >/dev/null 2>&1 && break; sleep 10; done
say "band W=2"
bash "$D/run_exp.sh" E7b_band_w2 --workers 2 --prefetch 4 --max-steps 1200 --server-dp 1 \
     --samp-seed 42 --ag-max 1 --data "$BAND" --no-server --no-stop-server >> "$O" 2>&1
say "band W=2 done"
[ -f "$D/logs/vllm_e1.pid" ] && kill "$(cat $D/logs/vllm_e1.pid)" 2>/dev/null
sleep 5; pkill -f "vllm serve /home/y50063564/Qwen3-4B" 2>/dev/null
say "E7B_DONE"
