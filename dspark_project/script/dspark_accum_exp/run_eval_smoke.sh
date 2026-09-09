#!/bin/bash
# Smoke: eval val100 for accum12 epoch0 (fast pipeline validation on card 11).
export ASCEND_RT_VISIBLE_DEVICES=11
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178
export NO_PROXY="$no_proxy"
cd /home/y50063564/dspark_project/script/dspark_accum_exp || exit 1
PY=/usr/local/python3.12.13/bin/python3
LOG=/home/y50063564/processed_data/dspark_data/dspark_accum/analysis/smoke_overfit.log
OUT=/home/y50063564/processed_data/dspark_data/dspark_accum/analysis/smoke_overfit.csv
"$PY" eval_checkpoints.py --runs accum12 --epochs 0 --subsets val100 \
    --out "$OUT" > "$LOG" 2>&1
echo "eval exit: $?"
tail -n 25 "$LOG"
