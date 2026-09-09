#!/bin/bash
# Full overfit/degradation evaluation: accum12 + gqa_ctx_l0_a12, epochs 0..2,
# subsets train1000/val/val100/out1000. Card 11; logs + CSV under analysis/.
export ASCEND_RT_VISIBLE_DEVICES=11
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178
export NO_PROXY="$no_proxy"
cd /home/y50063564/dspark_project/script/dspark_accum_exp || exit 1
PY=/usr/local/python3.12.13/bin/python3
ROOT=/home/y50063564/processed_data/dspark_data/dspark_accum/analysis
"$PY" eval_checkpoints.py \
    --runs accum12 gqa_ctx_l0_a12 \
    --epochs 0 1 2 \
    --subsets train1000 val val100 out1000 \
    --out "$ROOT/overfit_check.csv" > "$ROOT/overfit_check.log" 2>&1
echo "eval_full exit: $?"
tail -n 60 "$ROOT/overfit_check.log"
