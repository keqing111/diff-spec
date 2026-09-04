#!/bin/bash
# 单卡 GQA baseline（5 层、无 diff），与 dspark_diff_l0_5l_{kfix,ctxonly} 同一套配置，仅去掉 diff。
# 用于和单卡 diff 实验做同量纲(样本数/epoch)对比。卡 12。
set -e

export ASCEND_RT_VISIBLE_DEVICES=12
# vLLM 是内网地址, 必须绕过公司 http 代理(否则 openai 请求被代理劫持返回 HTML)
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178
export NO_PROXY="$no_proxy"
PYTHON=/usr/local/python3.12.13/bin/python3
TORCHRUN=/usr/local/python3.12.13/bin/torchrun

RUN_NAME=baseline_single
OUT=/home/y50063564/processed_data/dspark_data/dspark_${RUN_NAME}
LOG_DIR=$OUT/logs
CKPT_DIR=$OUT/checkpoints
mkdir -p "$LOG_DIR" "$CKPT_DIR"
LOG_FILE="$LOG_DIR/train_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train.pid"

nohup "$TORCHRUN" --standalone --nproc_per_node 1 \
    /home/y50063564/dspark_project/speculators/scripts/train.py \
    --verifier-name-or-path /home/y50063564/Qwen3-4B \
    --data-path /home/y50063564/data/open_perfectblend_qwen3_4b_700k \
    --vllm-endpoint http://80.48.17.178:1123/v1 \
    --save-path "$CKPT_DIR" \
    --draft-vocab-size 32000 \
    --epochs 3 \
    --lr 6e-4 \
    --checkpoint-freq 0.1 \
    --total-seq-len 4096 \
    --speculator-type dspark \
    --draft-attn-impl sdpa \
    --block-size 8 \
    --max-anchors 256 \
    --num-layers 5 \
    --target-layer-ids 1 9 17 25 33 \
    --markov-rank 256 \
    --markov-head-type vanilla \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' \
    --confidence-head-alpha 1.0 \
    --on-missing generate \
    --on-generate delete \
    --request-timeout 900 \
    --max-retries 5 \
    --num-workers 4 \
    > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Log file: $LOG_FILE"
