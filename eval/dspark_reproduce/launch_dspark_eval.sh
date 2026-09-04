#!/bin/bash
# 复现 HF DSpark 评测: 以 dspark 投机解码模式启动 vLLM(卡 12, 端口 1125)
#   verifier = 本地 /home/y50063564/Qwen3-4B(几何与官方 Qwen3-4B 一致)
#   draft    = 下载的 RedHatAI/Qwen3-4B-speculator.dspark
# 用法: bash launch_dspark_eval.sh    (前台运行; 就绪后另开终端跑 eval_run.sh)
export ASCEND_RT_VISIBLE_DEVICES=12
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_ASCEND_ENABLE_NZ=0

vllm serve /home/y50063564/Qwen3-4B \
    --host 0.0.0.0 \
    --port 1125 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --speculative-config '{"model": "/home/y50063564/models/RedHatAI/Qwen3-4B-speculator.dspark", "num_speculative_tokens": 7, "method": "dspark"}'
