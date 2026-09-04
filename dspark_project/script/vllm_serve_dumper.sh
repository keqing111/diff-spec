#!/usr/bin/env bash
# Dumper(Plan B) 版 serve —— 用 dspark_hs_dumper 生成 hidden_states。
# dspark_hs_dumper.py + model_runner 接线已迁移到已安装的 vllm-ascend
# (/vllm-workspace/vllm-ascend)，此处正常启动即可。
set -eo pipefail
cd /home/y50063564/dspark_project

export ASCEND_RT_VISIBLE_DEVICES=3
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_VERSION="0.22.1"
export VLLM_ASCEND_ENABLE_NZ=0

# ===== Dumper 开关 =====
export DSPARK_HS_DUMP=1                                    # 开启 Plan B HS dumper
export DSPARK_HS_DIR=/home/y50063564/dspark_project/hs_dump_test   # HS 输出目录
export DSPARK_HS_LAYERS=1,2                                # aux target layers（逗号分隔）

vllm serve /home/y50063564/dspark_project/fakemodelconfig \
    --host 0.0.0.0 \
    --port 1125 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --seed 1024 \
    --max-num-seqs 4 \
    --max-model-len 1024 \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --hf-overrides '{"use_index_cache": true}' \
    --load-format dummy \
    --no-enable-chunked-prefill \
    --enforce-eager
