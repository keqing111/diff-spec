#!/bin/bash
# Single-API-endpoint hidden-state server with vLLM data-parallel size 2 on
# cards 10 and 15. Topology mirrors the .185 dp4 server (ONE port, DP engines
# behind it) so the trainer is completely unaware of the extra replica.
set -e

export ASCEND_RT_VISIBLE_DEVICES=10,15
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_VERSION="0.22.1"
export VLLM_ASCEND_ENABLE_NZ=0
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178
export NO_PROXY="$no_proxy"

RUN_ROOT=/home/y50063564/processed_data/dspark_data/dspark_accum
HS_DIR=$RUN_ROOT/_hs_e1_dp2
LOG_DIR=$RUN_ROOT/logs
mkdir -p "$HS_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_e1_dp2_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/vllm_e1_dp2.pid"

nohup vllm serve /home/y50063564/Qwen3-4B \
    --host 0.0.0.0 \
    --port 1132 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --max-num-seqs 8 \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --data-parallel-size 2 \
    --hf-overrides '{"use_index_cache": true}' \
    --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [1,9,17,25,33,36]}}}' \
    --kv_transfer_config "{\"kv_connector\": \"ExampleHiddenStatesConnector\", \"kv_role\": \"kv_producer\", \"kv_connector_extra_config\": {\"shared_storage_path\": \"$HS_DIR\"}}" \
    --no-enable-chunked-prefill \
    > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "vLLM dp2 log : $LOG_FILE"
echo "vLLM dp2 pid : $(cat "$PID_FILE")"
