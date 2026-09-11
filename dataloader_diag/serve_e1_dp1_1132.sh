#!/bin/bash
# Diagnostic hidden-state vLLM server: SINGLE replica (dp1) on card 4, port 1130.
# Reproduces the "server dp1" half of the slow-rank combo.
set -e
export ASCEND_RT_VISIBLE_DEVICES=10
export HCCL_OP_EXPANSION_MODE="AIV" OMP_PROC_BIND=false OMP_NUM_THREADS=1 HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0 VLLM_VERSION="0.22.1" VLLM_ASCEND_ENABLE_NZ=0
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178
export NO_PROXY="$no_proxy"
D=/home/y50063564/dataloader_diag
mkdir -p "$D/_hs_e1_dp1" "$D/logs"
LOG="$D/logs/vllm_e1_$(date +%Y%m%d_%H%M%S).log"
nohup vllm serve /home/y50063564/Qwen3-4B --host 0.0.0.0 --port 1132 \
  --tensor-parallel-size 1 --seed 1024 --max-num-seqs 8 --max-model-len 4096 \
  --trust-remote-code --gpu-memory-utilization 0.90 --enforce-eager --data-parallel-size 1 \
  --hf-overrides '{"use_index_cache": true}' \
  --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [1,9,17,25,33,36]}}}' \
  --kv_transfer_config "{\"kv_connector\": \"ExampleHiddenStatesConnector\", \"kv_role\": \"kv_producer\", \"kv_connector_extra_config\": {\"shared_storage_path\": \"$D/_hs_e1_dp1\"}}" \
  --no-enable-chunked-prefill > "$LOG" 2>&1 &
echo $! > "$D/logs/vllm_e1.pid"
echo "diag server log: $LOG  pid: $(cat $D/logs/vllm_e1.pid)"
