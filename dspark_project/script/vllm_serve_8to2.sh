# 8:2(=4:1) 生产配置 vLLM server:1 个服务,2 个 data-parallel 副本(卡 14,15)
# 用法: 先撤掉旧 vLLM(卡 7 / 端口 1123),再起本脚本;随后起 train_8to2.sh
export ASCEND_RT_VISIBLE_DEVICES=14,15
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_VERSION="0.22.1"
export VLLM_ASCEND_ENABLE_NZ=0

vllm serve /home/y50063564/Qwen3-4B \
    --host 0.0.0.0 \
    --port 1123 \
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
    --kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "/home/y50063564/processed_data/dspark_data/dspark_8dp"}}' \
    --no-enable-chunked-prefill
