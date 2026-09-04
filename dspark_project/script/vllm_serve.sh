export ASCEND_RT_VISIBLE_DEVICES=2
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_VERSION="0.22.1"
export VLLM_ASCEND_ENABLE_NZ=0

vllm serve /home/y50063564/dspark_project/fakemodelconfig \
    --host 0.0.0.0 \
    --port 1123 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --seed 1024 \
    --max-num-seqs 4 \
    --max-model-len 1024 \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --hf-overrides '{"use_index_cache": true}' \
    --load-format dummy \
    --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [1,2,3]}}}' \
    --kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "/home/y50063564/dspark_project/processed_data"}}' \
    --no-enable-chunked-prefill \
    --enforce-eager