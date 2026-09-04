# 生产 4:1 配置 vLLM server:1 副本(卡 15),服务 4 个 trainer(卡 11-14)。
# 共用服务器约束:只用最后 5 张卡。先起本脚本,再起 train_qwen3-4b.sh。
export ASCEND_RT_VISIBLE_DEVICES=15
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
    --data-parallel-size 1 \
    --hf-overrides '{"use_index_cache": true}' \
    --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [1,9,17,25,33,36]}}}' \
    --kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "/home/y50063564/processed_data/dspark_data/dspark_4to1"}}' \
    --no-enable-chunked-prefill
    
    #--kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "/home/z00841464/dspark_project/outputs/hidden_states"}}' \
    #--kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "/data/z00841464/dspark_data/hidden_states"}}' \
