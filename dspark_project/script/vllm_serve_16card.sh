# 16 卡满配 vLLM server(另一台服务器 80.48.17.185,无共用约束):
#   4 个 data-parallel 副本(卡 12-15),服务 12 个 trainer(卡 0-11)。
# 前提: docker 环境与路径和 80.48.17.178 一致
#   - /home/y50063564/Qwen3-4B 与数据目录需存在
#   - 先检查 npu-smi info 没有 Alarm 设备
# 用法: 先起本脚本,再起 train_16card.sh
export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_VERSION="0.22.1"
export VLLM_ASCEND_ENABLE_NZ=0
export NO_PROXY=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178
export no_proxy=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178

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
    --data-parallel-size 4 \
    --hf-overrides '{"use_index_cache": true}' \
    --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [1,9,17,25,33,36]}}}' \
    --kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "/home/y50063564/processed_data/dspark_data/dspark_16card"}}' \
    --no-enable-chunked-prefill
