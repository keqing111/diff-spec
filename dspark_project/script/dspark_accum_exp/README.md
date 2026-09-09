# DSpark accum 对比实验（qwen3-4B, 5 层 GQA）

对比同一 dspark 配置在**不同参数更新频率**下的接受长度与 loss 曲线：
同一份 5w 子集 ×3 epoch，两组唯一差别是 `--grad-accum`（更新频率）。

| 组 | 卡 | 更新频率 | 语义 |
|----|----|----------|------|
| A `accum=1` | 13 | 每个微批更新 | 现状默认 |
| B `accum=12` | 14 | 每 12 个微批更新一次 | loss÷12 累积（= DDP dp=12 平均语义）|

hidden states 完全在线：训练时按需向卡 15 的 vLLM server 生成（载入即删）。

## 目录
- `prepare_dataset.py` —— 从 `open_perfectblend_qwen3_4b_700k` 裁前 5 万行 ->
  `data/open_perfectblend_qwen3_4b_50k`（train 0.9 / val 0.1 在加载时切）。
- `verify_accum_math.py` —— 纯 CPU 数值对拍：grad_accum(loss/accum) 的梯度
  ≡ 把 accum 个微批拼成大 batch 求均值（即 DDP 平均）。已跑通（~1e-8）。
- `serve_vllm.sh` —— 卡 15 起 DSpark hidden-state vLLM server（端口 1123）。
- `train_accum1.sh` / `train_accum12.sh` —— 两组训练（torchrun 单卡 13/14）。
- `run_all.sh` —— 依次：起 server -> 等就绪 -> 并发起两组训练。
- `compare_accum.py` —— 解析两组日志 -> CSV + accept_len/loss 对比图。

## 代码改动（speculators 仓库，语义 B）
`src/speculators/train/trainer.py` + `scripts/train.py`：
- 新增可选 `--grad-accum N`（默认 1，不影响既有 run）。
- `train_epoch`：每个微批照常 forward 并记录 accept_len/loss；`loss /= accum`
  后 backward 累积；累计满 N 个微批才 `clip -> optimizer.step ->
  zero_grad -> scheduler.step`；epoch 末不完整窗口丢弃并清空梯度。
- LR scheduler 步数按优化步计：total = epochs × (微批数 // accum)，与
  dp-N 全局步节奏一致。`accum=1` 时行为与改造前逐字节等价。

## 关键配置（两组完全一致，除 grad-accum / run-name / 输出目录 / 卡）
verifier=Qwen3-4B，data=50k 子集，lr=6e-4，epochs=3，total_seq_len=4096，
speculator=dspark，draft_attn_impl=sdpa，num_layers=5，
target_layer_ids=1 9 17 25 33，markov_rank=256，markov_head=vanilla，
enable_confidence_head + with_markov，loss_fn={"ce":0.1,"tv":0.9}，
confidence_head_alpha=1.0，checkpoint_freq=1（每 epoch 末存），log_freq=20，
on_missing=generate，on_generate=delete。

## 输出
processed_data/dspark_data/dspark_accum/
  accum1/   accum12/   logs/(server)   analysis/(图与 csv)
