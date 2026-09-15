# Baseline — DSpark / Qwen3-4B / open_perfectblend_70w(16 卡)

**用途**: 架构创新前的参照基线。创新对比指标 = ① 同训练数据量下 loss 收敛速度; ② accept_len(投机解码核心)。

## 训练配置
- verifier: `/home/y50063564/Qwen3-4B`(32 Q 头 / 8 KV 头 / head_dim 128)
- draft: DSpark, 5 层, block_size=8, draft_vocab=32000, target_layers=[1,9,17,25,33], sliding_window 2048
- 数据: `open_perfectblend_qwen3_4b_70w`(700k 样本, ~498M token)
- 规模: 185 服务器 **16 卡** — dp=12 训练(卡 0-11)+ vLLM dp=4(卡 12-15)
- 每 rank 步数/epoch ≈ 9,410; 3 epoch 共 28,190 步; total_seq_len=4096
- loss 构成: `0.1·ce + 0.9·tv + 1.0·confidence`

## 每 epoch 验证指标

| epoch | val accept_len | val accept_rate | val full_acc | val loss* |
|---|---|---|---|---|
| 0 | 4.277 | 0.654 | 0.676 | 0.003 |
| 1 | 4.625 | 0.697 | 0.718 | 0.003 |
| 2 | **4.758** | 0.713 | 0.735 | 0.003 |

> *`val/loss` 是衰减伪影(恒 ~0.003, 不反映收敛), **收敛对比请用 `train/loss` 或 `val accept_len`**。

## 每 epoch 每 pos 接受率(重点数据)

| epoch | pos1 | pos2 | pos3 | pos4 | pos5 | pos6 | pos7 |
|---|---|---|---|---|---|---|---|
| 0 | 0.813 | 0.756 | 0.709 | 0.668 | 0.630 | 0.595 | 0.560 |
| 1 | 0.841 | 0.791 | 0.749 | 0.712 | 0.677 | 0.644 | 0.611 |
| 2 | **0.852** | **0.805** | **0.765** | **0.728** | **0.696** | **0.664** | **0.632** |

train 每 epoch 每 pos 均值(供收敛曲线参考):

| epoch | pos1 | pos2 | pos3 | pos4 | pos5 | pos6 | pos7 |
|---|---|---|---|---|---|---|---|
| 0 | 0.758 | 0.692 | 0.641 | 0.597 | 0.559 | 0.525 | 0.491 |
| 1 | 0.829 | 0.776 | 0.732 | 0.692 | 0.657 | 0.623 | 0.589 |
| 2 | 0.852 | 0.804 | 0.763 | 0.727 | 0.694 | 0.661 | 0.628 |

(数值同 `baseline_per_pos.csv`。)

## 收敛速度参考点(给对比用)

| 指标 | step 0 | ~4,700 步 | ~9,410 步(ep0末) | ~18,821 步(ep1末) | 28,190 步(ep2末) |
|---|---|---|---|---|---|
| train accept_len | 1.00 | ~4.0 | ~4.4 | ~4.76 | ~4.75 |
| train loss | 0.529 | ~0.19 | ~0.15 | ~0.131 | ~0.128 |
| full_acc(训练均值/epoch) | — | — | 0.609 | 0.700 | 0.733 |

## 文件说明
- `train_logs.log` — 原始训练日志(185, 16 卡, 3 epoch)
- `train_metrics.csv` — 28,190 步 train 指标
- `val_metrics.csv` — 3 个 epoch 验证指标
- `baseline_per_pos.csv` — 每 epoch 每 pos 接受率(val + train 均值)
- `loss_curve.png` / `accept_len_curve.png` / `epoch_summary.png` — 曲线
- `reconstruct_metrics.py` — 还原脚本(重跑: `python3 reconstruct_metrics.py train_logs.log -o .`)
- `compare/` — 与本地 4:1 训练的同 step 对比
