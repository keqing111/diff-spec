# 20260911_attn_gqa_5w

**训练特性**：draft 注意力 = Qwen3-4B 原生 GQA (32Q/8KV/head_dim 128), 5 层全 full attention

| 项 | 值 |
|---|---|
| 训练时间 | 2026-09-11 13:34 -> 17:38 |
| 数据 | `/home/y50063564/data/open_perfectblend_qwen3_4b_50k  (5w, train ratio 0.9 -> 45k doc/epoch)` |
| 启动方式 | `bash train_attn_run.sh gqa 11` |
| 训练脚本 | `train_attn_run.sh`（本目录内） |
| 原始 run 目录 | `/home/y50063564/processed_data/dspark_data/dspark_attn/gqa` |
| checkpoint（**未搬运**） | `/home/y50063564/processed_data/dspark_data/dspark_attn/gqa/checkpoints/` |
| train 记录条数 | 27362 |

## 验证指标（每 epoch）

| epoch | val accept_len | val accept_rate | val loss | val full_acc |
|---|---|---|---|---|
| 0 | 2.8194 | 0.4663 | 0.63747 | 0.4980 |
| 1 | 3.3921 | 0.5454 | 0.54620 | 0.5781 |
| 2 | 3.6601 | 0.5805 | 0.50685 | 0.6127 |

## 训练指标（整 epoch 全微步均值）

| epoch | train accept_len | train loss |
|---|---|---|
| 0 | 2.2262 | 0.74743 |
| 1 | 3.1504 | 0.58148 |
| 2 | 3.6743 | 0.50094 |

## 文件

| 路径 | 内容 |
|---|---|
| `logs/` | 训练日志原始文件（多行 rich 格式） |
| `smoke/` | 上线前的 dry-run 与 30 微步冒烟日志 |
| `metrics/train_metrics.csv` | 从日志抽出的逐步 `train/loss`、`train/accept_len`、`train/accept_rate` |
| `metrics/val_metrics.csv` | 每 epoch 验证指标 |
| `config.txt` | 启动时记录的 arm / 卡 / 日志路径 / git HEAD |
| `diff.patch` | 启动时工作区的 git diff（含 grad-accum、插桩、注意力改动） |

> 权重不在此目录；见上表 checkpoint 路径。
