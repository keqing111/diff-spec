# 20260911_attn_mla_5w

**训练特性**：draft 注意力 = 轻量 MLA (KV 经 rank-512 共享 latent; 保留 q_norm/k_norm), 5 层全 full attention

| 项 | 值 |
|---|---|
| 训练时间 | 2026-09-11 13:34 -> 17:38 |
| 数据 | `/home/y50063564/data/open_perfectblend_qwen3_4b_50k  (5w, train ratio 0.9 -> 45k doc/epoch)` |
| 启动方式 | `bash train_attn_run.sh mla 13` |
| 训练脚本 | `train_attn_run.sh`（本目录内） |
| 原始 run 目录 | `/home/y50063564/processed_data/dspark_data/dspark_attn/mla` |
| checkpoint（**未搬运**） | `/home/y50063564/processed_data/dspark_data/dspark_attn/mla/checkpoints/` |
| train 记录条数 | 27362 |

## 验证指标（每 epoch）

| epoch | val accept_len | val accept_rate | val loss | val full_acc |
|---|---|---|---|---|
| 0 | 2.8807 | 0.4743 | 0.62927 | 0.5063 |
| 1 | 3.4284 | 0.5499 | 0.54119 | 0.5824 |
| 2 | 3.6717 | 0.5822 | 0.50453 | 0.6148 |

## 训练指标（整 epoch 全微步均值）

| epoch | train accept_len | train loss |
|---|---|---|
| 0 | 2.2751 | 0.73983 |
| 1 | 3.2021 | 0.57407 |
| 2 | 3.6995 | 0.49790 |

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
