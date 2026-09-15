# 20260911_attn_mha_5w

**训练特性**：draft 注意力 = MHA (KV 头 8->32, head_dim 128, Q 头 32 不变), 5 层全 full attention

| 项 | 值 |
|---|---|
| 训练时间 | 2026-09-11 13:34 -> 17:38 |
| 数据 | `/home/y50063564/data/open_perfectblend_qwen3_4b_50k  (5w, train ratio 0.9 -> 45k doc/epoch)` |
| 启动方式 | `bash train_attn_run.sh mha 12` |
| 训练脚本 | `train_attn_run.sh`（本目录内） |
| 原始 run 目录 | `/home/y50063564/processed_data/dspark_data/dspark_attn/mha` |
| checkpoint（**未搬运**） | `/home/y50063564/processed_data/dspark_data/dspark_attn/mha/checkpoints/` |
| train 记录条数 | 27362 |

## 验证指标（每 epoch）

| epoch | val accept_len | val accept_rate | val loss | val full_acc |
|---|---|---|---|---|
| 0 | 2.7585 | 0.4577 | 0.64628 | 0.4894 |
| 1 | 3.3519 | 0.5412 | 0.55124 | 0.5738 |
| 2 | 3.6175 | 0.5765 | 0.51067 | 0.6087 |

## 训练指标（整 epoch 全微步均值）

| epoch | train accept_len | train loss |
|---|---|---|
| 0 | 2.1964 | 0.75262 |
| 1 | 3.0978 | 0.58819 |
| 2 | 3.6288 | 0.50633 |

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
