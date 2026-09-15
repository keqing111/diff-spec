# 20260910-12_ctx_only_layer0_70w

**训练特性**：draft attention ctx-only @layer0 (纯 GQA, 无 diff 分支), dp4 + accum3 (= dp12 等效), log-freq 1

| 项 | 值 |
|---|---|
| 训练时间 | 2026-09-10 01:59 -> 2026-09-12 06:56  (中途暂停在 gs=28910, 09-11 17:39 续训完 ep1-2) |
| 数据 | `/home/y50063564/data/open_perfectblend_qwen3_4b_700k  (70w, train ratio 0.9 -> 630k doc/epoch)` |
| 启动方式 | `bash train_dp4_accum3_ctx70w.sh   (卡 11-14, dp4)` |
| 训练脚本 | `train_dp4_accum3_ctx70w.sh`（本目录内） |
| 原始 run 目录 | `/home/y50063564/processed_data/dspark_data/dspark_accum/dp4_accum3_ctx70w` |
| checkpoint（**未搬运**） | `/home/y50063564/processed_data/dspark_data/dspark_accum/dp4_accum3_ctx70w/checkpoints/` |
| train 记录条数 | 86723 |

## 验证指标（每 epoch）

| epoch | val accept_len | val accept_rate | val loss | val full_acc |
|---|---|---|---|---|
| 0 | 4.6454 | 0.6908 | 0.02350 | 0.7113 |
| 1 | 4.9709 | 0.7301 | 0.02067 | 0.7496 |
| 2 | 5.1096 | 0.7455 | 0.01944 | 0.7654 |

## 训练指标（整 epoch 全微步均值）

| epoch | train accept_len | train loss |
|---|---|---|
| 0 | 4.1548 | 0.17875 |
| 1 | 4.8228 | 0.14017 |
| 2 | 5.0831 | 0.12562 |

## 文件

| 路径 | 内容 |
|---|---|
| `train_dp4_accum3_ctx70w.sh` | 训练脚本（**唯一权威的配置来源**，见下） |
| `logs/` | 训练日志原始文件（多行 rich 格式），epoch0 由 3 个日志拼接 |
| `metrics/train_metrics.csv` | 从日志抽出的逐步 `train/loss`、`train/accept_len`、`train/accept_rate` |
| `metrics/val_metrics.csv` | 每 epoch 验证指标 |

**注意：本组没有 `config.txt` 和 `diff.patch`。** 该 launcher 早于「启动时记录 git HEAD +
工作区 diff」的约定，原始 run 目录里也确实只有 `checkpoints/` 和 `logs/`：
```
$ ls .../dspark_accum/dp4_accum3_ctx70w/
checkpoints  logs
```
影响与替代来源：
- **训练超参**可从 `train_dp4_accum3_ctx70w.sh` 完整读出（`--grad-accum 3`、`--epochs 3`、`--lr 6e-4`、
  `--num-workers 6`、`--gqa-context-only-layer-indices 0`、`--total-seq-len 4096`……），该脚本即当时所用。
- **当时的代码状态没有快照。** 该组用到 `--grad-accum` 与 `--gqa-context-only-layer-indices` 两处未提交改动，
  都在 `dspark_project/speculators` 工作区；本组之后（2026-09-11）又加入了注意力三变体改动，所以
  **当前工作区 diff 并不等于本组启动时的状态**，不要拿它当本组的代码快照。
- 插桩/采集代码的历史快照见 `/home/y50063564/dataloader_diag/snapshot/`（2026-09-08 前后，仅覆盖
  dataloader 根因分析那批改动，同样**不是**本组启动态的精确快照）。

> 权重不在此目录；见上表 checkpoint 路径。
