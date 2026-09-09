# diff-spec —— DSpark diff-transformer 实验归档（脚本 / 报告 / 图 / 训练日志）

本仓库镜像自容器工作区 `/home/y50063564/processed_data/archive_dspark_20260904/`
（即原来的 `eval/` 与 `dspark_data/diff_analysis/`）。仅收录：

- 脚本 `.py` / `.sh`
- 分析报告 `.md`
- 分析图 `.png` / `.jpg`
- 四类训练日志 `.log`：`l0_5l`、`baseline`、`ctx_only`、`kfix`

不收录模型权重、`.pt` 实验张量、`.csv/.json` 指标等大/过程文件。

创建时间：2026-09-04

---

## 目录结构

```
diff-spec/
├── README.md                # 本文件
├── eval/                    # 对应本地归档 eval/（按原目录镜像，自包含）
│   ├── attention_experiment/        # 注意力分布实验：脚本 + REPORT.md + 图 + prompts
│   ├── baseline_qwen3_4b_70w/       # 16卡 DSpark/Qwen3-4B/70w baseline：还原脚本 + BASELINE.md + 图
│   ├── diff_l0_3l_metrics/          # diff(layer0,3层) 训练指标图
│   ├── diff_l0_5l_metrics/          # diff(layer0,5层) 训练指标图
│   ├── diff_l1_5l_metrics/          # diff(layer1,5层) 训练指标图
│   ├── dspark_reproduce/            # HF DSpark 真实投机解码复现评测脚本
│   └── diff_vs_baseline.py          # 对比画图脚本 + 两张对比图
├── diff_analysis/           # 本地归档 dspark_data/diff_analysis/（分析产出）
│   ├── README.md / reports/         # 分析报告
│   ├── scripts/                     # 分析/绘图脚本
│   ├── baseline_attention/  head_attention_comparison/  l0_5l_diff_attention/
│   ├── per_query_attention/ raw_data/  training_curve_comparisons/
├── dspark_project/          # 训练代码与启动脚本（镜像容器 dspark_project/）
│   ├── speculators/         #   训练/推理代码快照（含未提交 diff 改动，无 git 历史/旧分支）
│   └── script/              #   各 run 启动脚本 *.sh（整目录镜像）
└── logs/                    # 四类 run 的训练日志（>100MB 的按 50MB 分片）
    ├── l0_5l/
    ├── baseline/
    ├── ctx_only/
    └── kfix/
```

---

## 训练日志说明

本地源目录：`/home/y50063564/processed_data/dspark_data/<run>/logs/`。

| run | 状态 | 文件 | 存储 |
|---|---|---|---|
| l0_5l | 已完成 | `train_l0_5l_20260831_085711.log`（~626MB） | 分片 `.00`–`.11` |
| l0_5l | 已完成（重启段） | `train_l0_5l_20260903_064547.log`（~95MB） | 单文件 |
| baseline | 已完成（16卡） | `train_logs.log`（~52MB） | 单文件 |
| ctx_only | 已完成（重启前段） | `train_l0_5l_ctxonly_20260903_013053.log`（~51MB） | 单文件 |
| ctx_only | **进行中**（快照） | `train_l0_5l_ctxonly_20260903_064548.log`（~234MB） | 分片 `.00`–`.04` |
| kfix | 已完成（重启前段） | `train_l0_5l_kfix_20260902_101942.log`（~187MB） | 分片 `.00`–`.03` |
| kfix | **进行中**（快照） | `train_l0_5l_kfix_20260903_064547.log`（~232MB） | 分片 `.00`–`.04` |

> 同一 run 出现多个日志文件是因为训练中途掉线后重启，属同一训练过程的不同段，应合并看待。

### 为什么有“进行中”的日志

`ctx_only`、`kfix` 两个 run 在归档上传时（2026-09-04 早）仍在训练，日志持续增长。
此处保存的是上传时刻的**快照**（读取不影响训练进程）。训练结束后若需最新完整日志，请
对增量部分重新分片上传。

---

## 合并分片（还原完整日志）

GitHub 单文件限制 100MB，因此 >100MB 的日志以 **50MB/片** 存储，文件名为
`<原名>.00, <原名>.01, …`。还原时按字典序合并即可，例如：

```bash
# l0_5l 主日志
cat logs/l0_5l/train_l0_5l_20260831_085711.log.* > train_l0_5l_20260831_085711.log
# ctx_only 进行中段
cat logs/ctx_only/train_l0_5l_ctxonly_20260903_064548.log.* > train_l0_5l_ctxonly_20260903_064548.log
# kfix 两段
cat logs/kfix/train_l0_5l_kfix_20260902_101942.log.* > train_l0_5l_kfix_20260902_101942.log
cat logs/kfix/train_l0_5l_kfix_20260903_064547.log.* > train_l0_5l_kfix_20260903_064547.log
```

（shell 通配 `.*` 会按 `.00 .01 …` 顺序展开，等价于 `cat part 依序拼接`。）

---

## 备注

- 仓库内脚本为自包含归档，其中保留的绝对路径指向**容器本地环境**
  （`/home/y50063564/processed_data/archive_dspark_20260904/…` 及 `…/dspark_data/…`），
  clone 到别处重跑前需按本地实际路径调整。
- 各实验结论速览见 `eval/README.md` 与 `diff_analysis/reports/DSPARK_DIFF_DEGENERATION_REPORT.md`。

---

## dspark_project/ —— 训练代码与启动脚本

镜像自容器 `/home/y50063564/dspark_project/`，2026-09-04 快照。

- **`speculators/`**：训练/推理代码（python 库 `src/speculators` + `scripts/` 等）。
  **快照式并入**：已包含工作区中**未提交的 diff 改动**（`scripts/train.py`、
  `src/speculators/models/dflash/attention.py / config.py / core.py / model_definitions.py` 等）
  与未跟踪文件（如 `.github/workflows/`、`scripts/inspect_attn_layers.py`）。
  **不含 `.git`、旧 remote（`origin = speculators-for-glm52`）与旧分支 `diff-dspark`** ——
  即"不保留以前的分支"，从本仓库起独立管理。
  本地原目录仍被正在运行的训练进程使用，故未移动；此处为只读快照。
- **`script/`**：各实验启动脚本 `*.sh`（`train_*.sh`、`vllm_serve_*.sh`）等，整目录镜像。

---

## 2026-09-09 增量：accum / diff-ctx / gqa-ctx 对比 + 消融（5w 子集）+ 代码改动

- **代码**（`dspark_project/speculators/…`，4 文件增量）：
  - `--grad-accum N`（scripts/train.py + train/trainer.py）：按 micro-batch 累积、`loss/N` 反传取平均后 step，等价 DDP(dp=N) 语义；N=1 行为不变。
  - `--gqa-context-only-layer-indices`（models/dflash config/core）：让纯 GQA 层也用"只看 base 上下文"的 mask（原本只发给 diff 层）。
- **启动/工具脚本**：`dspark_project/script/dspark_accum_exp/`（各 run 启动、CPU 数值对拍 `verify_accum_math.py`、`prep_eval_subsets.py`/`eval_checkpoints.py`（过拟合检查）、interim/final/4-in-1/消融 绘图脚本、watcher）。
- **日志** `logs/{accum1,accum12,diff_l0_ctx_a12,diff_l3_ctx_a12,gqa_ctx_l0_a12}/`：50k×3 epochs（log_freq=20，各 ~2.5MB）。
- **结果** `dspark_accum_results/`：对比图（accum1 vs accum12、diff/gqa-ctx@l0 vs plain、4-in-1、消融@l0）+ `overfit_check.csv/.log`（epoch0-2 × train1000/val/out1000 的 accept_len、tv、draft entropy）。

**要点**：accum12(=dp12 平均语义) 优于 accum1；diff-ctx 与 gqa-ctx@l0 均优于 plain accum12，且差异基本来自 **ctx-only 本身**（层0 上 diff 分支仅 ~1% 边际增益）；overfit 检查未见明显过拟合。70w 的 dp4+accum3(ctx-only@l0，≈dp12) 长跑进行中，结果后补。
