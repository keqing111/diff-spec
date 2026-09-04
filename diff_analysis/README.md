# DSPark Diff-Transformer 实验分析目录

统一存放 diff 注意力实验的**图、数据、报告、脚本**。命名规范见文末。

## 目录说明

| 目录 | 内容 |
|---|---|
| `l0_5l_diff_attention/` | **l0_5l（5层 diff@layer0，旧代码）** 的注意力图：每层 last-block 热力图(`layer{N}_lastblock_*`)、每层逐头(`layer{N}_perhead_*`)、diff 层 pair 分支图(`layer0_diff_pairs_lastquery`、`branch0/1/subtracted`)、原始张量(`layer0_diff_branches.pt`) |
| `baseline_attention/` | **baseline（12卡 GQA）** 同一样本/位置的每层逐头注意力图 + `baseline_attn.pt` |
| `per_query_attention/` | 多样本、block 内**逐 query 位置**的 diff 注意力（原 `few_tokens`，`samp{N}_q{Q}.png` = 样本N在 query Q）|
| `training_curve_comparisons/` | 训练曲线对比（loss/accept_len/... 随样本数或 epoch）|
| ├ `4way` / `4way_samples` | 4 组训练对比（baseline+l0_5l+l1_5l+l0_3l），横轴样本数 |
| ├ `l0_5l_12card_vs_baseline` | 12卡 diff@layer4 早期 vs baseline |
| └ `acceptlen_3way_epoch`、`l0_5l_vs_baseline` 等 | 具体两两/三路对比 |
| `raw_data/` | 原始数据（非图） |
| ├ `merged_logs/` | 拼接后的完整训练日志（l0_5l、ctxonly）|
| ├ `baseline_metrics/` | reconstruct_metrics 解析出的 baseline CSV/图 |
| ├ `dspark_16card_12card_metrics/` | 12卡 diff@L4 run 的指标 |
| └ `backups/` | 被覆盖前的重要图备份（如 diff pairs 的 epoch0 版）|
| `reports/` | Markdown 报告（退化分析等）|
| `scripts/` | 生成这些图/数据的分析脚本 |

## 关键实验配置速查

- **baseline**：12 卡 GQA（无 diff），3 epoch，`/home/y50063564/checkpoint_best`
- **l0_5l**：单卡 5 层 diff@layer0，旧代码（K 共享），3 epoch，`dspark_diff_l0_5l/`
- **kfix**：单卡 5 层 diff@layer0，当前代码（K 独立），训练中，`dspark_diff_l0_5l_kfix/`
- **ctxonly**：单卡 5 层 diff@layer0 + 只看上下文，训练中，`dspark_diff_l0_5l_ctxonly/`
- 训练运行输出目录在 `/home/y50063564/processed_data/dspark_data/dspark_diff_*`（**不属于**本分析目录）

## 命名规范（今后遵守）

- 模型配置用全称：`l0_5l`=5层diff在layer0；`kfix`=K独立；`ctxonly`=只看上下文；`baseline`=GQA无diff
- 比较类目录用 `X_vs_Y` + 明确横轴（`_epoch`/`_samples`）
- 图名 = `{内容}_{样本/query}_{视图}`，如 `baseline_layer3_perhead_q63`
- 不再用 `few_tokens`、`4way` 这类无意义名
