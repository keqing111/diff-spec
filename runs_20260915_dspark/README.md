# 20260915 dspark runs — 归档与总结

本次整理收录 4 组训练：**3 组不同 draft 注意力的 5w 训练** + **1 组 70w 的 ctx-only 训练**。
每组的训练脚本、原始日志、逐步指标 CSV、逐 epoch 验证指标、启动时的 git diff 都在各自的子目录里。

**模型权重（checkpoint）没有搬进本目录**（合计 ≈38G），各自保留在原机器上的 run 目录，路径见每个子目录的 README。

## 日志压缩（重要）

本目录里**所有 `.log` 都已 gzip**（`.log.gz`），原始未压缩总 471MB → 现在 32MB。
无单文件超过 50MB，所以**没有用 split 切块**，不需要拼接。

读取方式：

```bash
zcat 20260911_attn_gqa_5w/logs/train_attn_gqa_a12_20260911_133414.log.gz | less
zcat shared/logs/vllm_20260911_130537.log.gz | grep -c "GET /metrics"
# 或一次性全部还原：
find . -name '*.log.gz' -exec gunzip -k {} \;
```

`metrics/*.csv`、`compare/*.csv`、`diff.patch`、`config.txt` 都是明文，未压缩。

## 一、三组不同注意力（5w）

数据 `open_perfectblend_qwen3_4b_50k`（45k train doc/epoch × 3 epoch），单卡、`--grad-accum 12`、
`--full-attention-indices 0 1 2 3 4`（5 层全 full attention，对齐 Qwen3-4B 原生 pattern），
**除注意力类型外其余参数完全一致**（三组共用同一份 `scripts/train_attn_run.sh`，只差一个 flag）。

| 组 | KV 头 | K/V 投影 | 每层 attn 参数 | draft 总参数 |
|---|---|---|---|---|
| MHA | 32 | `k_proj(4096,2560)`,`v_proj(4096,2560)` | 41,943,296 | 1,134,220,929 |
| **GQA(原生)** | 8 | `k_proj(1024,2560)`,`v_proj(1024,2560)` | 26,214,656 | 1,055,577,729 |
| MLA r=512 | 8 | `kv_down(512,2560)`→`k_up/v_up(1024,512)` | 23,331,072 | 1,041,159,809 |

三个 arm 都保留 Qwen3 的 `q_norm`/`k_norm`（per-head RMSNorm，head_dim=128）——这也正好把 MLA 低秩投影
带来的 K/V 尺度差归一化掉，使 MLA 相对 GQA 的差异只剩「低秩瓶颈」这一个变量。

### 验证结果

| epoch | MHA | GQA(原生) | MLA r=512 |
|---|---|---|---|
| 0 | 2.7585 | 2.8194 | 2.8807 |
| 1 | 3.3519 | 3.3921 | 3.4284 |
| 2 | 3.6175 | 3.6601 | 3.6717 |

### 结论

**没有证据表明换注意力架构能提升接受长度。** epoch2 的 val accept_len 为
MHA 3.6175 / GQA 3.6601 / MLA 3.6717，**跨度仅 1.5%**；而这个平台**同配置重跑不可逐位复现**
（实测同配置两次 |Δloss| = 0.0165），1.5% 落在噪声量级内。
另需注意三组的可训练参数量本就不同（MHA +78.6M vs GQA，MLA −14.4M），
所以这也不是一个纯粹「机制」层面的对照。**单 seed、无重跑**，结论仅到此为止。

- 日志：`20260911_attn_mha_5w/logs/`、`20260911_attn_gqa_5w/logs/`、`20260911_attn_mla_5w/logs/`
- 注意力几何/参数离线对拍：`scripts/attn_config_check.py`（可复跑）
- 权重形状核对：`scripts/check_ckpt_attn.py`

## 二、70w 的 ctx-only（layer0）

`20260910-12_ctx_only_layer0_70w/`：数据 `open_perfectblend_qwen3_4b_700k`（630k train doc/epoch × 3 epoch），
dp4 + `--grad-accum 3`（每个优化步 12 个 pack，与 dp12 等效），`--gqa-context-only-layer-indices 0`，
卡 11-14。中途暂停在 `global_step=28910`，2026-09-11 17:39 续训完 epoch1-2。

### 与 dp12 baseline 的对比（同数据量：均为 630k doc/epoch，均为 sliding_window 2048）

| epoch | dp12 baseline val accept_len | ctx-only val accept_len | Δ |
|---|---|---|---|
| 0 | 4.2770 | 4.6454 | +0.3684 |
| 1 | 4.6250 | 4.9709 | +0.3459 |
| 2 | 4.7580 | **5.1096** | +0.3516 |

train/accept_len 的整 epoch 均值同样 3/3 同号（+0.35~+0.40）。**6/6 个测量点同号且幅度一致**，
这是本批实验里唯一一个幅度清晰超过噪声的效应（相对 +7~8%）。

但要注意口径差异，不要把这条结论外推：

1. **x 轴必须用 epoch 而不是 `global_step`**：两边 `global_step` 语义不同
   （baseline 每 rank 1 微步/优化步 = 9410/epoch；ctx 是 dp4+accum3 的 rank-local 微步 = 9637/epoch），
   两者每优化步都是 12 个 pack、每 epoch 都是 630k doc，所以按 epoch 对齐才是同数据量。
2. **不是同一台机器**：baseline 在 185 的 16 卡（dp12 训练 + srv dp4），ctx 在本机 dp4+accum3。
3. **`train/loss` 通道单独看不显著**：Δloss 仅 0.002~0.003，小于同配置重跑噪声 0.0165；
   信号在接受率上，不在复合 loss 上。
4. **单 seed、无重跑**。
5. ctx 的 epoch0 曲线由 4 个日志拼接（中途换过 worker/server 配置，sampler seed 未变），按
   `global_step` 去重合并，曲线有接缝。

对照数据：`compare/dp12_baseline_reference/`（baseline 的 `BASELINE.md` / `train_metrics.csv` / `val_metrics.csv`），
原始日志在 `archive_dspark_20260904/eval/baseline_qwen3_4b_70w/train_logs.log`（未搬运）。

## 三、目录说明

| 路径 | 内容 |
|---|---|
| `scripts/` | 本次用到的全部启动/分析/绘图脚本 |
| `shared/logs/` | 服务全部 4 组训练的同一个 vLLM hidden-states server 日志（卡15/端口1123，max-num-seqs 24） |
| `compare/` | 跨组对比图与 CSV（`ctx70w_vs_dp12.png` 等） |
| `<run>/` | 每组一个目录：训练脚本 + logs + metrics + README（三组注意力另有 `config.txt`/`diff.patch`/`smoke/`） |

## 四、复现命令

```bash
# 三组注意力（server 先起在卡15）
bash scripts/serve_vllm_attn.sh
bash scripts/train_attn_run.sh gqa 11
bash scripts/train_attn_run.sh mha 12
bash scripts/train_attn_run.sh mla 13

# 70w ctx-only（dp4，卡 11-14）
bash scripts/train_dp4_accum3_ctx70w.sh

# 对比图
python3 scripts/ctx70w_vs_dp12.py
```

> 注意：源码改动（`--grad-accum`、`--draft-attention-type`、`--mla-kv-lora-rank`、
> 新增 `Qwen3DFlashMLAttention`）都在 `dspark_project/speculators` 工作区，**未提交**。
>
> `config.txt` / `diff.patch` **只有三组注意力有** —— 那是 2026-09-11 我开始用
> `train_attn_run.sh` 记录 git HEAD 与工作区 diff 之后。**70w 那组启动更早，没有留下代码快照**
> （原始 run 目录里只有 `checkpoints/` 和 `logs/`），已在其 README 中写明。
> 不要用当前工作区 diff 反推它启动时的代码状态：60 小时后加入的注意力三变体改动也在当前 diff 里。
