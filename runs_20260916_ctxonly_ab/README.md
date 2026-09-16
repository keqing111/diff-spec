# 20260916 — ctx-only 的 anchor 信息泄漏：修复对照实验

**这是一次证伪实验。它推翻了 `runs_20260915_dspark/` 里关于 ctx-only 提升接受长度的结论。**

日期：2026-09-15 实施、2026-09-16 整理
相关代码分支：`ctxonly-anchor-fix-20260915`（基于 `snapshot-20260915`，`containing` 本次修复）

---

## 1. 问题

`ctx-only`（`--gqa-context-only-layer-indices` / `--diff-attention-context-only`）训练时，
query 被限制为"只看 base 上下文，不看自己合成的 anchor block"。

但掩码里为了「保证每个 query 至少有一个合法目标，避免 softmax 全 `-inf` 出 NaN」，
把可见 base 的上界从 `< anchor` 放宽成了 **`<= anchor`**：

```python
# src/speculators/models/dflash/attention.py（修复前）
before_anchor = (
    kv_base_pos <= q_anchor if not include_same_block else kv_base_pos < q_anchor
)
#                ^^^^^^^^^^ ctx-only 时额外放进了 anchor 自己
```

于是训练期 query 能 attend 到 **anchor 位置自己的 K/V**，而那是 verifier 在 anchor token 上的
hidden state。

**推理期拿不到这份信息。** vLLM 侧：

```python
# vllm/v1/spec_decode/dflash.py: set_inputs_first_pass
#   DFlash cross-attention: context K/V from target hidden states,
#                           Q from query embeddings (bonus + mask tokens)
# vllm/v1/spec_decode/utils.py: copy_and_expand_dflash_inputs_kernel
#   2. Computes query positions (last_target_pos + 1 + offset)
last_pos  = target_positions[valid_ctx_end - 1]   # 最后一个被 target 处理过的位置
query_pos = last_pos + 1 + query_off              # bonus 在 last_pos+1
```

draft 的 context K/V 只覆盖 target **已经处理过**的位置，即到 `anchor - 1`；anchor/bonus
位置此刻还没有 hidden state。所以 `<= anchor` 是**训练期独有、推理期不存在**的信号泄漏。

## 2. 修复

- `attention.py` 新增 `anchor_in_context`，默认 `False` = `< anchor`（与推理一致）。
- NaN 不再靠泄漏来兜：`utils.py` 的 `select_anchors` 新增
  `document_ids` / `require_context_predecessor`，**剔除「没有同文档前驱」的 anchor**
  （文档首 token）——这类 anchor 在 `<` 下可见集为空，必须排除而不是放行。
  正常情况只损失 **2/3992 = 0.05%** 的候选位置。
- 旧行为保留为 `--ctx-only-anchor-in-context`，供本对照实验复现。

完整代码改动见 `ctxonly_anchor_fix.patch`（5 文件，+93/−4），掩码单测见
`ctxonly_mask_unit_test.py`。

## 3. 实验设计

同一份数据、同一超参，**唯一变量就是那个 flag**：

| arm | flag | 含义 |
|---|---|---|
| `fix` | 无 | 修复后掩码（与推理一致） |
| `legacy` | `--ctx-only-anchor-in-context` | 复现旧的泄漏行为 |

- 数据：`open_perfectblend_qwen3_4b_50k`（5w，train ratio 0.9 → 45k doc/epoch）
- 3 epoch、`--grad-accum 12`、单卡、`--gqa-context-only-layer-indices 0`、sliding window 2048
- 逐微步 `--log-freq 1`，两个 arm 同时跑（卡 11 / 卡 12），共用卡15 的 vLLM hidden-states server
- 启动：`bash scripts/train_ctxonly_ab_run.sh {fix|legacy} <card>`

`legacy` 是**内置的框架可信度对照**：它必须复现既有那份老代码跑出来的
`gqa_ctx_l0_a12`（epoch2 = 4.0541），否则说明对照框架与老代码不等价，`fix` 的结果也不能信。

## 4. 结果

val/accept_len_epoch：

| run | ep0 | ep1 | ep2 | val loss (ep2) | accept_rate (ep2) |
|---|---|---|---|---|---|
| **`legacy`**（泄漏） | 3.3085 | 3.8206 | **4.0582** | 0.44368 | 0.6166 |
| **`fix`**（无泄漏） | 2.7739 | 3.3633 | **3.6224** | 0.51180 | 0.5765 |
| plain GQA（历史参照，老代码 09-08） | 2.8319 | 3.4090 | 3.6739 | 0.50503 | — |
| ctx-only 旧（历史参照，老代码 09-08） | 3.3050 | 3.8126 | 4.0541 | 0.44362 | — |

### 4.1 框架可信度：通过

```
legacy 4.0582  vs  老代码 4.0541   Δ = 0.0041   (0.1%)
loss   0.44368 vs  0.44362         Δ = 0.00006
```

几乎逐位复现。⇒ 对照框架与老代码等价，且这个 5w/单卡配置的**跨会话噪声 ≈ 0.1%**。

### 4.2 泄漏是 ctx-only 增益的**全部**来源

同会话、同代码、只差一个 flag：

```
accept_len   4.0582 → 3.6224   = −0.4358
accept_rate  0.6166 → 0.5765   = −0.0401
val loss     0.44368 → 0.51180 = +0.0681（变差）
```

−0.436 是噪声（≈0.004）的 **100 倍**。连 loss 也一起变差，说明泄漏不只是抬高接受率，
是真的在帮模型。

### 4.3 修掉泄漏后，ctx-only 本身没有收益

`fix` (3.6224) vs plain GQA (3.6739)：

| | ep0 | ep1 | ep2 |
|---|---|---|---|
| Δ | −0.0580 | −0.0457 | −0.0515 |

3 个 epoch 全部同号，loss 也一致更差（0.51180 vs 0.50503）。
⇒ 「关掉 layer0 的 block 内自条件化」这个想法**自己赚不到东西，反而小亏一点**。

> ⚠️ 4.3 跨了会话（plain GQA 是 2026-09-08 老代码跑的）。4.1 已证明该配置跨会话可复现性
> 约 0.1%，所以 −0.05 大概率是真的，但严格结论仍需一组同会话的 plain GQA 对照。

## 5. 影响范围

| 之前的结论 | 状态 |
|---|---|
| ctx-only 70w 的 +0.35~0.40（相对 dp12 baseline） | **作废** —— 同一机制，5w 上量化出来是 +0.43，全部来自泄漏 |
| `diff-ctx-only @l0 / @l3` 两组 5w 消融 | **同样受影响**（走同一个 `include_same_block=False` 路径） |
| 三组注意力 MHA / 原生 GQA / MLA 的对比 | 不受影响（那三组用 full attention，未开 ctx-only） |
| `--grad-accum` 语义验证 | 不受影响 |

## 6. 文件

| 路径 | 内容 |
|---|---|
| `AB_RESULT.txt` | watcher 跑完自动生成的对照表 |
| `ctxonly_anchor_fix.patch` | 修复本身的代码 diff（相对 `snapshot-20260915`，5 文件 +93/−4） |
| `ctxonly_mask_unit_test.py` | 掩码单测：普通 / 旧 / 修复三种模式的可见列对照 |
| `scripts/train_ctxonly_ab_run.sh` | 两个 arm 共用的启动脚本（按 arm 参数化） |
| `scripts/ctxonly_ab_watch.sh` | 跑完自动出对照表的 watcher |
| `<arm>/config.txt` | 启动记录（arm、卡、数据、git HEAD） |
| `<arm>/diff.patch` | 启动时工作区的完整 git diff |
| `<arm>/logs/*.log.gz` | 训练日志（157MB 未压缩 → 每个约 3MB） |
| `<arm>/metrics/train_metrics.csv` | 逐微步 loss / accept_len / accept_rate，27362 行 |
| `<arm>/metrics/val_metrics.csv` | 每 epoch 验证指标 |

**不含任何模型权重/checkpoint**（每个 arm 的 checkpoint 约 9.2G，留在原 run 目录）。
日志已 gzip，无单文件超 50MB，未使用 split。

## 7. 复现

```bash
# 代码：speculators 仓库的 ctxonly-anchor-fix-20260915 分支，或对本目录的 patch
# 起 vLLM hidden-states server，然后：
bash scripts/train_ctxonly_ab_run.sh fix    <card>
bash scripts/train_ctxonly_ab_run.sh legacy <card>
```
