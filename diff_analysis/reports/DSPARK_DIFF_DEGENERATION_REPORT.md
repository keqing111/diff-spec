# DSPark Diff-Transformer 层注意力退化分析报告

- 日期: 2026-09-03
- 模型: DSPark 草稿模型 (DFlash 骨架 + Markov/confidence 头), verifier = Qwen3-4B
- 改动: 将草稿模型部分层 (默认 layer0/1) 的 GQA 注意力替换为 Diff-Transformer (diffv1) 注意力
- 分析对象: `dspark_diff_l0_5l/checkpoints/checkpoint_best` (epoch 0 末, global_step 127728)

---

## 1. 背景: 什么是 diff 注意力退化

diffv1 注意力把注意力头两两配成 pair, 每 pair 两个分支 (branch0/branch1) 各自对 key 集做一次
softmax, 然后做差 `attn = branch0 - λ·branch2`, λ 由 4 个可训练向量经 `exp(Σq1·k1) - exp(Σq2·k2)
+ λ_init` 得出。论文设想: 两个分支都关注**同一段上下文**, 用不同 Q/K 投影使它们侧重不同 token,
相减后**抵消公共噪声、放大有用 token**(差分注意力)。

DFlash 的注意力结构里, 每个 anchor block 的 query 可以关注:
1. base 上下文 (anchor 之前, 同 doc, 滑动窗口内)
2. 自己的**合成 block** (mask token, 即 KV 里 seq 之后自己的那一块)

KV = `[base tokens | 所有 anchor 的合成 block]`。

**退化现象**: 观测到 diff 层的大量 pair 出现"一个分支 ~98% 注意力落在自己的合成 block 上、
几乎不看 base 上下文; 另一个分支才看 base"。这偏离了论文设计——差分没有在上下文上发生, 而变成了
"一个分支看上下文、另一个看本地合成块"。

---

## 2. 证据 1: 多样本统计 (40 个真实训练样本, 138,624 个 (query,pair) 实例)

数据来源: 从 `open_perfectblend_qwen3_4b_700k` 取 40 个真实样本, 每个跑 verifier+DSpark forward,
统计每个 (query,pair) 的 branch0/branch1 注意力落在 base(上下文) vs mask(合成 block) 区间的比例。
判定 (阈值 base 占比 > 0.1):
- `both_base`: 两分支都在上下文上
- `one_to_mask`: 一个分支看 base, 另一个 ~98% 看合成 block
- `both_mask`: 两分支都不看上下文

| 模式 | 占比 |
|---|---|
| both_base (两分支都在上下文做差) | **44.4%** |
| one_to_mask (一个分支 ~不看上下文) | **54.4%** |
| both_mask | 1.1% |

**按 pair 统计** (显示这是固定结构而非随机):

| pair | both_base% | one_to_mask% |
|---|---|---|
| **4, 8, 15** | **~99%** | ~0.4% |
| 7, 5, 10 | 75% / 70% / 63% | 24~41% |
| 1, 13, 9 | 19~27% | 72~80% |
| **0, 2, 3, 11, 12, 14** | **~12-19%** | **~82-88%** |

**结论**: ~3 个 pair (4,8,15) 几乎永远双分支都在上下文; ~6 个 pair (0,2,3,11,12,14) 几乎总是
一个分支逃到合成 block 上。这是一个**稳定学到的分工**, 不是随机。

---

## 3. 证据 2: 单样本逐 pair 的 branch 落点 (France 样本, 最后 query q63)

样例: "The capital of France is Paris. ... The capital of France is", 答案 token `ĠParis` 在位置 5。

最后 query q63 每个 pair 的 branch0/branch1 在 base vs 自己 block 的占比 (从
`layer0_diff_branches.pt` 提取):

```
pair | b0base% b0block% | b1base% b1block% | 模式
   0 |   0.8%  99.2% |  88.2%  11.8% | b1->base,b0->block
   1 |   0.1%  99.9% |  94.7%   5.3% | b1->base,b0->block
   2 |  94.1%   5.9% |   0.7%  99.3% | b0->base,b1->block
   3 |  77.4%  22.6% |   0.1%  99.9% | b0->base,b1->block
   4 |  93.3%   6.7% |  89.2%  10.8% | both-base
   5 |   5.7%  94.3% |  92.3%   7.7% | b1->base,b0->block
   6 |  90.0%  10.0% |   2.0%  98.0% | b0->base,b1->block
   7 |  73.9%  26.1% |   4.7%  95.3% | b0->base,b1->block
   8 |  25.4%  74.6% |  86.5%  13.5% | both-base
   9 |   0.2%  99.8% |  33.7%  66.3% | b1->base,b0->block
  10 |  85.6%  14.4% |   0.9%  99.1% | b0->base,b1->block
  11 |  88.3%  11.7% |   0.3%  99.7% | b0->base,b1->block
  12 |   1.9%  98.1% |  88.3%  11.7% | b1->base,b0->block
  13 |   1.0%  99.0% |  85.1%  14.9% | b1->base,b0->block
  14 |  66.2%  33.8% |   1.6%  98.4% | b0->base,b1->block
  15 |  86.9%  13.1% |  77.0%  23.0% | both-base
```

---

## 4. λ 值与逐层答案注意力

**λ 没有退化** (checkpoint_best, layer0):
```
exp(sum(q1k1)) = 1.206   exp(sum(q2k2)) = 0.600   λ_init = 0.200
λ_full = 1.206 - 0.600 + 0.200 = 0.806
```

**逐层对答案 token 的注意力** (France 样本, "Paris", 最后 block queries 平均, 占 base 注意力总和的比):
| 层 | 答案注意力占比 |
|---|---|
| layer0 (diff) | 5.0% |
| layer1 (GQA) | 1.8% |
| layer2 (GQA) | 1.8% |
| layer3 (GQA) | 2.3% |
| layer4 (GQA) | 3.1% |

另观测到 (Japan 样本): layer4 的末尾 query 中, "Tokyo" 成为注意力第 1 (权重 0.282),
而 layer0 (diff) 注意力被序列开头的 "The" 主导 (权重 0.742), Tokyo 不进前 10。
**即关键词聚焦仍主要出现在深层 (layer4), diff@layer0 没有显著提升底层的答案筛选能力。**

---

## 5. 相关图 (位于本目录)

### diff 层逐 pair 分支曲线
- `layer0_diff_pairs_lastquery.png` — 16 pair × 最后 query 的 branch0(蓝)/branch1(橙)/相减(绿) 曲线
- `layer0_diff_branch0_lastblock_avg.png` / `layer0_diff_branch1_lastblock_avg.png` — branch0/branch1 在最后 block 的 16pair×KV 热力图
- `layer0_diff_subtracted_lastblock_avg.png` — 相减后的热力图

### 逐层最后 block 注意力热力图 (8 query × KV, 按 head 平均 / 逐 head)
- `layer0_lastblock_avgattn.png` ... `layer4_lastblock_avgattn.png`
- `layer0_lastblock_perhead.png` ... `layer4_lastblock_perhead.png`

### 多样本逐 token 注意力 (block 内多个位置)
- `few_tokens/samp1_q56.png` `samp1_q57.png` `samp1_q60.png` `samp1_q63.png` (France 样本)
- `few_tokens/samp2_q40.png` `samp2_q41.png` `samp2_q44.png` `samp2_q47.png` (Water 样本)
- `few_tokens/samp3_q40.png` `samp3_q41.png` `samp3_q44.png` `samp3_q47.png` (Elephants 样本)

---

## 6. 附加发现: context-only 导致的 NaN

为消除"分支逃到 block"的路径, 试验了 context-only diff 注意力 (diff 层只能看 base 上下文)。
发现 NaN: 打包序列里 **doc 边界/开头的 anchor**, 其"本 doc 内 anchor 之前"的 base token 为空,
query 无任何合法注意力目标 → softmax 全 -inf → NaN → 梯度 NaN → 权重污染。
修复: context-only 时允许 query 关注自己的 anchor base token (`kv_base_pos <= q_anchor`), 保证
每个 query ≥1 个目标。

---

## 7. 结论与开放问题

1. **diff 层存在明显退化**: 约 6 个 pair (0,2,3,11,12,14) 稳定地一个分支逃到合成 block,
  差分能力只在少数 pair (4,8,15) 的上下文上发挥。λ 值正常 (0.806)。
2. **关键词聚焦仍在深层 (layer4)**: diff@layer0 未显著提升底层答案筛选。
3. 当前正在跑的三组对照 (都 5 层 diff@layer0):
   - `dspark_diff_l0_5l` (旧代码, K 共享)
   - `dspark_diff_l0_5l_kfix` (K 独立)
   - `dspark_diff_l0_5l_ctxonly` (context-only, 修复 NaN 后)
4. **开放问题**: 分支的 block/base 分工是有害退化还是有用分工? diff 是否值得保留 pair 做差结构,
  还是退化为"加权混合 block+ctx"的常规注意力?
