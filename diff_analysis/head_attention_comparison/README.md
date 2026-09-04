# Head-level 注意力对比: l0_5l vs baseline

**目的**：在完全相同的条件下（同一样本、同一 anchor、同一 query 位置），对比
l0_5l（diff@layer0 模型，3-epoch 最优权重）和 baseline（GQA，12卡，最优权重）
每层每个注意力头的注意力分布。

## 生成方式

- 样本：Japan/Tokyo 文本（71 token）
- 一个进程先后跑两个模型，每次 forward 前 `torch.manual_seed(0)` → anchor 一致
- query = 每个模型最后一个 block 的末尾位置 q63
- 每张图：x = 完整 KV（0-70 base token + 71-134 合成 block），灰色虚线 = base/block 分界，
  红色虚线 = Tokyo 位置
- 脚本：`../scripts/plot_head_attention_comparison.py`

## 文件

- `l0_5l/layer{N}_heads{H}_q63.png` — l0_5l 第 N 层，H 个头
  - layer0: H=16（**diff 层**，16 个 pair 相减后的注意力）
  - layer1-4: H=32（GQA）
- `baseline/layer{N}_heads32_q63.png` — baseline 第 N 层，32 个头（全 GQA）

## 对比要点

- **l0_5l layer0（diff）vs baseline layer0（GQA）**：看 diff 的 16 个 pair 注意力形态
  是否和 GQA 的 32 头不同、是否聚焦到关键词
- **层间演化**：两模型的 layer1→4 注意力如何逐层从"位置0(The)主导"转向"关键词聚焦"
- 配套数值统计见退化分析报告（`../reports/`）
