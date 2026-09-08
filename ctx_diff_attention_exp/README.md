# ctx diff-Transformer 注意力 / 提取能力实验

对象:**ctxonly / DSpark diff@layer0 + context-only(K独立,新码)** 的最终 checkpoint
(`dspark_diff_l0_5l_ctxonly/checkpoints/2`, epoch2 末)。验证 diff 层注意力是否真的"分化"成功、
以及 ctx 草稿的"从上下文提取关键信息"能力(needle / 多干扰 / 更新)。

## 文件
- `ctx_harness.py` — 加载 ctx checkpoint(强制末尾锚点、抓 diff 层 b0/b1 与各层 attention)与共享度量。
- `exp1_needle_ctx.py` — needle(Velora/Alpha): 长度 128/512/2048 × 位置 10/50/90%,
  纵向逐层关键token占比(L0 用 b0+b1 与 b0−b1)+ P;横向 diff 层每 pair;答案槽 j=1..7 扫描。
- `exp2_distractors_ctx.py` — 语义相似干扰 0/1/2/4/8/16/32,多样本;target×draft 全 2x2 记录 +
  L3+L4 把真关键句排第1 的定位命中率。
- `exp3_update_ctx.py` — 旧码→更新→新码(单token 码),变体 was_changed/is_now × 问新/旧;全样本记录。
- `plot_report.py` / `summarize.py` — 汇总画图/打印。
- `results_exp{1,2,3}/` — 结果 json(含逐样本原始记录)。

## 运行(需 NPU + 本机路径)
```bash
ASCEND_RT_VISIBLE_DEVICES=11 python3 exp1_needle_ctx.py -o results_exp1
ASCEND_RT_VISIBLE_DEVICES=12 python3 exp2_distractors_ctx.py -o results_exp2
ASCEND_RT_VISIBLE_DEVICES=12 python3 exp3_update_ctx.py -o results_exp3
python3 plot_report.py; python3 summarize.py
```

## 要点结论
- 提取鲁棒: 2048 长度下 P≈0.93–0.96;真正聚焦关键 token 的是深层 GQA 层(L3/L4);
  diff 层 L0 对关键 token 注意力很低、b0−b1≈0 —— 以"答案槽指向关键token"衡量,diff 分化不明显。
- 多干扰 0→32: ctx 定位命中率 100%、P(真码)≈0.88–0.95(远强于旧 GQA draft 的 n≥4 崩溃)。
- 更新场景: 问新码时 target(verifier)基本答不对(0/16),而 ctx draft 反而答对 9–15/16("target 错 draft 对"为主);
  问旧码两者都对。target 能答对"新码"的样本仍需更友好 prompt 才能批量构造。
