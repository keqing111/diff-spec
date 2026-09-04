# eval — DSpark 训练/评测/实验 工作目录

本目录按"**每个实验/任务自包含一个文件夹(含其脚本、数据、结果)**"组织。

## 目录结构

```
eval/
├── README.md                       # 本索引
├── baseline_qwen3_4b_70w/          # Baseline: DSpark/Qwen3-4B/70w 16卡(dp=12)训练
│   ├── BASELINE.md                 #   摘要 + 每epoch每pos接受率(对比实验参照)
│   ├── reconstruct_metrics.py      #   还原脚本(日志→指标/图)
│   ├── train_logs.log              #   原始训练日志(3 epoch)
│   ├── train_metrics.csv / val_metrics.csv / baseline_per_pos.csv
│   ├── loss_curve.png / accept_len_curve.png / epoch_summary.png
│   └── compare/                    #   与本地 4:1 训练的同step对比
│       ├── compare_local_vs_185.png
│       └── local_4to1/             #   本地 4:1 解析结果
├── dspark_reproduce/               # HF DSpark 权重复现评测
│   ├── reproduce_dspark_eval.py    #   真实投机解码→接受率/平均长度 vs HF
│   ├── launch_dspark_eval.sh       #   起 vLLM dspark 服务
│   ├── run_dspark_eval.sh          #   evaluate.py(需 guidellm)
│   └── math_reasoning/             #   math_reasoning.jsonl 评测集
└── attention_experiment/           # 注意力分布实验(自包含)
    ├── REPORT.md                   #   完整实验报告
    ├── collect_attention.py / analyze_attention.py / plot_attention_focused.py / plot_attention_layer.py
    ├── attention_raw/              #   10 样本原始注意力 .pt
    └── *.png                       #   结果图
```

## 各块结论速览

- **Baseline**(`baseline_qwen3_4b_70w/BASELINE.md`): 3 epoch; 每 epoch 均值 accept_len 3.76→4.47→4.74, loss 0.180→0.143→0.129; val accept_len 4.277→4.625→4.758; **每 pos 接受率**(epoch2): pos1-7 = 85.2→63.2%。
- **本地 4:1 vs 185 16卡(同 step)**: 185 每步 loss 更低、accept_len 更高 —— dp=12 有效 batch 更大梯度更稳。**创新对比必须保持 dp 一致**。
- **HF DSpark 复现**(`dspark_reproduce/`): 真实投机解码 avg_len **4.60**(HF 上报 5.37), 逐位置形态一致。差异可能来自"解析式 vs 真实解码"口径。
- **注意力实验**(`attention_experiment/REPORT.md`): 关键答案 token 绝对注意力 ~2.2%(×6.5 均匀), 被 BOS(41%, 深层 64%)主导; 分布尖峰; 越深越聚焦。

## 关键用法

```bash
# baseline 还原/重出曲线
python3 baseline_qwen3_4b_70w/reconstruct_metrics.py baseline_qwen3_4b_70w/train_logs.log -o .

# 复现 HF DSpark 评测
bash dspark_reproduce/launch_dspark_eval.sh          # 起服务(卡12,端口1125), 另开终端:
python3 dspark_reproduce/reproduce_dspark_eval.py     # 真实投机解码 → 接受率 vs HF

# 注意力实验(见 attention_experiment/REPORT.md)
```
