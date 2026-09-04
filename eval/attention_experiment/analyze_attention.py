#!/usr/bin/env python3
"""分析注意力采集数据: 答案词/句、BOS、末尾 token 的注意力占比, 层间趋势, 分布集中度。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

RAW = Path("/home/y50063564/processed_data/archive_dspark_20260904/eval/attention_raw")


def analyze_one(r: dict) -> dict:
    step = r["answer_step_first"]
    attn = r["attentions"][step]["attn"].float()  # [L, H, T]
    L, H, T = attn.shape
    aw = r["answer_word_positions"]
    asen = r["answer_sentence_positions"]
    bos, last = r["bos_position"], r["last_prompt_position"]
    a_avg = attn.mean(dim=1)  # [L, T] 头平均

    aw_attn = a_avg[:, aw].sum(dim=1)        # [L] 答案词
    asen_attn = a_avg[:, asen].sum(dim=1)    # [L] 答案句
    bos_attn = a_avg[:, bos]                 # [L] BOS
    last_attn = a_avg[:, last]               # [L] 末尾(最后 prompt token)

    uniform = 1.0 / T
    # 答案词在注意力分布里的最好排名(层0 为参考, 越小越靠前)
    order0 = a_avg[0].argsort(descending=True).tolist()
    aw_rank0 = min(order0.index(p) for p in aw)
    # 分布集中度: top-5 token 的注意力总和(层平均)
    top5 = a_avg.topk(5, dim=1).values.sum(dim=1)  # [L]
    # 熵(层平均)
    p = a_avg.clamp(min=1e-9)
    ent = -(p * p.log()).sum(dim=1)  # [L]

    return {
        "id": r["id"], "T": T, "step": step,
        "aw_attn": aw_attn.numpy(), "asen_attn": asen_attn.numpy(),
        "bos_attn": bos_attn.numpy(), "last_attn": last_attn.numpy(),
        "uniform": uniform, "aw_rank0": aw_rank0,
        "top5": top5.numpy(), "entropy": ent.numpy(),
        "L": L,
    }


def main() -> None:
    samples = sorted(RAW.glob("*.pt"))
    res = [analyze_one(torch.load(f, weights_only=False)) for f in samples]
    print(f"样本数: {len(res)}\n")

    # ---- 各层段汇总(early 0-11 / mid 12-23 / late 24-35) ----
    seg = {"early(0-11)": slice(0, 12), "mid(12-23)": slice(12, 24), "late(24-35)": slice(24, 36)}
    print(f"{'样本':24} {'T':>4} {'答案词%':>7} {'x均匀':>5} {'答案句%':>7} {'BOS%':>5} "
          f"{'末尾%':>5} {'top5%':>6} {'答案词rank':>9}")
    rows = []
    for r in res:
        aw_m = r["aw_attn"].mean()
        print(f"{r['id']:24} {r['T']:>4} {aw_m*100:>6.2f}% {aw_m/r['uniform']:>5.1f} "
              f"{r['asen_attn'].mean()*100:>6.2f}% {r['bos_attn'].mean()*100:>4.1f}% "
              f"{r['last_attn'].mean()*100:>4.1f}% {r['top5'].mean()*100:>5.1f}% "
              f"{r['aw_rank0']:>9}")
        rows.append(r)

    # ---- 层间趋势(答案词/BOS/末尾, 3 层段均值) ----
    print("\n=== 层间趋势(各层段均值) ===")
    print(f"{'层段':16} {'答案词%':>8} {'BOS%':>7} {'末尾%':>7} {'top5%':>7} {'熵(bit)':>8}")
    for name, sl in seg.items():
        aw = np.mean([r["aw_attn"][sl].mean() for r in res])
        bo = np.mean([r["bos_attn"][sl].mean() for r in res])
        la = np.mean([r["last_attn"][sl].mean() for r in res])
        t5 = np.mean([r["top5"][sl].mean() for r in res])
        en = np.mean([r["entropy"][sl].mean() for r in res])
        print(f"{name:16} {aw*100:>7.2f}% {bo*100:>6.1f}% {la*100:>6.1f}% {t5*100:>6.1f}% {en:>8.2f}")

    # ---- 答案词注意力随层的完整曲线(全部样本均值) ----
    L = res[0]["L"]
    aw_by_layer = np.mean([r["aw_attn"] for r in res], axis=0)   # [L]
    bo_by_layer = np.mean([r["bos_attn"] for r in res], axis=0)
    la_by_layer = np.mean([r["last_attn"] for r in res], axis=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = np.arange(L)
    a1.plot(xs, aw_by_layer * 100, label="answer word", lw=2)
    a1.plot(xs, bo_by_layer * 100, label="BOS", lw=1.5, ls="--")
    a1.plot(xs, la_by_layer * 100, label="last prompt token", lw=1.5, ls=":")
    a1.set_xlabel("layer"); a1.set_ylabel("head-avg attention (%)")
    a1.set_title("attention to key/BOS/last by layer (mean over 10 samples)")
    a1.legend(); a1.grid(alpha=0.3)
    a2.plot(xs, aw_by_layer / np.array([r["uniform"] for r in res]).mean() * 100, lw=2)
    a2.set_xlabel("layer")
    a2.set_ylabel("answer-word attention x uniform baseline")
    a2.set_title("answer word vs uniform (1.0 = uniform)")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RAW.parent / "attention_analysis.png", dpi=150)
    print(f"\n已写 {RAW.parent / 'attention_analysis.png'}")

    # ---- 结论摘要 ----
    unis = np.array([r["uniform"] for r in res]).mean()
    aw_all = np.mean([r["aw_attn"].mean() for r in res])
    print(f"\n=== 摘要 ===")
    print(f"平均上下文长度: {np.mean([r['T'] for r in res]):.0f} token, 均匀基线={unis*100:.2f}%")
    print(f"答案词平均注意力: {aw_all*100:.2f}%  (×{aw_all/unis:.1f} 均匀基线)")
    print(f"答案词在注意力分布中的排名(层0): 中位 {np.median([r['aw_rank0'] for r in res])}, "
          f"范围 {min(r['aw_rank0'] for r in res)}-{max(r['aw_rank0'] for r in res)}")
    print(f"BOS 平均: {np.mean([r['bos_attn'].mean() for r in res])*100:.2f}%")
    print(f"末尾 token 平均: {np.mean([r['last_attn'].mean() for r in res])*100:.2f}%")


if __name__ == "__main__":
    main()
