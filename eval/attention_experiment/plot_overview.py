#!/usr/bin/env python3
"""项目总览图: baseline 训练曲线(accept_len/loss) + 3 个实验场景。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BASELINE_CSV = "/home/y50063564/processed_data/archive_dspark_20260904/eval/baseline_qwen3_4b_70w/train_metrics.csv"


def smooth(s, w=301):
    return s.rolling(w, center=True, min_periods=1).mean()


def main():
    df = pd.read_csv(BASELINE_CSV)
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])

    # --- 上排: baseline 训练曲线 ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df["global_step"], smooth(df["train/loss"]), color="#d62728", lw=2)
    ax1.set_xlabel("global_step"); ax1.set_ylabel("train/loss")
    ax1.set_title("Baseline: loss (3 epochs, dp=12)"); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df["global_step"], smooth(df["train/accept_len"]), color="#1f77b4", lw=2)
    ax2.set_xlabel("global_step"); ax2.set_ylabel("train/accept_len")
    ax2.set_title("Baseline: accept_len (3 epochs, dp=12)"); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    # 每 epoch 均值柱状(基线最终能力)
    g = df.groupby("epoch")[["train/accept_len", "train/loss"]].mean()
    x = np.arange(3); w = 0.35
    ax3.bar(x - w/2, g["train/accept_len"], w, color="#1f77b4", label="accept_len")
    ax3.set_ylabel("accept_len")
    ax3b = ax3.twinx()
    ax3b.bar(x + w/2, g["train/loss"], w, color="#d62728", label="loss")
    ax3b.set_ylabel("loss")
    ax3.set_xticks(x); ax3.set_xticklabels(["ep0", "ep1", "ep2"])
    ax3.set_title("Baseline: per-epoch mean"); ax3.grid(alpha=0.2)

    # --- 下排: 3 个实验场景 ---
    ax4 = fig.add_subplot(gs[1, 0])  # 实验1 热力图(L4)
    rows = json.load(open(HERE / "exp1_results_alpha/exp1_results.json"))
    Ls = [128, 256, 512, 1024, 2048]; Ps = [10, 25, 50, 75, 90]
    mat = np.array([[next(x for x in rows if x["length"]==L and x["position_pct"]==p)["key_attn_by_layer"][4]*100
                     for p in Ps] for L in Ls])
    im = ax4.imshow(mat, cmap="viridis", aspect="auto")
    ax4.set_xticks(range(len(Ps))); ax4.set_xticklabels([f"{p}%" for p in Ps])
    ax4.set_yticks(range(len(Ls))); ax4.set_yticklabels([str(l) for l in Ls])
    ax4.set_xlabel("key position"); ax4.set_ylabel("length")
    ax4.set_title("Exp1: key-token L4 attention (%)")
    fig.colorbar(im, ax=ax4, fraction=0.046)

    ax5 = fig.add_subplot(gs[1, 1])  # 实验2
    rows2 = json.load(open(HERE / "exp2_results_alpha/exp2_results.json"))
    ns = [r["n_distractors"] for r in rows2]
    p = [r["p_first_subword"] for r in rows2]
    key = [r["key_attn_by_layer"][4]*100 for r in rows2]
    ax5.plot(ns, p, "o-", color="#d62728", lw=2, label="P(Alpha)")
    ax5.set_xscale("log", base=2); ax5.set_xticks(ns); ax5.set_xticklabels(ns)
    ax5.set_xlabel("n distractors"); ax5.set_ylabel("P(Alpha)")
    ax5b = ax5.twinx()
    ax5b.plot(ns, key, "s--", color="#1f77b4", lw=1.5, label="key L4 attn%")
    ax5b.set_ylabel("key L4 attn (%)")
    ax5.set_title("Exp2: P(Alpha) & key-attn vs distractors")
    ax5.grid(alpha=0.2)

    ax6 = fig.add_subplot(gs[1, 2])  # 实验3 前缀
    data = [("aligned×1", 0.304, 0.645), ("was_changed", 0.492, 0.28),
            ("aligned×2", 0.971, 0.001), ("aligned+mis", 0.578, 0.31)]
    names = [d[0] for d in data]; pn = [d[1] for d in data]; po = [d[2] for d in data]
    x = np.arange(len(names)); wd = 0.35
    ax6.bar(x-wd/2, pn, wd, color="#2ca02c", label="P(Delta) new")
    ax6.bar(x+wd/2, po, wd, color="#d62728", label="P(Beta) old")
    ax6.set_xticks(x); ax6.set_xticklabels(names, rotation=12, fontsize=8)
    ax6.set_ylabel("probability")
    ax6.set_title("Exp3: prefix-alignment (draft P)")
    ax6.legend(fontsize=8); ax6.grid(axis="y", alpha=0.2)

    fig.suptitle("DSpark draft: baseline training + extraction experiment overview", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(HERE / "overview.png", dpi=150)
    print(f"已写 {HERE}/overview.png")


if __name__ == "__main__":
    main()
