#!/usr/bin/env python3
"""diff-transformer 训练 vs baseline: accept_len / loss 对比。

横轴 = global_step(每模型 4096 token/步, 即"看过的数据量")。
注意: baseline 是 dp=12(每步梯度来自 12×4096 多样数据, 有效 batch 更大),
diff 是单卡 dp=1。同一 global_step 下 baseline 的梯度质量更好。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

BASE = "/home/y50063564/processed_data/archive_dspark_20260904/eval/baseline_qwen3_4b_70w/train_metrics.csv"
RUNS = {
    "baseline (dp=12, gqa)": BASE,
    "diff l0 3l": "/home/y50063564/processed_data/archive_dspark_20260904/eval/diff_l0_3l_metrics/train_metrics.csv",
    "diff l0 5l": "/home/y50063564/processed_data/archive_dspark_20260904/eval/diff_l0_5l_metrics/train_metrics.csv",
    "diff l1 5l": "/home/y50063564/processed_data/archive_dspark_20260904/eval/diff_l1_5l_metrics/train_metrics.csv",
}
COLORS = {"baseline": "#1f77b4", "l0_3l": "#d62728", "l0_5l": "#2ca02c", "l1_5l": "#ff7f0e"}


def smooth(s, w=201):
    return s.rolling(w, center=True, min_periods=1).mean()


def main():
    data = {}
    for name, path in RUNS.items():
        df = pd.read_csv(path)
        data[name] = df
        print(f"{name:22s}: {len(df)} 步, accept_len 末值={df['train/accept_len'].iloc[-1]:.3f}, "
              f"loss 末值={df['train/loss'].iloc[-1]:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    # (1) accept_len vs global_step, 0-30k (baseline 有数据的公平窗口)
    ax = axes[0, 0]
    for name, df in data.items():
        c = COLORS[list(COLORS.keys())[0]] if "baseline" in name else COLORS[{"diff l0 3l":"l0_3l","diff l0 5l":"l0_5l","diff l1 5l":"l1_5l"}[name]]
        sub = df[df["global_step"] <= 30000]
        ax.plot(sub["global_step"], smooth(sub["train/accept_len"]), lw=2, label=name, color=c)
    ax.set_xlabel("global_step (tokens seen = step × 4096)")
    ax.set_ylabel("train/accept_len"); ax.set_title("accept_len (window 0-30k)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (2) loss vs global_step, 0-30k
    ax = axes[0, 1]
    for name, df in data.items():
        c = COLORS[list(COLORS.keys())[0]] if "baseline" in name else COLORS[{"diff l0 3l":"l0_3l","diff l0 5l":"l0_5l","diff l1 5l":"l1_5l"}[name]]
        sub = df[df["global_step"] <= 30000]
        ax.plot(sub["global_step"], smooth(sub["train/loss"]), lw=2, label=name, color=c)
    ax.set_xlabel("global_step (tokens seen = step × 4096)")
    ax.set_ylabel("train/loss"); ax.set_title("loss (window 0-30k)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (3) accept_len 全范围(看 diff 的长期走势)
    ax = axes[1, 0]
    for name, df in data.items():
        c = COLORS[list(COLORS.keys())[0]] if "baseline" in name else COLORS[{"diff l0 3l":"l0_3l","diff l0 5l":"l0_5l","diff l1 5l":"l1_5l"}[name]]
        ax.plot(df["global_step"], smooth(df["train/accept_len"], 501), lw=2, label=name, color=c)
    ax.set_xlabel("global_step"); ax.set_ylabel("accept_len")
    ax.set_title("accept_len (full range)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (4) loss 全范围
    ax = axes[1, 1]
    for name, df in data.items():
        c = COLORS[list(COLORS.keys())[0]] if "baseline" in name else COLORS[{"diff l0 3l":"l0_3l","diff l0 5l":"l0_5l","diff l1 5l":"l1_5l"}[name]]
        ax.plot(df["global_step"], smooth(df["train/loss"], 501), lw=2, label=name, color=c)
    ax.set_xlabel("global_step"); ax.set_ylabel("loss")
    ax.set_title("loss (full range)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Diff-Transformer draft vs baseline GQA (x = data seen per model)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("/home/y50063564/processed_data/archive_dspark_20260904/eval/diff_vs_baseline.png", dpi=150)
    print("已写 /home/y50063564/processed_data/archive_dspark_20260904/eval/diff_vs_baseline.png")


if __name__ == "__main__":
    main()
