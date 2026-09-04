#!/usr/bin/env python3
"""画某样本在某层的注意力分布(答案生成步), 高亮 BOS / 答案词 / 末尾 token。

用法:
    python3 plot_attention_layer.py [--sample s01_statement_tokyo_short]
                                    [--layer 30] [-o eval/attention_layer.png]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

RAW = Path("/home/y50063564/processed_data/archive_dspark_20260904/eval/attention_raw")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s01_statement_tokyo_short")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("-o", default="/home/y50063564/processed_data/archive_dspark_20260904/eval/attention_layer.png")
    args = ap.parse_args()

    r = torch.load(RAW / f"{args.sample}.pt", weights_only=False)
    step = r["answer_step_first"]
    attn = r["attentions"][step]["attn"].float()  # [L, H, T]
    L, H, T = attn.shape
    tokens = r["tokens"]
    aw = r["answer_word_positions"]
    asen = r["answer_sentence_positions"]
    bos, last = r["bos_position"], r["last_prompt_position"]

    a = attn[args.layer].mean(dim=0).numpy()  # [T] 该层头平均
    a_avg = attn.mean(dim=1).numpy()          # [L, T] 各层头平均

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    # ---- 左: 该层注意力分布(按 token 位置) ----
    x = np.arange(T)
    colors = ["#cccccc"] * T
    colors[bos] = "#1f77b4"
    for p in aw:
        colors[p] = "#d62728"
    colors[last] = "#2ca02c"
    ax.bar(x, a * 100, color=colors, width=0.8)
    if asen:
        ax.axvspan(min(asen), max(asen), color="orange", alpha=0.12)
    for p in sorted(set([bos, last] + aw)):
        ax.annotate(
            f"{p}:{tokens[p][:10].strip()!r}",
            (p, a[p] * 100), textcoords="offset points",
            xytext=(0, 4), fontsize=6.5, rotation=90, ha="center",
        )
    ax.set_xlabel("token position")
    ax.set_ylabel(f"layer {args.layer} head-avg attention (%)")
    ax.set_title(f"{args.sample} | T={T} | answer_step={step} | "
                 f"orange = answer sentence span")
    handles = [
        mpatches.Patch(color="#1f77b4", label=f"BOS (pos {bos})"),
        mpatches.Patch(color="#d62728", label=f"answer word {r['answer_word']}"),
        mpatches.Patch(color="#2ca02c", label=f"last prompt token (pos {last})"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)

    # ---- 右: 各层的答案词 / BOS / 末尾 注意力 ----
    awl = a_avg[:, aw].sum(axis=1) * 100
    bl = a_avg[:, bos] * 100
    ll = a_avg[:, last] * 100
    xs = np.arange(L)
    ax2.plot(xs, awl, label=f"answer word", color="#d62728", lw=2)
    ax2.plot(xs, bl, label="BOS", color="#1f77b4", lw=1.5, ls="--")
    ax2.plot(xs, ll, label="last token", color="#2ca02c", lw=1.5, ls=":")
    ax2.axvline(args.layer, color="gray", lw=1, ls="-")
    ax2.set_xlabel("layer")
    ax2.set_ylabel("attention (%)")
    ax2.set_title(f"per-layer (line = layer {args.layer})")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(args.o, dpi=150)
    print(f"已写 {args.o}")
    print(f"层{args.layer}: 答案词={a[aw].sum()*100:.2f}%  BOS={a[bos]*100:.2f}%  "
          f"末尾={a[last]*100:.2f}%  答案句span={a[asen].sum()*100:.2f}%")


if __name__ == "__main__":
    main()
