#!/usr/bin/env python3
"""改进版注意力绘图: 分组聚焦式(突出 BOS/答案词/query, 其余汇总) + 多层标注 top token。

用法:
    # s05 聚焦图(长序列, 不均匀横轴)
    python3 plot_attention_focused.py --sample s05_math_medium --layer 30 \\
        --out ../attention_experiment/attention_layer_s05_L30.png
    # s01 多层标注图
    python3 plot_attention_focused.py --sample s01_statement_tokyo_short \\
        --multilayer "5 18 32" --out ../attention_experiment/multilayer_evolution_s01.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

RAW = Path(__file__).parent / "attention_raw"


def full_toks(r):
    """prompt tokens + 已生成 token(补齐答案步的输入)。"""
    t = list(r["tokens"])
    for i, g in enumerate(r["generated_tokens"][: r["answer_step_first"] + 1]):
        t.append(f"[gen{i}]{g.strip()}")
    return t


def notables(r, L):
    step = r["answer_step_first"]
    T = r["attentions"][step]["seq_len_at_step"]
    a = r["attentions"][step]["attn"][L].mean(0).numpy()
    toks = full_toks(r)
    bos = r["bos_position"]
    aw = sorted(set(r["answer_word_positions"]))
    qpos = T - 1  # 当前 query 位置(答案步最后一个输入位置)
    return dict(a=a, toks=toks, bos=bos, aw=aw, qpos=qpos, T=T, step=step)


def top_others(info, k=3):
    a, aw, bos, qpos = info["a"], info["aw"], info["bos"], info["qpos"]
    notable = set([bos, qpos]) | set(aw)
    order = a.argsort()[::-1]
    return [p for p in order if p not in notable][:k]


def plot_focused(r, L, out):
    info = notables(r, L)
    a, toks, bos, aw, qpos = info["a"], info["toks"], info["bos"], info["aw"], info["qpos"]
    bos_v = a[bos]
    aw_v = a[aw].sum()
    q_v = a[qpos]
    rest_v = 1.0 - bos_v - aw_v - q_v
    T = info["T"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})

    # ---- 左: 分组聚焦(不均匀横轴) ----
    cats = [
        ("BOS", bos_v, "#1f77b4"),
        (f"answer word\n{toks[aw[0]].strip()!r}" + (f"+{toks[aw[-1]].strip()!r}" if len(aw) > 1 else ""), aw_v, "#d62728"),
        (f"query/current\n(tok {qpos}: {toks[qpos].strip()!r})", q_v, "#2ca02c"),
        (f"other {T-3} tokens", rest_v, "#bbbbbb"),
    ]
    xs = np.arange(len(cats))
    ax.bar(xs, [c[1] * 100 for c in cats], color=[c[2] for c in cats], width=0.6)
    for x, (name, v, c) in zip(xs, cats):
        ax.text(x, v * 100 + 0.5, f"{v*100:.1f}%", ha="center", fontsize=11, fontweight="bold")
        ax.text(x, -1.5, name, ha="center", fontsize=8)
    ax.set_ylim(0, max(c[1] for c in cats) * 100 * 1.15)
    ax.set_ylabel("head-avg attention (%)")
    ax.set_title(f"{r['id']}  layer {L}  (T={T}, answer_step={r['answer_step_first']})")
    ax.grid(axis="y", alpha=0.3)

    # ---- 右: 各层 答案词/BOS/query ----
    attn = r["attentions"][r["answer_step_first"]]["attn"].float().mean(1).numpy()
    Ln = attn.shape[0]
    xs2 = np.arange(Ln)
    ax2.plot(xs2, attn[:, aw].sum(1) * 100, label="answer word", color="#d62728", lw=2)
    ax2.plot(xs2, attn[:, bos] * 100, label="BOS", color="#1f77b4", lw=1.5, ls="--")
    ax2.plot(xs2, attn[:, qpos] * 100, label="query/current", color="#2ca02c", lw=1.5, ls=":")
    ax2.axvline(L, color="gray", lw=1)
    ax2.set_xlabel("layer"); ax2.set_ylabel("attention (%)")
    ax2.set_title("per-layer (vertical = chosen layer)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2)

    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"已写 {out}")
    print(f"  层{L}: BOS={bos_v*100:.1f}% 答案词={aw_v*100:.1f}% query={q_v*100:.1f}% 其余={rest_v*100:.1f}%")


def plot_multilayer(r, layers, out):
    fig, axes = plt.subplots(1, len(layers), figsize=(7 * len(layers), 4.5))
    if len(layers) == 1:
        axes = [axes]
    for ax, L in zip(axes, layers):
        info = notables(r, L)
        a, toks, bos, aw, qpos = info["a"], info["toks"], info["bos"], info["aw"], info["qpos"]
        T = info["T"]
        colors = ["#cccccc"] * T
        colors[bos] = "#1f77b4"
        for p in aw:
            colors[p] = "#d62728"
        colors[qpos] = "#2ca02c"
        others = top_others(info, 3)
        for p in others:
            colors[p] = "#ff7f0e"
        ax.bar(np.arange(T), a * 100, color=colors, width=0.8)

        # 标注: BOS/答案词/query + top-3 其它
        annotate = [bos] + aw + [qpos] + others
        for p in sorted(set(annotate)):
            ax.annotate(
                f"p{p} {toks[p].strip()!r}\n{a[p]*100:.1f}%",
                (p, a[p] * 100), textcoords="offset points", xytext=(0, 3),
                fontsize=7, ha="center", rotation=0,
            )
        ax.set_title(
            f"L{L}: BOS {a[bos]*100:.1f}% | ans {a[aw].sum()*100:.1f}% | "
            f"query {a[qpos]*100:.1f}%", fontsize=9
        )
        ax.set_xlabel("token position"); ax.set_ylabel("head-avg attention (%)")
        ax.set_ylim(0, max(a) * 100 * 1.35)
        ax.grid(alpha=0.2)

    handles = [
        mpatches.Patch(color="#1f77b4", label="BOS"),
        mpatches.Patch(color="#d62728", label="answer word"),
        mpatches.Patch(color="#2ca02c", label="query/current"),
        mpatches.Patch(color="#ff7f0e", label="top-3 other"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle(f"{r['id']}  attention at answer step (T={info['T']})", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"已写 {out}")
    for L in layers:
        info = notables(r, L)
        a, toks, bos, aw, qpos = info["a"], info["toks"], info["bos"], info["aw"], info["qpos"]
        print(f"  L{L}: BOS {a[bos]*100:.1f}% ans {a[aw].sum()*100:.1f}% query {a[qpos]*100:.1f}%")
        for p in top_others(info, 3):
            print(f"     top其他 p{p} {toks[p].strip()!r} {a[p]*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--multilayer", default=None, help="空格分隔的层列表, 如 '5 18 32'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    r = torch.load(RAW / f"{args.sample}.pt", weights_only=False)
    if args.multilayer:
        layers = [int(x) for x in args.multilayer.split()]
        plot_multilayer(r, layers, args.out)
    else:
        plot_focused(r, args.layer, args.out)


if __name__ == "__main__":
    main()
