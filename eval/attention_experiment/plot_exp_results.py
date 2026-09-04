#!/usr/bin/env python3
"""为 3 组实验生成图:
  exp1_heatmap.png      - 长度×位置, L4 关键token注意力热力图
  exp1_layer_trend.png  - 每层注意力趋势(代表配置)
  exp2_distractors.png  - P(Alpha) 与 关键/distractor L4注意力 vs distractor数
  exp3_prefix.png       - 前缀测试各配置的 P(Delta)/P(Beta)
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent


def plot_exp1():
    rows = json.load(open(HERE / "exp1_results_alpha/exp1_results.json"))
    Ls = [128, 256, 512, 1024, 2048]
    Ps = [10, 25, 50, 75, 90]
    mat = np.zeros((len(Ls), len(Ps)))
    for r in rows:
        i, j = Ls.index(r["length"]), Ps.index(r["position_pct"])
        mat[i, j] = r["key_attn_by_layer"][4] * 100  # L4
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    im = a1.imshow(mat, cmap="viridis", aspect="auto")
    a1.set_xticks(range(len(Ps))); a1.set_xticklabels([f"{p}%" for p in Ps])
    a1.set_yticks(range(len(Ls))); a1.set_yticklabels([str(l) for l in Ls])
    a1.set_xlabel("key position"); a1.set_ylabel("length")
    a1.set_title("exp1: L4 key-token attention (%)")
    fig.colorbar(im, ax=a1)
    for i in range(len(Ls)):
        for j in range(len(Ps)):
            a1.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=8, color="white")
    # 每层趋势(128/512/2048 @ pos50)
    for L in [128, 512, 2048]:
        r = next(x for x in rows if x["length"] == L and x["position_pct"] == 50)
        a2.plot(range(5), [v * 100 for v in r["key_attn_by_layer"]], "o-", label=f"L={L}")
    a2.set_xlabel("draft layer"); a2.set_ylabel("key-token attention (%)")
    a2.set_title("exp1: per-layer trend (pos 50%)"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "exp1_heatmap.png", dpi=150)
    print("exp1_heatmap.png")


def plot_exp2():
    rows = json.load(open(HERE / "exp2_results_alpha/exp2_results.json"))
    ns = [r["n_distractors"] for r in rows]
    p = [r["p_first_subword"] for r in rows]
    key = [r["key_attn_by_layer"][4] * 100 for r in rows]
    dist = [r["dist_attn_by_layer"][4] * 100 for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.plot(ns, p, "o-", color="#d62728", lw=2, label="P(Alpha)")
    a1.axhline(0.5, color="gray", ls="--", lw=0.8)
    a1.set_xscale("log", base=2); a1.set_xticks(ns); a1.set_xticklabels(ns)
    a1.set_xlabel("n distractors"); a1.set_ylabel("P(Alpha)")
    a1.set_title("exp2: P(Alpha) vs distractor count"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(ns, key, "o-", color="#1f77b4", lw=2, label="key token L4 attn")
    a2.plot(ns, dist, "s--", color="#ff7f0e", lw=1.5, label="avg distractor L4 attn")
    a2.set_xscale("log", base=2); a2.set_xticks(ns); a2.set_xticklabels(ns)
    a2.set_xlabel("n distractors"); a2.set_ylabel("attention (%)")
    a2.set_title("exp2: L4 attention vs distractor count"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "exp2_distractors.png", dpi=150)
    print("exp2_distractors.png")


def plot_exp3():
    # 前缀测试各配置
    data = [("aligned×1", 0.304, 0.645), ("was_changed", 0.492, 0.28),
            ("aligned×2", 0.971, 0.001), ("aligned+mis", 0.578, 0.31),
            ("mis+aligned", 0.963, 0.002)]
    names = [d[0] for d in data]
    pnew = [d[1] for d in data]; pold = [d[2] for d in data]
    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - w / 2, pnew, w, label="P(Delta) 新码", color="#2ca02c")
    ax.bar(x + w / 2, pold, w, label="P(Beta) 旧码", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("probability"); ax.set_title("exp3: prefix-alignment effect on P(Delta)/P(Beta)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "exp3_prefix.png", dpi=150)
    print("exp3_prefix.png")


if __name__ == "__main__":
    plot_exp1(); plot_exp2(); plot_exp3()
