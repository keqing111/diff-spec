#!/usr/bin/env python3
"""汇总绘图: exp1/exp2/exp3 结果 (读 results_*/*.json, 不需要 NPU)。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/home/y50063564/processed_data/dspark_data/ctx_diff_attention_exp")


def load(p):
    return json.load(open(BASE / p))


# ---------------- exp1 ----------------
v1 = load("results_exp1/exp1_vertical.json")
h1 = load("results_exp1/exp1_horizontal.json")
j1 = load("results_exp1/exp1_jscan.json")

order = [(L, p) for L in (128, 512, 2048) for p in (10, 50, 90)]
cellmap = {(r["length"], r["position_pct"]): r for r in v1}

fig = plt.figure(figsize=(17, 11))
# 1a vertical heatmap
ax = fig.add_subplot(2, 2, (1, 3))
lay_cols = ["L0(b0+b1)", "L0(diff)", "L1", "L2", "L3", "L4"]
M = np.zeros((len(order), 6))
Pvals = []
for i, (L, p) in enumerate(order):
    r = cellmap[(L, p)]
    l = r["layer"]
    M[i] = [l["0"]["both_raw"], l["0"]["diff_raw"]] + [l[str(k)]["both_raw"] for k in range(1, 5)]
    Pvals.append(r["p_first_subword"] or float("nan"))
M *= 100.0
im = ax.imshow(M, aspect="auto", cmap="viridis")
ax.set_xticks(range(6)); ax.set_xticklabels(lay_cols)
ax.set_yticks(range(9))
ax.set_yticklabels([f"L{L} p{p}%" for L, p in order])
for i in range(9):
    for j in range(6):
        ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7, color="w")
for i, pv in enumerate(Pvals):
    ax.text(6.35, i, f"P={pv:.2f}", ha="left", va="center", fontsize=8, color="#000")
fig.colorbar(im, ax=ax, fraction=0.03, label="% attention on key span")
ax.set_title("exp1 ctx: key-token attention share at answer slot per layer (9 cells)  |  right: draft P(Alpha)")

# 1b horizontal pair (one representative cell shown)
axs = fig.add_subplot(2, 2, 2)
k0 = "L512_p50"
A = np.array([[r["b0"], r["b1"], r["diff"]] for r in h1[k0]]) * 100.0
im = axs.imshow(A, aspect="auto", cmap="RdBu_r", vmin=-max(abs(A.min()), abs(A.max())), vmax=max(abs(A.min()), abs(A.max())))
axs.set_yticks(range(16)); axs.set_xticks(range(3)); axs.set_xticklabels(["b0", "b1", "diff"])
axs.set_title("exp1 horizontal: diff-L0 16 pairs b0/b1/diff on key span (512/50)")
fig.colorbar(im, ax=axs, fraction=0.05)

# 1c jscan
ax = fig.add_subplot(2, 2, 4)
k0 = "L512_p50"
js = j1[k0]
js = {int(j): v for j, v in js.items()}
cols = {"0": "#1f77b4", "1": "#ff7f0e", "2": "#2ca02c", "3": "#d62728", "4": "#9467bd"}
for li in range(5):
    ys = [js[j][str(li)]["both_raw"] * 100 for j in sorted(js)]
    ax.plot(sorted(js), ys, "o-", label=f"L{li}" if li else "L0(b0+b1)", color=cols[str(li)])
    if li == 0:
        ax.plot(sorted(js), [js[j]["0"]["diff_raw"] * 100 for j in sorted(js)], "s--", label="L0(diff)", color="#8c564b")
ax.set_xlabel("answer slot j in last anchor block")
ax.set_ylabel("key-token attn %")
ax.legend(fontsize=8)
ax.set_title("exp1 j-scan (512/50)")
fig.tight_layout()
fig.savefig(BASE / "exp1_summary.png", dpi=150)
plt.close(fig)

# ---------------- exp2 ----------------
s2 = load("results_exp2/exp2_summary.json")
ns = sorted(int(k) for k in s2)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
xs = np.arange(len(ns))
pvals = [s2[str(n)]["p_true_tc"] or float("nan") for n in ns]
hits = [s2[str(n)]["hit_top1_tc"] if s2[str(n)]["hit_top1_tc"] is not None else float("nan") for n in ns]
a1.plot(xs, pvals, "o-", label="draft P(true) | target-OK")
a1.plot(xs, hits, "s--", label="top1-locate hit | target-OK")
a1.set_xticks(xs); a1.set_xticklabels(ns)
a1.set_xlabel("#distractors (log-ish)")
a1.legend(); a1.grid(alpha=.3); a1.set_ylim(0, 1.05)
a1.set_title("exp2: draft P & localization vs distractors")
# stacked cells (order: TT TF FT FF)
stack_order = [("TT", "tgtOK-draftOK", "#1f77b4"), ("TF", "tgtOK-draftX", "#2ca02c"),
               ("FT", "tgtX-draftOK", "#d62728"), ("FF", "both X", "#7f7f7f")]
mat = np.array([[s2[str(n)]["cells"][k] for n in ns] for k, _, _ in stack_order])
bottoms = np.vstack([np.zeros(len(ns)), np.cumsum(mat, axis=0)[:-1]])
for i, (k, lab, c) in enumerate(stack_order):
    a2.bar(xs, mat[i], label=lab, color=c, bottom=bottoms[i])
a2.set_xticks(xs); a2.set_xticklabels(ns); a2.set_xlabel("#distractors")
a2.set_title("exp2: per-n 2x2 (tgt x draft) sample counts")
a2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(BASE / "exp2_summary.png", dpi=150)
plt.close(fig)

# ---------------- exp3 ----------------
s3 = load("results_exp3/exp3_summary.json")
keys = list(s3)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8))
kk = ["TT", "TF", "FT", "FF"]
x = np.arange(len(keys))
for lab, c in [("TT", "#1f77b4"), ("TF", "#2ca02c"), ("FT", "#d62728"), ("FF", "#7f7f7f")]:
    vals = [s3[k]["cells"][lab] for k in keys]
    a1.bar(x, vals, label=f"tgt{lab[0]} draft{lab[1]}", color=c)
a1.set_xticks(x); a1.set_xticklabels([k.replace("/", " ") for k in keys], rotation=20, ha="right")
a1.set_ylabel("# samples (n=16/cond)"); a1.legend(); a1.set_title("exp3: 2x2 counts (all samples incl. target-wrong)")
a2x = np.arange(len(keys))
an = [s3[k]["attn_new-mean"] for k in keys]; ao = [s3[k]["attn_old-mean"] for k in keys]
w = 0.4
a2.bar(a2x - w / 2, an, w, label="attn on NEW-code span", color="#2ca02c")
a2.bar(a2x + w / 2, ao, w, label="attn on OLD-code span", color="#1f77b4")
a2.set_xticks(a2x); a2.set_xticklabels([k.replace("/", " ") for k in keys], rotation=20, ha="right")
a2.legend(); a2.set_title("exp3: answer-slot attn (sum 5 layers) on new vs old span")
fig.tight_layout()
fig.savefig(BASE / "exp3_summary.png", dpi=150)
plt.close(fig)

print("wrote exp1_summary.png, exp2_summary.png, exp3_summary.png")
