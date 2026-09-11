#!/usr/bin/env python3
"""Buffer-depletion test.

Per rank and per training step i (all in one aligned timeline file):
    D_i        = docs in this batch
    L_i        = next(loader) wait  (load_ms)
    W_i        = FSDP/forward wait  (t_fwd - t_batch)

Hypothesis: the stall is driven by *cumulative consumption* over a trailing
window, not by the current batch's doc count. We therefore test
    corr(L_i, C^{(K)}_{i-lag})   with  C^{(K)}_i = sum_{j=i-K+1..i} D_j
for K in {4,8,16} and lag in 0..8, and compare against corr(L_i, D_i).
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RAW = os.environ.get("DL_RAW", "/home/y50063564/dataloader_diag/raw_tl_w2")
OUT = os.environ.get("DL_OUT", "/home/y50063564/dataloader_diag/analysis_buffer")
KS = (4, 8, 16)
LAGS = range(0, 9)


def load_ranks():
    per = defaultdict(list)
    for f in glob.glob(f"{RAW}/rank*/timeline_rank*.jsonl"):
        r = int(f.split("rank")[-1].split("/")[0])
        for line in open(f, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            per[r].append(d)
    out = {}
    for r, rows in per.items():
        df = pd.DataFrame(rows).sort_values("global_step").drop_duplicates("global_step")
        df = df.reset_index(drop=True)
        df["docs"] = df["docs_in_batch"].astype(float)
        df["load"] = df["load_ms"].astype(float)
        df["fwd_wait"] = (df["t_fwd"] - df["t_batch"]) * 1000.0
        for K in KS:
            df[f"C{K}"] = df["docs"].rolling(K, min_periods=1).sum()
        out[r] = df
    return out


def corr_table(df, target):
    """best (K, lag, rho) for each trailing-window K, and for the current batch."""
    y = df[target].values
    res = {}
    base = spearmanr(y, df["docs"].values).statistic
    res["D_i (same step)"] = (None, 0, base)
    for K in KS:
        best = (-2, None)
        for lag in LAGS:
            x = df[f"C{K}"].shift(lag).values
            m = ~np.isnan(x)
            if m.sum() < 50:
                continue
            rho = spearmanr(y[m], x[m]).statistic
            if rho > best[0]:
                best = (rho, lag)
        res[f"C{{K={K}}}"] = (K, best[1], best[0])
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    data = load_ranks()
    if not data:
        print("no timeline data"); return
    print(f"ranks: {sorted(data)}   steps/rank: " +
          ", ".join(f"r{r}:{len(d)}" for r, d in sorted(data.items())))

    print("\n=== per-rank summary (median) ===")
    print(f"{'rank':>4} {'docs/batch':>10} {'load_ms':>9} {'fwd_wait_ms':>12}")
    for r, d in sorted(data.items()):
        print(f"{r:>4} {d.docs.median():>10.2f} {d.load.median():>9.1f} {d.fwd_wait.median():>12.1f}")

    for target, tname in (("load", "next(loader) wait L_i"), ("fwd_wait", "FSDP/forward wait W_i")):
        print(f"\n=== Spearman({tname}, trailing-docs) ===")
        print(f"{'rank':>4} " + " ".join(f"{k:>22}" for k in
              ["D_i (same step)"] + [f"C{{K={K}}} best(lag,rho)" for K in KS]))
        for r, d in sorted(data.items()):
            res = corr_table(d, target)
            cells = [f"{res['D_i (same step)'][2]:>22.3f}"]
            for K in KS:
                _, lag, rho = res[f"C{{K={K}}}"]
                cells.append(f"{('lag%d, rho=%.3f' % (lag, rho)):>22}")
            print(f"{r:>4} " + " ".join(cells))

    # ---- plot: docs/batch + cumulative curves + load + fwd wait ----
    ranks = sorted(data)
    fig, axes = plt.subplots(3, len(ranks), figsize=(5 * len(ranks), 11), sharex=True)
    if len(ranks) == 1:
        axes = axes.reshape(3, 1)
    for j, r in enumerate(ranks):
        d = data[r]
        x = d.global_step.values
        ax0, ax1, ax2 = axes[0, j], axes[1, j], axes[2, j]
        ax0.bar(x, d.docs, width=1.0, color="#bbbbbb", label="docs/batch")
        ax0.set_ylabel("docs / batch"); ax0.set_title(f"rank{r}")
        axc = ax0.twinx()
        for K, c in zip(KS, ("#1f77b4", "#2ca02c", "#d62728")):
            axc.plot(x, d[f"C{K}"], lw=1.2, color=c, label=f"C(K={K})")
        axc.set_ylabel("trailing Σdocs"); ax0.legend(fontsize=7, loc="upper left")
        axc.legend(fontsize=7, loc="lower right")
        ax1.plot(x, d.load, lw=0.8, color="#d62728")
        ax1.set_ylabel("next(loader) wait (ms)")
        ax1.axhline(d.load.median(), ls=":", color="gray")
        ax2.plot(x, d.fwd_wait, lw=0.8, color="#1f77b4")
        ax2.set_ylabel("FSDP/forward wait (ms)"); ax2.set_xlabel("global_step")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    fig.suptitle("Buffer dynamics: docs/batch vs next(loader) & FSDP wait  (raw_tl_w2, W2/dp4)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = f"{OUT}/buffer_dynamics.png"
    fig.savefig(p, dpi=130)
    print(f"\nwrote {p}")
    for r, d in data.items():
        d.to_csv(f"{OUT}/timeline_rank{r}.csv", index=False)
    print(f"csvs under {OUT}/")


if __name__ == "__main__":
    main()
