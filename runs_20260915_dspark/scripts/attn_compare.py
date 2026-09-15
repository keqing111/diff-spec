#!/usr/bin/env python3
"""Compare the three draft-attention runs (MHA / native GQA / MLA).

Parses train/* metrics out of the (multi-line, rich-wrapped) training logs,
writes per-step CSVs, and plots train/accept_len + train/loss vs global_step
together with the per-epoch val points.

Everything the three runs share is identical; the only difference between the
logs is --draft-attention-type (and --mla-kv-lora-rank for the mla arm).

Run:  python3 attn_compare.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

ROOT = "/home/y50063564/processed_data/dspark_data/dspark_attn"
OUT = os.path.join(ROOT, "analysis")
ARMS = [("mha", "MHA (KV 32)"), ("gqa", "GQA native (KV 8)"), ("mla", "MLA r=512")]
# mha=green, gqa=blue (reference), mla=red -- reference in the middle of the order.
COLORS = {"mha": "#2ca02c", "gqa": "#1f77b4", "mla": "#d62728"}

LOGPAT = os.path.join(ROOT, "%s", "logs", "train_attn_*_a12_*.log")
FIELDS = ("loss", "accept_len", "accept_rate", "ce_loss", "tv_loss")


def parse_log(path: str) -> dict[str, np.ndarray]:
    """Flatten the log and split it into per-global_step records.

    The log wraps each record across many physical lines, and global_step is
    printed last, so: strip all whitespace, split on 'global_step=', and take
    the last occurrence of each metric inside each chunk.
    """
    with open(path, "r", errors="replace") as fh:
        flat = re.sub(r"\s+", " ", fh.read())

    chunks = flat.split("global_step=")
    # chunks[0] is the preamble before the first record.
    steps: list[int] = []
    cols: dict[str, list[float]] = {k: [] for k in FIELDS}
    epochs: list[int] = []
    for chunk in chunks[1:]:
        m = re.match(r"(\d+)", chunk)
        if not m:
            continue
        step = int(m.group(1))
        vals = {}
        ok = True
        for k in FIELDS:
            hits = re.findall(rf"train/{k}=([0-9.eE+-]+)", chunk)
            if not hits:
                ok = False
                break
            vals[k] = float(hits[-1])
        if not ok:
            continue
        ep = re.findall(r"epoch=(\d+)", chunk)
        steps.append(step)
        epochs.append(int(ep[-1]) if ep else -1)
        for k in FIELDS:
            cols[k].append(vals[k])

    out = {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}
    out["global_step"] = np.asarray(steps, dtype=np.int64)
    out["epoch"] = np.asarray(epochs, dtype=np.int64)
    return out


def read_vals(arm: str) -> dict[int, dict]:
    """Per-epoch val metrics from each checkpoint's val_metrics.json."""
    res = {}
    for d in sorted(glob.glob(os.path.join(ROOT, arm, "checkpoints", "[0-9]*"))):
        f = os.path.join(d, "val_metrics.json")
        if os.path.isfile(f):
            with open(f) as fh:
                res[int(os.path.basename(d))] = json.load(fh)
    return res


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for arm, _ in ARMS:
        matches = sorted(glob.glob(LOGPAT % arm))
        # the completed run is the one with the most records
        best, best_n = None, -1
        for p in matches:
            d = parse_log(p)
            if d["global_step"].size > best_n:
                best, best_n = d, d["global_step"].size
        if best is None:
            print(f"!! no log for {arm}")
            return 1
        data[arm] = best
        print(f"{arm}: {best['global_step'].size} records, "
              f"steps {best['global_step'].min()}..{best['global_step'].max()}")

    # per-step CSV
    for arm, _ in ARMS:
        d = data[arm]
        path = os.path.join(OUT, f"train_{arm}.csv")
        keys = ["global_step", "epoch", *FIELDS]
        with open(path, "w") as fh:
            fh.write(",".join(keys) + "\n")
            for i in range(d["global_step"].size):
                fh.write(",".join(str(d[k][i]) for k in keys) + "\n")
        print(f"wrote {path}")

    # ---- summary: per-epoch means of the *train* metric -------------------
    print("\n" + "=" * 78)
    print("per-epoch mean of train/accept_len  (micro-batch level, n per epoch)")
    print("=" * 78)
    print(f"{'epoch':>5} | " + " | ".join(f"{a:>22}" for a, _ in ARMS))
    for ep in range(3):
        row = []
        for arm, _ in ARMS:
            d = data[arm]
            m = d["epoch"] == ep
            row.append(f"{d['accept_len'][m].mean():.4f} (n={m.sum()})")
        print(f"{ep:>5} | " + " | ".join(f"{c:>22}" for c in row))

    print("\n" + "=" * 78)
    print("per-epoch mean of train/loss")
    print("=" * 78)
    print(f"{'epoch':>5} | " + " | ".join(f"{a:>22}" for a, _ in ARMS))
    for ep in range(3):
        row = []
        for arm, _ in ARMS:
            d = data[arm]
            m = d["epoch"] == ep
            row.append(f"{d['loss'][m].mean():.5f}")
        print(f"{ep:>5} | " + " | ".join(f"{c:>22}" for c in row))

    # ---- how big is the gap vs how big is the noise? ----------------------
    print("\n" + "=" * 78)
    print("signal vs noise on the epoch-2 accept_len")
    print("=" * 78)
    e2 = {}
    for arm, _ in ARMS:
        d = data[arm]
        m = d["epoch"] == 2
        x = d["accept_len"][m]
        e2[arm] = x
        se = x.std(ddof=1) / np.sqrt(x.size)
        # split-half: means of the two halves of the epoch, a drift-robust check
        h = x.size // 2
        print(f"  {arm:>4}: mean={x.mean():.4f}  sd={x.std(ddof=1):.4f}  "
              f"se={se:.4f}  half1={x[:h].mean():.4f} half2={x[h:].mean():.4f}")
    means = np.array([e2[a].mean() for a, _ in ARMS])
    print(f"\n  across-arm spread (max-min) = {means.max() - means.min():.4f}")
    print(f"  within-arm step noise (se)  ~ {np.mean([e2[a].std(ddof=1)/np.sqrt(e2[a].size) for a,_ in ARMS]):.4f}")
    print("  NOTE: se covers per-step noise only; run-to-run (seed/platform) noise")
    print("        is NOT captured and is known to be of the same order here.")

    # ---- plots -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"\n!! matplotlib unavailable ({exc}); CSVs still written")
        return 0

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    (ax_al, ax_loss), (ax_al_zoom, ax_val) = axes

    W = 200
    for arm, label in ARMS:
        d = data[arm]
        gs = d["global_step"]
        ax_al.plot(gs, d["accept_len"], color=COLORS[arm], lw=0.4, alpha=0.18)
        s = smooth(d["accept_len"], W)
        ax_al.plot(gs[W - 1:], s, color=COLORS[arm], lw=1.9, label=f"{label}")

        ax_loss.plot(gs, d["loss"], color=COLORS[arm], lw=0.4, alpha=0.18)
        sl = smooth(d["loss"], W)
        ax_loss.plot(gs[W - 1:], sl, color=COLORS[arm], lw=1.9, label=f"{label}")

        # epoch-2 zoom
        m = d["epoch"] == 2
        ax_al_zoom.plot(gs[m], d["accept_len"][m], color=COLORS[arm], lw=0.5, alpha=0.2)
        sz = smooth(d["accept_len"][m], W)
        ax_al_zoom.plot(gs[m][W - 1:], sz, color=COLORS[arm], lw=2.0, label=f"{label}")

        vals = read_vals(arm)
        eps = sorted(vals)
        ax_val.plot(eps, [vals[e]["accept_len_epoch"] for e in eps],
                    "o-", color=COLORS[arm], lw=2, ms=7, label=f"{label}")

    for ax in (ax_al, ax_loss, ax_al_zoom, ax_val):
        ax.grid(alpha=0.3)
        ax.legend()
    ax_al.set(title=f"train/accept_len  (faint=raw, bold={W}-step mean)",
              xlabel="global_step (micro-batch)", ylabel="accept_len")
    ax_loss.set(title=f"train/loss  (faint=raw, bold={W}-step mean)",
                xlabel="global_step (micro-batch)", ylabel="loss")
    ax_al_zoom.set(title=f"epoch 2 zoom -- train/accept_len ({W}-step mean)",
                   xlabel="global_step (micro-batch)", ylabel="accept_len")
    ax_val.set(title="val/accept_len_epoch", xlabel="epoch", ylabel="accept_len")
    ax_val.set_xticks([0, 1, 2])

    fig.suptitle("DSpark draft attention: MHA vs Qwen3-4B native GQA vs MLA r=512 "
                 "(single card, accum=12, full attention, 50k x 3 epochs)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = os.path.join(OUT, "attn_compare.png")
    fig.savefig(png, dpi=140)
    print(f"\nwrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
