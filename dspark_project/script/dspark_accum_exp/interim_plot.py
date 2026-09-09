#!/usr/bin/env python3
"""Interim comparison plot: raw (light) + sliding-mean (dark) of train loss and
accept_len vs global step, for accum1 vs accum12. Re-runnable as more epochs land.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

START_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] INFO\s+(train|val)/")
CONT_RE = re.compile(r"^\s+\S+=")
KV_RE = re.compile(r"([\w./]+)=([^,\s]+)")


def parse_train(log: Path):
    seq = []
    cur = None
    for line in open(log, encoding="utf-8", errors="replace"):
        m = START_RE.match(line)
        if m:
            if cur:
                seq.append(cur)
            cur = {"text": line, "type": m.group(2)}
            continue
        if cur is not None and CONT_RE.match(line):
            cur["text"] += line
            continue
        if cur is not None:
            seq.append(cur)
            cur = None
    if cur:
        seq.append(cur)
    rows = []
    for r in seq:
        if r["type"] != "train":
            continue
        kv = dict(KV_RE.findall(r["text"]))
        if "global_step" not in kv:
            continue
        rows.append({
            "step": int(kv["global_step"]),
            "epoch": int(kv.get("epoch", -1)),
            "loss": float(kv.get("train/loss", float("nan"))),
            "accept_len": float(kv.get("train/accept_len", float("nan"))),
        })
    return pd.DataFrame(rows).sort_values("step").drop_duplicates("step", keep="last")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accum1", required=True, type=Path)
    ap.add_argument("--accum12", required=True, type=Path)
    ap.add_argument("-o", type=Path, default=Path(
        "/home/y50063564/processed_data/dspark_data/dspark_accum/analysis/interim_epoch1.png"))
    ap.add_argument("--win", type=int, default=1000,
                    help="sliding-mean window in micro-batches")
    ap.add_argument("--min-window-per-epoch", type=float, default=0.2,
                    help="window must be <= this fraction of the run so far")
    args = ap.parse_args()
    args.o.parent.mkdir(parents=True, exist_ok=True)

    df1, df2 = parse_train(args.accum1), parse_train(args.accum12)
    dfs = [("accum=1 (per micro-batch)", df1), ("accum=12 (every 12)", df2)]
    colors = {"accum=1 (per micro-batch)": "#1f77b4",
              "accum=12 (every 12)": "#d62728"}

    for name, d in dfs:
        starts = []
        prev = None
        for _, r in d.iterrows():
            if r["epoch"] != prev:
                starts.append(int(r["step"]))
                prev = r["epoch"]
        spe = None
        if len(starts) >= 2:
            spe = (starts[-1] - starts[0]) / (len(starts) - 1)
        prog = f" (~{d['step'].max()/spe:.2f} epochs)" if spe else ""
        print(f"[{name}] train samples={len(d)} step {d['step'].min()}..{d['step'].max()}"
              f"{prog} (loss {d['loss'].min():.3f}..{d['loss'].max():.3f}, "
              f"accept_len {d['accept_len'].min():.3f}..{d['accept_len'].max():.3f})")
        # sliding mean over a fixed micro-batch window, computed on logged points
        for col in ("loss", "accept_len"):
            win_pts = max(3, int(args.win / 20))       # log_freq=20
            win_pts = min(win_pts, len(d) - 1)
            d[f"{col}_sma"] = d[col].rolling(win_pts, center=True, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for name, d in dfs:
        c = colors[name]
        for ax, col in ((ax1, "loss"), (ax2, "accept_len")):
            ax.plot(d["step"], d[col], color=c, lw=0.8, alpha=0.30, label=None)
            ax.plot(d["step"], d[f"{col}_sma"], color=c, lw=2.4,
                    label=f"{name} (SMA)")

    ax2.set_xlabel("global step (micro-batch)")
    ax1.set_ylabel("train/loss");      ax1.set_title("train loss (light=per-step, dark=SMA)")
    ax2.set_ylabel("train/accept_len"); ax2.set_title("train accept_len (light=per-step, dark=SMA)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.o, dpi=150)
    print(f"wrote {args.o}")


if __name__ == "__main__":
    main()
