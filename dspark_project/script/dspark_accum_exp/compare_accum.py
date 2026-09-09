#!/usr/bin/env python3
"""Parse the two dspark_accum train logs and plot accept_len / loss.

Usage:
    python compare_accum.py \
        --accum1 <log1> --accum12 <log12> \
        [-o <out_dir>] [--spe <steps_per_epoch>]

Reads the console metric-logger format produced by speculators (same block
format as processed_data/.../epoch_comp/reconstruct_metrics.py):
  * a block starts with "[HH:MM:SS] INFO  train/..." or "val/..._epoch=..."
  * continuation lines are indented  key=value,
Outputs <out_dir>/accum_vs_accum_*.png and *_metrics.csv, plus a console
per-epoch mean table.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FLOAT_KEYS = {
    "train/loss", "train/ce_loss", "train/tv_loss", "train/accept_rate",
    "train/accept_len", "train/full_acc",
    *(f"train/position_{k}_acc" for k in range(1, 8)),
    "lr/Muon", "lr/AdamW",
    "val/loss_epoch", "val/ce_loss_epoch", "val/tv_loss_epoch",
    "val/accept_rate_epoch", "val/accept_len_epoch", "val/full_acc_epoch",
    *(f"val/position_{k}_acc_epoch" for k in range(1, 8)),
}
INT_KEYS = {"global_step", "epoch"}

START_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] INFO\s+(train|val)/")
CONT_RE = re.compile(r"^\s+\S+=")
KV_RE = re.compile(r"([\w./]+)=([^,\s]+)")


def parse_log(log_path: str) -> list[dict]:
    blocks: list[dict] = []
    current: dict | None = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = START_RE.match(line)
            if m:
                if current:
                    blocks.append(current)
                current = {"_ts": m.group(1), "_type": m.group(2), "_text": line}
                continue
            if current is not None and CONT_RE.match(line):
                current["_text"] += line
                continue
            if current is not None:
                blocks.append(current)
                current = None
    if current:
        blocks.append(current)

    records: list[dict] = []
    for b in blocks:
        pairs = dict(KV_RE.findall(b["_text"]))
        rec: dict = {"_ts": b["_ts"], "_type": b["_type"]}
        for k, v in pairs.items():
            if k in INT_KEYS:
                try:
                    rec[k] = int(v)
                except ValueError:
                    rec[k] = v
            elif k in FLOAT_KEYS:
                try:
                    rec[k] = float(v)
                except ValueError:
                    rec[k] = v
            else:
                rec[k] = v
        if b["_type"] == "train" and "global_step" in rec:
            records.append(rec)
        elif b["_type"] == "val" and "epoch" in rec:
            records.append(rec)
    return records


def to_frame(records: list[dict], kind: str) -> pd.DataFrame:
    rows = [r for r in records if r["_type"] == kind]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(
        "global_step" if kind == "train" else "epoch"
    ).drop_duplicates(
        "global_step" if kind == "train" else "epoch", keep="last"
    ).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accum1", required=True, type=Path)
    ap.add_argument("--accum12", required=True, type=Path)
    ap.add_argument("-o", "--out", type=Path,
                    default=Path("/home/y50063564/processed_data/dspark_data/dspark_accum/analysis"))
    ap.add_argument("--spe", type=float, default=None,
                    help="steps-per-epoch (optional; inferred from epoch jumps if absent)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = {
        "accum=1 (per micro-batch)": args.accum1,
        "accum=12 (every 12 micro-batches)": args.accum12,
    }
    colors = {"accum=1 (per micro-batch)": "#1f77b4",
              "accum=12 (every 12 micro-batches)": "#d62728"}

    frames = {}
    for name, p in runs.items():
        recs = parse_log(str(p))
        tr = to_frame(recs, "train")
        va = to_frame(recs, "val")
        print(f"[{name}] train records={len(tr)}  val records={len(va)}")
        if not len(tr):
            raise SystemExit(f"no train metrics parsed from {p}")
        # steps/epoch from epoch jumps on global_step
        starts = []
        prev = None
        for _, r in tr.iterrows():
            if r["epoch"] != prev:
                starts.append(int(r["global_step"]))
                prev = r["epoch"]
        if len(starts) >= 2:
            spe = float(np.mean([b - a for a, b in zip(starts, starts[1:])]))
        elif args.spe:
            spe = args.spe
        else:
            spe = float("nan")
        print(f"    steps/epoch ~ {spe:.1f}  (epochs seen: {sorted(tr['epoch'].unique())})")
        tr["fepoch"] = tr["global_step"] / spe if spe == spe else tr["epoch"].astype(float)
        tr.to_csv(args.out / f"train_metrics_{name.split()[0].replace('=','_')}.csv", index=False)
        va.to_csv(args.out / f"val_metrics_{name.split()[0].replace('=','_')}.csv", index=False)
        frames[name] = (tr, va)
        print("    per-epoch train means:")
        for e, g in tr.groupby("epoch"):
            print(f"      epoch {int(e)}: n={len(g):5d}  accept_len={g['train/accept_len'].mean():.4f}  "
                  f"loss={g['train/loss'].mean():.4f}")
        if len(va):
            print("    per-epoch val:")
            for _, r in va.iterrows():
                print(f"      val epoch {int(r['epoch'])}: accept_len_epoch={r.get('val/accept_len_epoch', float('nan')):.4f}  "
                      f"loss_epoch={r.get('val/loss_epoch', float('nan')):.4f}")

    # ---- Plots ----
    GRID, SMOOTH = 0.02, 21

    def smooth(df, col):
        x = df["fepoch"].values if "fepoch" in df else df["global_step"].values
        y = df[col].values
        bins = np.floor(x / GRID) * GRID
        g = pd.DataFrame({"b": bins, "y": y}).groupby("b").mean().sort_index()
        g["s"] = g["y"].rolling(SMOOTH, center=True, min_periods=1).mean()
        return g

    # x = fractional epoch
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for name, (tr, va) in frames.items():
        c = colors[name]
        for ax, col in ((ax1, "train/accept_len"), (ax2, "train/loss")):
            g = smooth(tr, col)
            ax.plot(g["b"], g["s"], color=c, lw=2.2, label=name)
            if len(va):
                ax.plot(va["epoch"], va["val/accept_len_epoch" if col.startswith("accept")
                                     else "val/loss_epoch"],
                        marker="o", ls="", ms=7, color=c)
    for ax, ylab in ((ax1, "accept_len"), (ax2, "loss")):
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    ax1.set_title("accept_len vs epoch (train line + val points)")
    ax2.set_title("loss vs epoch (train line + val points)")
    fig.tight_layout()
    fig.savefig(args.out / "accum_accept_len_loss_vs_epoch.png", dpi=150)
    print(f"wrote {args.out / 'accum_accept_len_loss_vs_epoch.png'}")

    # x = global micro-batch (identical data consumption axis)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for name, (tr, va) in frames.items():
        c = colors[name]
        for ax, col in ((ax1, "train/accept_len"), (ax2, "train/loss")):
            g = smooth(tr, col)
            ax.plot(g.index * GRID, g["s"], color=c, lw=1.8, label=name)
    for ax, ylab in ((ax1, "accept_len"), (ax2, "loss")):
        ax.set_xlabel("global micro-batch (same data per point in both runs)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    ax1.set_title("accept_len vs micro-batch")
    ax2.set_title("loss vs micro-batch")
    fig2.tight_layout()
    fig2.savefig(args.out / "accum_accept_len_loss_vs_step.png", dpi=150)
    print(f"wrote {args.out / 'accum_accept_len_loss_vs_step.png'}")


if __name__ == "__main__":
    main()
