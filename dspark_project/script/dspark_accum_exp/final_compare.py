#!/usr/bin/env python3
"""Final comparison figures for the completed runs.

Style = user-approved interim style: light = per-step raw, dark = sliding mean,
x = global step (micro-batch). Two figures are produced:
  A) plain accum=1 vs plain accum=12
  B) plain accum=12 (reference) vs diff-ctx@l0-accum12 vs diff-ctx@l3-accum12
Each panel additionally marks per-epoch validation points (val/loss_epoch,
val/accept_len_epoch) at the epoch-end step.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/home/y50063564/processed_data/dspark_data/dspark_accum")
ANALYSIS = ROOT / "analysis"
START_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] INFO\s+(train|val)/")
CONT_RE = re.compile(r"^\s+\S+=")
KV_RE = re.compile(r"([\w./]+)=([^,\s]+)")


def parse(log: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks, cur = [], None
    for line in open(log, encoding="utf-8", errors="replace"):
        m = START_RE.match(line)
        if m:
            if cur:
                blocks.append(cur)
            cur = {"text": line, "type": m.group(2)}
            continue
        if cur is not None and CONT_RE.match(line):
            cur["text"] += line
            continue
        if cur is not None:
            blocks.append(cur)
            cur = None
    if cur:
        blocks.append(cur)
    tr, va = [], []
    for b in blocks:
        kv = dict(KV_RE.findall(b["text"]))
        if b["type"] == "train" and "global_step" in kv and "train/loss" in kv:
            tr.append({"step": int(kv["global_step"]), "epoch": int(kv.get("epoch", -1)),
                       "loss": float(kv["train/loss"]), "al": float(kv["train/accept_len"])})
        elif b["type"] == "val" and "epoch" in kv and "val/loss_epoch" in kv:
            va.append({"epoch": int(kv["epoch"]),
                       "loss": float(kv["val/loss_epoch"]), "al": float(kv["val/accept_len_epoch"])})
    dft = pd.DataFrame(tr).sort_values("step").drop_duplicates("step", keep="last")
    dfv = pd.DataFrame(va).sort_values("epoch").drop_duplicates("epoch", keep="last")
    return dft, dfv


RUNS = {
    "accum1":        (ROOT / "accum1/logs", "accum1"),
    "accum12":       (ROOT / "accum12/logs", "accum12"),
    "diff-l0-ctx-a12": (ROOT / "diff_l0_ctx_a12/logs", "diff_l0_ctx_a12"),
    "diff-l3-ctx-a12": (ROOT / "diff_l3_ctx_a12/logs", "diff_l3_ctx_a12"),
}
COLORS = {"accum1": "#1f77b4", "accum12": "#2ca02c",
          "diff-l0-ctx-a12": "#d62728", "diff-l3-ctx-a12": "#9467bd"}

data = {}
for name, (ld, base) in RUNS.items():
    log = sorted(ld.glob(f"train_{base}_*.log"))[-1]
    tr, va = parse(log)
    data[name] = (tr, va)
    spe = None
    if len(va):
        e = va["epoch"].iloc[-1]
        end_step = tr.loc[tr["epoch"] == e, "step"].max()
        spe = end_step / (e + 1)
    print(f"[{name}] {log.name}\n  train pts={len(tr)} step {tr.step.min()}..{tr.step.max()}"
          f"  val pts={len(va)}"
          + (f"  spe~{spe:.0f}" if spe else ""))
    if len(va):
        for _, v in va.iterrows():
            print(f"    val epoch {int(v['epoch'])}: loss={v['loss']:.4f}  accept_len={v['al']:.4f}")
    print(f"  per-epoch TRAIN mean:")
    for e, g in tr.groupby("epoch"):
        print(f"    epoch {int(e)}: loss={g.loss.mean():.4f}  accept_len={g.al.mean():.4f}")

ANALYSIS.mkdir(parents=True, exist_ok=True)
WIN = 1000  # sliding window in micro-batches (log_freq=20 -> win_pts = WIN/20)

def val_x(dfv, dft):
    """x (step) at which each epoch's validation ran = last train step of that epoch."""
    xs = {}
    for _, v in dfv.iterrows():
        e = int(v["epoch"])
        sub = dft[dft["epoch"] == e]
        xs[e] = sub["step"].max() if len(sub) else float("nan")
    return xs

def draw(figpath, labels, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 11), sharex=True)
    for name in labels:
        tr, va = data[name]
        c = COLORS[name]
        win_pts = max(3, min(int(WIN / 20), len(tr) - 1))
        for ax, (col, ylab) in ((ax1, ("loss", "loss")), (ax2, ("al", "accept_len"))):
            ax.plot(tr["step"], tr[col], color=c, lw=0.8, alpha=0.30)
            ax.plot(tr["step"], tr[col].rolling(win_pts, center=True, min_periods=1).mean(),
                    color=c, lw=2.2, label=f"{name}")
        if len(va):
            xs = val_x(va, tr)
            ax1.plot([xs[e] for e in va["epoch"]], va["loss"], marker="o", ls="", ms=6, color=c)
            ax2.plot([xs[e] for e in va["epoch"]], va["al"], marker="o", ls="", ms=6, color=c)
    ax2.set_xlabel("global step (micro-batch)")
    ax1.set_ylabel("train loss (line) / val loss (dot)"); ax1.set_title(f"loss  —  {title}")
    ax2.set_ylabel("accept_len (line) / val (dot)"); ax2.set_title(f"accept_len  —  {title}")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figpath, dpi=150)
    print(f"wrote {figpath}")


draw(ANALYSIS / "final_accum1_vs_accum12.png", ["accum1", "accum12"],
     "plain: accum=1 (every micro-batch) vs accum=12 (every 12)  [light=step, dark=SMA]")
draw(ANALYSIS / "final_diff_ctx_vs_accum12.png", ["accum12", "diff-l0-ctx-a12", "diff-l3-ctx-a12"],
     "accum=12: plain vs diff-ctx-only @layer0 vs @layer3  [light=step, dark=SMA]")
print("done.")
