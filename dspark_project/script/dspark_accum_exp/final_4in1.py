#!/usr/bin/env python3
"""All 4 completed runs on a single figure: loss & accept_len vs step.
Style: light = per-step raw, dark = sliding mean; dots = per-epoch validation.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/home/y50063564/processed_data/dspark_data/dspark_accum")
OUT = ROOT / "analysis" / "final_all4_loss_acceptlen.png"
START_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] INFO\s+(train|val)/")
CONT_RE = re.compile(r"^\s+\S+=")
KV_RE = re.compile(r"([\w./]+)=([^,\s]+)")


def parse(log: Path):
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
            va.append({"epoch": int(kv["epoch"]), "loss": float(kv["val/loss_epoch"]),
                       "al": float(kv["val/accept_len_epoch"])})
    dft = pd.DataFrame(tr).sort_values("step").drop_duplicates("step", keep="last")
    dfv = pd.DataFrame(va).sort_values("epoch").drop_duplicates("epoch", keep="last")
    return dft, dfv


RUNS = {
    "accum=1 (per micro-batch)": ("accum1", ROOT / "accum1/logs/train_accum1_*.log", "#7f7f7f"),
    "accum=12 (every 12)":       ("accum12", ROOT / "accum12/logs/train_accum12_*.log", "#1f77b4"),
    "diff-ctx @layer0, accum12": ("diff_l0_ctx_a12", ROOT / "diff_l0_ctx_a12/logs/train_diff_l0_ctx_a12_*.log", "#d62728"),
    "diff-ctx @layer3, accum12": ("diff_l3_ctx_a12", ROOT / "diff_l3_ctx_a12/logs/train_diff_l3_ctx_a12_*.log", "#2ca02c"),
}
WIN = 1000  # SMA window in micro-batches (log_freq=20 -> WIN/20 logged pts)

data = {}
for label, (base, globp, color) in RUNS.items():
    log = sorted(Path(globp.parent).glob(globp.name))[-1]
    tr, va = parse(log)
    data[label] = (tr, va, color)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
for label, (tr, va, color) in data.items():
    win_pts = max(3, min(int(WIN / 20), len(tr) - 1))
    for ax, col in ((ax1, "loss"), (ax2, "al")):
        ax.plot(tr["step"], tr[col], color=color, lw=0.8, alpha=0.28)
        ax.plot(tr["step"], tr[col].rolling(win_pts, center=True, min_periods=1).mean(),
                color=color, lw=2.4, label=label)
    if len(va):
        xs = {int(e): tr.loc[tr["epoch"] == e, "step"].max() for e in va["epoch"]}
        ax1.plot([xs[e] for e in va["epoch"]], va["loss"], marker="o", ls="", ms=7, color=color)
        ax2.plot([xs[e] for e in va["epoch"]], va["al"], marker="o", ls="", ms=7, color=color)

ax2.set_xlabel("global step (micro-batch)")
ax1.set_ylabel("loss");            ax1.set_title("train loss (light=per-step, dark=SMA, dot=val)")
ax2.set_ylabel("accept_len");      ax2.set_title("train accept_len (light=per-step, dark=SMA, dot=val)")
for ax in (ax1, ax2):
    ax.grid(alpha=0.25); ax.legend(fontsize=9, ncol=2)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
