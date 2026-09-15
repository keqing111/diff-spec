#!/usr/bin/env python3
"""ctx-only 70w (dp4+accum3) vs the dp12 700k baseline.

Both runs see the same data: open_perfectblend_qwen3_4b_700k, 3 epochs,
train ratio 0.9 (630k documents per epoch), sliding_window 2048 on all 5 draft
layers. They differ in how the effective batch is built:

  baseline : dp12, 1 micro-batch per rank per step -> 12 packs / optimizer step
  ctx-only : dp4 + accum=3                         -> 12 packs / optimizer step

so a shared x-axis is "epoch fraction" (both consume the same documents per
epoch); `global_step` is NOT shared because it counts rank-local micro-batches,
which differ between the two (9410 vs 9637 per epoch).

Run:  python3 ctx70w_vs_dp12.py
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

CTX_ROOT = "/home/y50063564/processed_data/dspark_data/dspark_accum/dp4_accum3_ctx70w"
BASE_ROOT = "/home/y50063564/processed_data/archive_dspark_20260904/eval/baseline_qwen3_4b_70w"
OUT = "/home/y50063564/dspark_project/script/dspark_accum_exp/results"

# Merge order matters: later files win for a given global_step (later attempt
# of the same step range is the one that actually advanced training).
CTX_LOGS = [
    "train_dp4_accum3_ctx70w_20260910_015955.log",
    "train_dp4_accum3_ctx70w_20260910_073303.log",
    "train_dp4_accum3_ctx70w_20260910_082546.log",
    "train_dp4_accum3_ctx70w_20260911_173929.log",
]
FIELDS = ("loss", "accept_len", "accept_rate")


def parse_ctx() -> dict[str, np.ndarray]:
    """Merge the ctx-only logs into one per-global_step table."""
    recs: dict[int, dict] = {}
    for name in CTX_LOGS:
        path = os.path.join(CTX_ROOT, "logs", name)
        if not os.path.isfile(path):
            print(f"  !! missing {path}")
            continue
        with open(path, "r", errors="replace") as fh:
            flat = re.sub(r"\s+", " ", fh.read())
        n_new = 0
        for chunk in flat.split("global_step=")[1:]:
            m = re.match(r"(\d+)", chunk)
            if not m:
                continue
            step = int(m.group(1))
            vals = {}
            for k in FIELDS:
                hits = re.findall(rf"train/{k}=([0-9.eE+-]+)", chunk)
                if not hits:
                    vals = None
                    break
                vals[k] = float(hits[-1])
            if vals is None:
                continue
            ep = re.findall(r"epoch=(\d+)", chunk)
            vals["epoch"] = int(ep[-1]) if ep else -1
            recs[step] = vals
            n_new += 1
        print(f"  {name}: {n_new} records")
    steps = np.array(sorted(recs), dtype=np.int64)
    out = {"global_step": steps}
    for k in (*FIELDS, "epoch"):
        out[k] = np.array([recs[s][k] for s in steps], dtype=np.float64)
    out["epoch"] = out["epoch"].astype(np.int64)
    return out


def read_baseline() -> dict[str, np.ndarray]:
    path = os.path.join(BASE_ROOT, "train_metrics.csv")
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in fh if ln.strip()]
    idx = {name: i for i, name in enumerate(header)}
    cols = {k: [] for k in (*FIELDS, "epoch", "global_step")}
    for r in rows:
        cols["loss"].append(float(r[idx["train/loss"]]))
        cols["accept_len"].append(float(r[idx["train/accept_len"]]))
        cols["accept_rate"].append(float(r[idx["train/accept_rate"]]))
        cols["epoch"].append(int(float(r[idx["epoch"]])))
        cols["global_step"].append(int(float(r[idx["global_step"]])))
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def read_val(root: str, arm_sub: str) -> dict[int, dict]:
    import json

    res = {}
    for d in sorted(glob.glob(os.path.join(root, arm_sub, "checkpoints", "[0-9]*"))):
        f = os.path.join(d, "val_metrics.json")
        if os.path.isfile(f):
            with open(f) as fh:
                res[int(os.path.basename(d))] = json.load(fh)
    return res


def read_val_csv(root: str) -> dict[int, dict]:
    """The dp12 baseline archive keeps its val rows in val_metrics.csv."""
    path = os.path.join(root, "val_metrics.csv")
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in fh if ln.strip()]
    idx = {name: i for i, name in enumerate(header)}
    res = {}
    for r in rows:
        e = int(float(r[idx["epoch"]]))
        res[e] = {
            "accept_len_epoch": float(r[idx["val/accept_len_epoch"]]),
            "accept_rate_epoch": float(r[idx["val/accept_rate_epoch"]]),
            "loss_epoch": float(r[idx["val/loss_epoch"]]),
        }
    return res


def to_epoch_frac(d: dict[str, np.ndarray]) -> np.ndarray:
    """epoch index -> continuous epoch fraction using the step span of each epoch."""
    gs, ep = d["global_step"], d["epoch"]
    frac = np.zeros(gs.size, dtype=np.float64)
    for e in np.unique(ep):
        m = ep == e
        lo, hi = gs[m].min(), gs[m].max()
        span = max(hi - lo, 1)
        frac[m] = e + (gs[m] - lo) / span
    return frac


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    print("parsing ctx-only logs:")
    ctx = parse_ctx()
    base = read_baseline()
    print(f"\nctx-only : {ctx['global_step'].size} steps, "
          f"gs {ctx['global_step'].min()}..{ctx['global_step'].max()}, "
          f"epochs {sorted(set(ctx['epoch'].tolist()))}")
    print(f"baseline : {base['global_step'].size} steps, "
          f"gs {base['global_step'].min()}..{base['global_step'].max()}, "
          f"epochs {sorted(set(base['epoch'].tolist()))}")

    ctx["frac"] = to_epoch_frac(ctx)
    base["frac"] = to_epoch_frac(base)

    # ---- per-epoch train means -------------------------------------------
    print("\n" + "=" * 84)
    print(f"{'epoch':>5} | {'baseline(dP12) accept_len':>26} | {'ctx-only(dP4+a3) accept_len':>28} | {'Δ':>8}")
    print("=" * 84)
    for e in range(3):
        b = base["accept_len"][base["epoch"] == e].mean()
        c = ctx["accept_len"][ctx["epoch"] == e].mean()
        print(f"{e:>5} | {b:>26.4f} | {c:>28.4f} | {c - b:>+8.4f}")

    print("\n" + "=" * 84)
    print(f"{'epoch':>5} | {'baseline train/loss':>26} | {'ctx-only train/loss':>28} | {'Δ':>8}")
    print("=" * 84)
    for e in range(3):
        b = base["loss"][base["epoch"] == e].mean()
        c = ctx["loss"][ctx["epoch"] == e].mean()
        print(f"{e:>5} | {b:>26.5f} | {c:>28.5f} | {c - b:>+8.5f}")

    # ---- val -------------------------------------------------------------
    bval = read_val_csv(BASE_ROOT)
    cval = read_val(CTX_ROOT, ".")
    if bval and cval:
        print("\n" + "=" * 84)
        print("validation accept_len (the metric that matters for spec decoding)")
        print("=" * 84)
        print(f"{'epoch':>5} | {'baseline':>10} | {'ctx-only':>10} | {'Δ':>8}")
        for e in sorted(set(bval) & set(cval)):
            b = bval[e]["accept_len_epoch"]
            c = cval[e]["accept_len_epoch"]
            print(f"{e:>5} | {b:>10.4f} | {c:>10.4f} | {c - b:>+8.4f}")

    # ---- csv -------------------------------------------------------------
    for name, d in (("baseline_dp12", base), ("ctxonly_dp4a3", ctx)):
        p = os.path.join(OUT, f"{name}.csv")
        with open(p, "w") as fh:
            fh.write("global_step,epoch,epoch_frac,train/loss,train/accept_len,train/accept_rate\n")
            for i in range(d["global_step"].size):
                fh.write(f"{int(d['global_step'][i])},{int(d['epoch'][i])},"
                         f"{d['frac'][i]:.6f},{d['loss'][i]},{d['accept_len'][i]},"
                         f"{d['accept_rate'][i]}\n")
        print(f"wrote {p}")

    # ---- plot ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"\n!! matplotlib unavailable ({exc})")
        return 0

    CB, CC = "#1f77b4", "#d62728"
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    ax_a, ax_l, ax_v = axes
    W = 200

    for ax, key, title, ylab in (
        (ax_a, "accept_len", "train/accept_len", "accept_len"),
        (ax_l, "loss", "train/loss", "loss"),
    ):
        ax.plot(base["frac"], base[key], color=CB, lw=0.35, alpha=0.15)
        ax.plot(ctx["frac"], ctx[key], color=CC, lw=0.35, alpha=0.15)
        ax.plot(base["frac"][W - 1:], smooth(base[key], W), color=CB, lw=2.0,
                label="dp12 baseline (700k, dp12, srv dp4)")
        ax.plot(ctx["frac"][W - 1:], smooth(ctx[key], W), color=CC, lw=2.0,
                label="ctx-only @layer0 (700k, dp4+accum3)")
        for e in (1, 2):
            ax.axvline(e, color="gray", ls=":", lw=1)
        ax.set(title=f"{title}  (faint=raw, bold={W}-step mean)",
               xlabel="epoch", ylabel=ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    if bval and cval:
        eps = sorted(set(bval) & set(cval))
        ax_v.plot(eps, [bval[e]["accept_len_epoch"] for e in eps], "o-",
                  color=CB, lw=2, ms=8, label="dp12 baseline")
        ax_v.plot(eps, [cval[e]["accept_len_epoch"] for e in eps], "s-",
                  color=CC, lw=2, ms=8, label="ctx-only @layer0")
        for e in eps:
            ax_v.annotate(f"{cval[e]['accept_len_epoch'] - bval[e]['accept_len_epoch']:+.3f}",
                          (e, cval[e]["accept_len_epoch"]), textcoords="offset points",
                          xytext=(0, 9), ha="center", fontsize=9, color="#444")
    ax_v.set(title="val/accept_len_epoch", xlabel="epoch", ylabel="accept_len")
    ax_v.set_xticks([0, 1, 2])
    ax_v.grid(alpha=0.3)
    if bval and cval:
        ax_v.legend()

    fig.suptitle("700k: ctx-only @draft layer 0 (dp4+accum3) vs dp12 baseline — "
                 "same 630k docs/epoch, sliding_window 2048 in both", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = os.path.join(OUT, "ctx70w_vs_dp12.png")
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
