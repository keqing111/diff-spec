#!/usr/bin/env python3
"""Analyse the DL_PROFILE JSONL output to localise the slow rank / worker.

Reads /home/y50063564/dataloader_diag/raw/rank*/**.jsonl and answers:
  L1  per-rank trainer wait for a batch (load_ms)
  L2  per-worker per-sample time (wall_ms) -> is the slowness a fixed worker?
  L3  collate time
  L4  cross-rank arrival spread at the forward entry (t_batch, same host clock),
      and which rank is the latest / how the spread relates to the slow steps.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

RAW = os.environ.get("DL_RAW", "/home/y50063564/dataloader_diag/raw")


def load(kind_files):
    rows = []
    for f in kind_files:
        for line in open(f, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(rows)


def pct(s, q):
    return float(np.percentile(s, q)) if len(s) else float("nan")


def main():
    files = glob.glob(f"{RAW}/rank*/**/*.jsonl", recursive=True)
    print(f"profile files: {len(files)}")
    if not files:
        print("no data yet"); return
    allrows = load(files)
    print("events:", allrows["kind"].value_counts().to_dict())

    # ---------------- L1: per-rank main-process wait ----------------
    load_df = allrows[allrows["kind"] == "load"]
    if len(load_df):
        print("\n=== L1 trainer wait for batch (load_ms) by rank ===")
        g = load_df.groupby("rank")["load_ms"]
        print(g.agg(n="count", p50=lambda s: pct(s, 50), p90=lambda s: pct(s, 90),
                    p99=lambda s: pct(s, 99), max="max").to_string())
        for thr in (50, 200, 1000):
            print(f"  load_ms>{thr}ms counts by rank:",
                  load_df[load_df.load_ms > thr]["rank"].value_counts().sort_index().to_dict())

    # ---------------- L2: per-worker per-sample ----------------
    smp = allrows[allrows["kind"] == "sample"]
    if len(smp):
        smp = smp.copy()
        smp["wid"] = smp["rank"].astype(str) + "/w" + smp["worker"].astype(str)
        print("\n=== L2 per-sample wall_ms by rank/worker ===")
        g = smp.groupby("wid")["wall_ms"]
        tab = g.agg(n="count", p50=lambda s: pct(s, 50), p90=lambda s: pct(s, 90),
                    p99=lambda s: pct(s, 99), max="max")
        print(tab.sort_values("p90", ascending=False).to_string())
        print("  (worker ids are stable per rank; a consistently slow w* = fixed worker)")

    # ---------------- L3: collate ----------------
    col = allrows[allrows["kind"] == "collate"]
    if len(col):
        print("\n=== L3 collate wall_ms ===")
        print(col.groupby("rank")["wall_ms"].agg(
            n="count", p50=lambda s: pct(s, 50), p90=lambda s: pct(s, 90), max="max").to_string())

    # ---------------- L4: cross-rank arrival spread ----------------
    tl = allrows[allrows["kind"] == "timeline"]
    if len(tl):
        tl = tl.sort_values("t_batch")
        # align by global_step: spread of t_batch across ranks for the same step
        piv = tl.pivot_table(index="global_step", columns="rank", values="t_batch", aggfunc="min")
        if piv.shape[1] > 1:
            spread = (piv.max(axis=1) - piv.min(axis=1)) * 1000  # ms
            late_rank = piv.idxmax(axis=1)
            print("\n=== L4 forward-entry (t_batch) cross-rank spread per step ===")
            print(f"  steps={len(spread)}  spread_ms: p50={pct(spread,50):.1f} p90={pct(spread,90):.1f} "
                  f"p99={pct(spread,99):.1f} max={spread.max():.1f}")
            print("  which rank is LATEST most often:",
                  late_rank.value_counts().to_dict())
            # do slow steps (big spread) coincide with big load on the late rank?
            merged = pd.DataFrame({"spread": spread, "late": late_rank})
            late_load = []
            for gs, r in merged["late"].items():
                sub = load_df[(load_df.global_step == gs) & (load_df["rank"] == r)]["load_ms"]
                late_load.append(float(sub.iloc[0]) if len(sub) else float("nan"))
            merged["late_rank_load_ms"] = late_load
            for thr in (200, 1000):
                sel = merged[merged.spread > thr]
                if len(sel):
                    print(f"  spread>{thr}ms: n={len(sel)}, median load of the late rank = "
                          f"{sel['late_rank_load_ms'].median():.1f} ms")
            # per-rank fwd duration & end-time skew
            for a, b in (("t_batch", "t_fwd"), ("t_fwd", "t_end")):
                piv2 = tl.pivot_table(index="global_step", columns="rank", values=a, aggfunc="min")
                piv3 = tl.pivot_table(index="global_step", columns="rank", values=b, aggfunc="min")
                dur = (piv3 - piv2) * 1000
                print(f"  per-rank {a}->{b} median ms:",
                      {int(c): round(float(dur[c].median()), 1) for c in dur.columns})

    # ---------------- save tidy csv ----------------
    out = os.environ.get("DL_OUT", "/home/y50063564/dataloader_diag/analysis")
    os.makedirs(out, exist_ok=True)
    if len(load_df):
        load_df.to_csv(f"{out}/l1_load.csv", index=False)
    if len(smp):
        smp.to_csv(f"{out}/l2_samples.csv", index=False)
    if len(col):
        col.to_csv(f"{out}/l3_collate.csv", index=False)
    if len(tl):
        tl.to_csv(f"{out}/l4_timeline.csv", index=False)
    print(f"\nwrote csvs under {out}/")


if __name__ == "__main__":
    main()
