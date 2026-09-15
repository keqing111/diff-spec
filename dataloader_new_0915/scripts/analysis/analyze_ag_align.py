#!/usr/bin/env python3
"""Are the ranks aligned when they enter FSDP collectives?

Two candidate alignments for comparing ranks:
  (a) by global_step label (what everyone assumed)
  (b) by collective ordinal k (HCCL matches collectives in issue order, so the
      k-th all-gather of rank r pairs with the k-th of rank r')

We compare both: if (b) yields a much smaller arrival spread and the gs labels
differ at (b)'s pairing, then per-gs alignment was a labeling artefact and any
"ready skew = 1 step" conclusion must be re-derived from (b).
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else "/home/y50063564/dataloader_diag/exp/E1_ref_w2_pf4"
IDX = int(os.environ.get("AG_IDX", "1"))          # which all-gather in the forward
OUT = os.environ.get("DL_OUT", "/home/y50063564/dataloader_diag/analysis/E1_align")


def load_ag(raw):
    per = {}
    for f in glob.glob(f"{raw}/rank*/ag_rank*.jsonl"):
        r = int(re.search(r"/rank(\d+)/", f).group(1))
        rows = []
        for line in open(f, encoding="utf-8", errors="replace"):
            d = json.loads(line)
            if d.get("idx_in_fwd") == IDX:
                rows.append((d["t_begin"], d["t_end"], int(d["global_step"])))
        rows.sort()
        per[r] = pd.DataFrame(rows, columns=["t_begin", "t_end", "gs"])
    return per


def main():
    os.makedirs(OUT, exist_ok=True)
    per = load_ag(RAW)
    ranks = sorted(per)
    n = min(len(per[r]) for r in ranks)
    print(f"ranks={ranks}  ag#{IDX} events per rank: " +
          ", ".join(f"r{r}:{len(per[r])}" for r in ranks) + f"  -> use n={n}")

    # ---- alignment (a): by global_step ----
    by_gs = {}
    for r in ranks:
        d = per[r].drop_duplicates("gs").set_index("gs")
        by_gs[r] = d
    common = sorted(set.intersection(*[set(by_gs[r].index) for r in ranks]))
    gs_spread_begin = (pd.DataFrame({r: by_gs[r]["t_begin"].reindex(common) for r in ranks})
                       .max(axis=1) - pd.DataFrame({r: by_gs[r]["t_begin"].reindex(common) for r in ranks}).min(axis=1)) * 1000

    # ---- alignment (b): by collective ordinal ----
    ordr = {r: per[r].reset_index(drop=True).iloc[:n] for r in ranks}
    begin = pd.DataFrame({r: ordr[r]["t_begin"].values[:n] for r in ranks})
    end = pd.DataFrame({r: ordr[r]["t_end"].values[:n] for r in ranks})
    gs = pd.DataFrame({r: ordr[r]["gs"].values[:n] for r in ranks})
    spread_b = (begin.max(axis=1) - begin.min(axis=1)) * 1000
    spread_e = (end.max(axis=1) - end.min(axis=1)) * 1000
    gs_mismatch = (gs.nunique(axis=1) > 1).mean()

    print("\n=== arrival spread at the collective (ms) ===")
    print(f"  (a) aligned by global_step label : p50={gs_spread_begin.median():.1f} "
          f"p90={gs_spread_begin.quantile(.9):.1f}")
    print(f"  (b) aligned by collective ordinal: p50={spread_b.median():.1f} "
          f"p90={spread_b.quantile(.9):.1f}")
    print(f"  completion spread (b)            : p50={spread_e.median():.1f} "
          f"p90={spread_e.quantile(.9):.1f}")
    print(f"  fraction of ordinal steps where the gs labels DISAGREE: {gs_mismatch*100:.1f}%")
    print(f"  -> if (b) spread << (a) spread and labels disagree, per-gs alignment was an artefact")

    # per-rank entry time relative to the earliest rank, under (b)
    rel = (begin.sub(begin.min(axis=1), axis=0) * 1000)
    print("\n=== per-rank tardiness under (b) (ms): median / p90 ===")
    for r in ranks:
        print(f"  rank{r}: p50={rel[r].median():.1f}  p90={rel[r].quantile(.9):.1f}")
    rel.to_csv(f"{OUT}/tardiness_by_ordinal.csv", index=False)

    # how the gs offset between ranks evolves (step-shift test)
    print("\n=== gs label offset between ranks under (b) (median) ===")
    for r in ranks[1:]:
        off = (gs[r] - gs[ranks[0]])
        print(f"  rank{r} - rank0: p50={off.median():.0f}  unique={sorted(off.unique())[:6]}")

    # does the tardiness equal one step time?
    step_med = float(np.median(np.diff(np.sort(per[ranks[0]]["t_begin"].values)))) * 1000
    print(f"\n  rank0 step period (median Δt_begin) = {step_med:.0f} ms")
    print(f"  median tardiness of the slowest rank = {rel.max(axis=1).median():.0f} ms")

    # write the two alignments' spreads for the report
    pd.DataFrame({"spread_by_gs_ms": gs_spread_begin}).to_csv(f"{OUT}/spread_by_gs.csv", index=False)
    pd.DataFrame({"spread_by_ordinal_ms": spread_b, "spread_end_by_ordinal_ms": spread_e,
                  "gs_mismatch": (gs.nunique(axis=1) > 1)}).to_csv(f"{OUT}/spread_by_ordinal.csv", index=False)
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
