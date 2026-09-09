#!/usr/bin/env python3
"""Build the fixed evaluation subsets used by the overfitting/degradation check.

Subsets (all saved as HF on-disk datasets in torch format, like the 50k crop):
  train1000 : 1000 docs drawn from the training partition [0, 45000) of the crop
  val       : all 5000 docs of the original validation partition [45000, 50000)
  val100    : 100 of those val docs (for draft-entropy)
  out1000   : 1000 docs sampled OUTSIDE the 5w crop (from shard 1 of the 700k)

One fixed RNG (seed) so the SAME docs are reused across every checkpoint & run.
Hidden states are generated online (into each subset dir /hidden_states) the
first time a subset is evaluated and cached for later checkpoints/runs.
"""
import argparse
import random
from pathlib import Path

import pyarrow.ipc as ipc
from datasets import Dataset, load_from_disk

CROP = Path("/home/y50063564/data/open_perfectblend_qwen3_4b_50k")
FULL = Path("/home/y50063564/data/open_perfectblend_qwen3_4b_700k")
OUT = Path("/home/y50063564/processed_data/dspark_data/dspark_eval")
SEED = 2026
COPIES = ["d2t.npy", "t2d.npy", "token_freq.pt"]


def save_subset(name: str, ds: Dataset) -> None:
    dst = OUT / name
    if dst.exists():
        import shutil
        shutil.rmtree(dst)
    ds = ds.select(range(len(ds)))  # copy materialisation
    ds.set_format("torch")
    ds.save_to_disk(str(dst))
    print(f"saved {name}: {len(ds)} rows -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    crop = load_from_disk(str(CROP))           # 50k, torch format
    train_idx = sorted(rng.sample(range(0, 45000), 1000))
    val_idx = list(range(45000, 50000))
    val100_idx = sorted(rng.sample(range(45000, 50000), 100))
    save_subset("train1000", crop.select(train_idx))
    save_subset("val", crop.select(val_idx))
    save_subset("val100", crop.select(val100_idx))

    # out1000: from shard 1 of the full corpus (rows [140000,280000)), all of
    # which are outside the 5w crop (rows [0,50000) of shard 0).
    shard1 = FULL / "data-00001-of-00005.arrow"
    with open(shard1, "rb") as fh:
        with ipc.open_stream(fh) as reader:
            table = reader.read_all()
    print(f"shard1 rows: {table.num_rows}")
    ds_out = Dataset(table)
    out_idx = sorted(rng.sample(range(0, ds_out.num_rows), 1000))
    save_subset("out1000", ds_out.select(out_idx))

    for name in ("train1000", "val", "val100", "out1000"):
        for c in COPIES:
            s = CROP / c if name != "out1000" else FULL / c
            if s.exists():
                import shutil
                shutil.copy2(s, OUT / name / c)
    print("subset dirs:")
    for p in sorted(OUT.iterdir()):
        if p.is_dir():
            print("  ", p)


if __name__ == "__main__":
    main()
