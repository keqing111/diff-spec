#!/usr/bin/env python3
"""Crop open_perfectblend_qwen3_4b_700k -> a 50k (5w) experiment subset.

Takes the first 50,000 rows of shard 0 (data-00000-of-00005.arrow) and writes
them as a self-contained HuggingFace ``datasets`` on-disk dataset with the same
column schema (input_ids / loss_mask / seq_len).  Also copies the vocab-mapping
files (d2t.npy, t2d.npy, token_freq.pt) so train.py's data loader and
parse_vocab_mappings keep working with ``--data-path <crop dir>`` unchanged.

The train/val split (0.9) is applied at load time inside ArrowDataset:
  train = rows [0, 45000),  val = rows [45000, 50000).
"""
import argparse
import shutil
from pathlib import Path

import pyarrow.ipc as ipc
from datasets import Dataset

SRC = Path("/home/y50063564/data/open_perfectblend_qwen3_4b_700k")
DEFAULT_DST = Path("/home/y50063564/data/open_perfectblend_qwen3_4b_50k")
N_CROP = 50_000

# Files that must live next to the arrow data for train.py / dataloader.
COPIES = ["d2t.npy", "t2d.npy", "token_freq.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--n", type=int, default=N_CROP)
    args = ap.parse_args()

    shards = sorted(args.src.glob("data-*.arrow"))
    print(f"source shards: {[s.name for s in shards]}")
    shard0 = shards[0]

    with open(shard0, "rb") as fh:
        with ipc.open_stream(fh) as reader:
            table = reader.read_all()
    print(f"shard {shard0.name}: {table.num_rows} rows, schema={table.schema}")

    n = min(args.n, table.num_rows)
    sub = table.slice(0, n)
    print(f"cropping to first {n} rows ...")

    ds = Dataset(sub)
    assert ds.num_rows == n, (ds.num_rows, n)
    assert list(ds.column_names) == ["input_ids", "loss_mask", "seq_len"], ds.column_names

    # CRITICAL: the source dataset is stored in torch format (its state.json has
    # "_format_type": "torch"), so load_from_disk returns torch Tensors. The
    # trainer (data.py build_client_item / _get_raw_data) calls .tolist(),
    # torch.equal(...) and feeds tensors to the device, so the crop must also be
    # torch-formatted or those calls break on python lists.
    ds.set_format("torch")

    if args.dst.exists():
        raise SystemExit(f"destination already exists: {args.dst} (refusing to overwrite)")
    ds.save_to_disk(str(args.dst))
    print(f"saved dataset -> {args.dst} ({ds.num_rows} rows)")

    for name in COPIES:
        s = args.src / name
        if s.exists():
            shutil.copy2(s, args.dst / name)
            print(f"  copied {name}")
        else:
            print(f"  WARNING: {s} not found, skipping")

    # quick sanity reload through the same API the trainer uses
    ds2 = Dataset.load_from_disk(str(args.dst))
    print("reload OK:", ds2.num_rows, "rows;",
          "seq_len range", ds2["seq_len"][0], "..", ds2["seq_len"][-1])


if __name__ == "__main__":
    main()
