#!/usr/bin/env python3
"""Build a narrow doc-length band subset (uniform request size).

Purpose: test whether the "rho<0.9" stability threshold is caused by
service-time VARIANCE. A narrow band makes every per-doc vLLM request nearly
the same cost (and, because packing is by tokens, also makes docs/batch equal
across ranks -> symmetric demand) within the CURRENT per-doc protocol.
"""
import argparse
import shutil
from pathlib import Path

import pyarrow.ipc as ipc
from datasets import Dataset

SRC = Path("/home/y50063564/data/open_perfectblend_qwen3_4b_700k")
DST = Path("/home/y50063564/data/open_perfectblend_qwen3_4b_band600_800")
LO, HI, N = 600, 800, 40000
COPIES = ["d2t.npy", "t2d.npy", "token_freq.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    ap.add_argument("--lo", type=int, default=LO)
    ap.add_argument("--hi", type=int, default=HI)
    ap.add_argument("--n", type=int, default=N)
    a = ap.parse_args()
    if a.dst.exists():
        raise SystemExit(f"exists: {a.dst}")

    kept = []
    import pyarrow as pa
    for shard in sorted(a.src.glob("data-*.arrow")):
        with open(shard, "rb") as fh:
            with ipc.open_stream(fh) as r:
                t = r.read_all()
        lens = t.column("seq_len").to_pylist()
        idx = [i for i, L in enumerate(lens) if a.lo <= L <= a.hi]
        if idx:
            kept.append(t.take(pa.array(idx)))
        got = sum(k.num_rows for k in kept)
        print(f"  {shard.name}: +{len(idx)} -> total {got}")
        if got >= a.n:
            break
    t = pa.concat_tables(kept).slice(0, a.n)
    ds = Dataset(t)
    print(f"band [{a.lo},{a.hi}] rows={ds.num_rows} seq_len range="
          f"{min(ds['seq_len'])}..{max(ds['seq_len'])}")
    ds.set_format("torch")
    ds.save_to_disk(str(a.dst))
    for c in COPIES:
        s = a.src / c
        if s.exists():
            shutil.copy2(s, a.dst / c)
    ds2 = Dataset.load_from_disk(str(a.dst))
    assert ds2.num_rows == a.n, ds2.num_rows
    print(f"saved {a.dst} rows={ds2.num_rows} (torch format, mappings copied)")


if __name__ == "__main__":
    main()
