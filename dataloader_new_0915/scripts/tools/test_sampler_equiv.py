#!/usr/bin/env python3
"""P0 check for the sampler change: policy=off must reproduce the ORIGINAL
sampler's batches bit-for-bit, and the new policies must actually change what
we intend (docs-balance equalises per-rank doc counts; perm swaps streams).
"""
import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/home/y50063564/dspark_project/speculators/src")

ORIG_PATH = "/home/y50063564/dataloader_diag/snapshot/distributed_batch_sampler.py.orig"


def load_orig():
    from importlib.machinery import SourceFileLoader

    loader = SourceFileLoader("orig_sampler", ORIG_PATH)
    spec = importlib.util.spec_from_loader("orig_sampler", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["orig_sampler"] = mod
    loader.exec_module(mod)
    return mod


def main():
    import os
    orig = load_orig()
    from speculators.train.distributed_batch_sampler import MultipackDistributedBatchSamplerV2 as New

    rng = np.random.default_rng(0)
    # 700k-ish doc length distribution: mostly short, some long (like the real corpus)
    lengths = np.clip(rng.lognormal(mean=6.2, sigma=0.55, size=20000).astype(int), 8, 4096)
    max_len, replicas = 4096, 4

    # ---- policy=off equivalence ----
    ok = True
    for epoch in (0, 1):
        for rank in range(replicas):
            a = orig.MultipackDistributedBatchSamplerV2(max_len, lengths, replicas, rank, seed=42)
            b = New(max_len, lengths, replicas, rank, seed=42)
            ba = a._generate_batches(epoch)
            bb = b._generate_batches(epoch)
            same = len(ba) == len(bb) and all(
                np.array_equal(x, y) for x, y in zip(ba, bb, strict=True))
            if not same:
                ok = False
                print(f"  MISMATCH epoch={epoch} rank={rank}: lens {len(ba)} vs {len(bb)}")
    print(f"[P0-sampler] policy=off identical to original: {ok}  "
          f"(epochs 0,1 x ranks 0..3, 20000 docs)")

    # ---- what do the policies do? ----
    def docs_per_rank(sampler):
        batches = sampler._generate_batches(0)
        return [len(b) for b in batches]

    os.environ.pop("SAMP_RANK_PERM", None)
    for pol in ("off", "docs", "tokens"):
        os.environ["SAMP_BALANCE"] = pol
        per_rank = [sum(docs_per_rank(New(max_len, lengths, replicas, r, seed=42)))
                    for r in range(replicas)]
        total = sum(per_rank)
        print(f"  policy={pol:6s} cumulative docs per rank={per_rank}  "
              f"spread(max-min)={max(per_rank)-min(per_rank)}  total={total}")

    # ---- permutation swaps streams ----
    os.environ["SAMP_BALANCE"] = "off"
    os.environ["SAMP_RANK_PERM"] = "3,2,1,0"
    perm_docs = [sum(docs_per_rank(New(max_len, lengths, replicas, r, seed=42)))
                 for r in range(replicas)]
    os.environ.pop("SAMP_RANK_PERM")
    base_docs = [sum(docs_per_rank(New(max_len, lengths, replicas, r, seed=42)))
                 for r in range(replicas)]
    print(f"  base docs/rank   = {base_docs}")
    print(f"  perm 3,2,1,0     = {perm_docs}  (expect reversed: {base_docs[::-1]})")
    print(f"  permutation reverses streams: {perm_docs == base_docs[::-1]}")


if __name__ == "__main__":
    main()
