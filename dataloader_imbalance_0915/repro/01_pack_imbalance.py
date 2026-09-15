#!/usr/bin/env python3
"""复现 1：DataLoader 造出的 pack 在 DP 各 rank 之间负载不均。

**本脚本不需要训练、不需要 server、不需要 GPU**：直接加载真实数据集，
用训练代码里同一个 sampler（MultipackDistributedBatchSamplerV2）离线生成
每个 rank 的 batch，统计 pack 里的**样本数**。

用法：
    python3 01_pack_imbalance.py                      # 默认 700k 语料，前 1500 步
    python3 01_pack_imbalance.py --steps 3000
    python3 01_pack_imbalance.py --data /path/to/dataset
    python3 01_pack_imbalance.py --compare            # 顺带对比窄带语料

核心结论（见 README §1）：
    token 打包约束的是 **token 数**，不是 **样本数**。
    四个 rank 的 pack 都精确填满 ~3878 token（填满率完全一致），
    但因为文档长度不等，pack 里的文档数相差 **26%**（4.67 vs 5.90）。
    而每个文档 = 一次 server 请求 ⇒ 每个 pack 的生产时间天然相差 26%。
"""
from __future__ import annotations

import argparse
import collections
import sys

import numpy as np

SPECULATORS_SRC = "/home/y50063564/dspark_project/speculators/src"
DEFAULT_DATA = "/home/y50063564/data/open_perfectblend_qwen3_4b_700k"
BAND_DATA = "/home/y50063564/data/open_perfectblend_qwen3_4b_band600_800"
TOTAL_SEQ_LEN = 4096


def load_lengths(data_path: str) -> np.ndarray:
    """取数据集的 seq_len 列（与 ArrowDataset._compute_approx_lengths 同源）。"""
    sys.path.insert(0, SPECULATORS_SRC)
    from datasets import load_from_disk

    ds = load_from_disk(data_path, keep_in_memory=False)
    return np.asarray(ds.with_format(None)["seq_len"], dtype=np.int64)


def pack_per_rank(lengths, n_ranks: int, n_steps: int, seed: int = 42):
    """用训练代码里的同一个 sampler，取每个 rank 的前 n_steps 个 pack。

    返回 {rank: (docs_per_pack[np.int], tokens_per_pack[np.int])}
    """
    sys.path.insert(0, SPECULATORS_SRC)
    from speculators.train.distributed_batch_sampler import (
        MultipackDistributedBatchSamplerV2,
    )

    out = {}
    for rank in range(n_ranks):
        sampler = MultipackDistributedBatchSamplerV2(
            batch_max_length=TOTAL_SEQ_LEN,
            lengths=list(lengths),
            num_replicas=n_ranks,
            rank=rank,
            seed=seed,
        )
        docs, toks = [], []
        for i, batch in enumerate(sampler):
            if i >= n_steps:
                break
            docs.append(len(batch))
            toks.append(int(sum(lengths[j] for j in batch)))
        out[rank] = (np.array(docs), np.array(toks))
    return out


def report(name: str, packs: dict, lengths: np.ndarray) -> None:
    ranks = sorted(packs)
    print("=" * 96)
    print(f"语料：{name}   文档数={len(lengths):,}   "
          f"长度 mean={lengths.mean():.0f} 中位={np.median(lengths):.0f} "
          f"p95={np.percentile(lengths, 95):.0f} max={lengths.max()}  "
          f"CV={lengths.std() / lengths.mean():.2f}")
    print("=" * 96)

    # ---- 表 1：packed tokens 恒定，但样本数不同 ----
    print("\n[表 1] 每个 rank 的 pack 统计（这是问题的核心）")
    print(f"  {'rank':<6}{'pack 数':>8}{'packed_tokens':>15}{'token 填满率':>13}"
          f"{'docs/pack':>11}{'docs 范围':>11}{'docs p10~p90':>15}")
    docs_mean = {}
    for r in ranks:
        docs, toks = packs[r]
        docs_mean[r] = docs.mean()
        print(f"  r{r:<5}{len(docs):>8}{np.mean(toks):>15.0f}"
              f"{np.mean(toks) / TOTAL_SEQ_LEN:>12.1%}{docs.mean():>11.2f}"
              f"{f'{docs.min()}~{docs.max()}':>11}"
              f"{f'{np.percentile(docs, 10):.0f}~{np.percentile(docs, 90):.0f}':>15}")
    lo, hi = min(docs_mean.values()), max(docs_mean.values())
    print(f"\n  ⇒ packed_tokens 各 rank 几乎相同（填满率一致），但 docs/pack "
          f"从 {lo:.2f} 到 {hi:.2f}，**相差 {100 * (hi / lo - 1):.1f}%**")
    print(f"  ⇒ 每个文档 = 一次 server 请求 ⇒ 每个 pack 的生产时间相差 {100 * (hi / lo - 1):.1f}%")

    # ---- 表 2：跨 rank 同一步也不相等 ----
    print("\n[表 2] 同一个 global step 上，四个 rank 的 pack 是否相同")
    n = min(len(packs[r][0]) for r in ranks)
    d = np.stack([packs[r][0][:n] for r in ranks])          # [rank, step]
    spread = d.max(axis=0) - d.min(axis=0)
    print(f"  同步步数 = {n}")
    print(f"  跨 rank docs 极差：均值 {spread.mean():.2f}   中位 {np.median(spread):.0f}   max {spread.max()}")
    print(f"  **四个 rank 完全相等的步占比 = {100 * np.mean(spread == 0):.1f}%**")
    print("\n  出现最多的 docs/pack 取值（全体 rank 合并）：")
    allv = d.ravel()
    vals, cnt = np.unique(allv, return_counts=True)
    order = np.argsort(-cnt)[:5]
    for i in order:
        print(f"      docs/pack = {int(vals[i]):<3} 占 {100 * cnt[i] / len(allv):5.1f}%")

    # ---- 表 3：为什么 ----
    print("\n[表 3] 原因：token 预算被长文档吃掉了额度")
    print(f"  文档长度分布：中位 {np.median(lengths):.0f} token，"
          f"但 p99 达 {np.percentile(lengths, 99):.0f} token")
    print(f"  一个 4096 token 的 pack：")
    print(f"      若全是中位长度文档 → 约 {TOTAL_SEQ_LEN / np.median(lengths):.1f} 个")
    print(f"      若全是 p99 长度文档 → 约 {TOTAL_SEQ_LEN / np.percentile(lengths, 99):.1f} 个")
    print(f"  样本数不参与打包约束，只是「填满 token 预算后的副产物」⇒ 各 rank 随机游走")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare", action="store_true", help="顺带对比窄带语料")
    args = ap.parse_args()

    lengths = load_lengths(args.data)
    report(f"{args.data.split('/')[-1]}（真实训练语料）",
           pack_per_rank(lengths, args.ranks, args.steps, args.seed), lengths)

    if args.compare:
        import os
        if os.path.isdir(BAND_DATA):
            bl = load_lengths(BAND_DATA)
            report(f"{BAND_DATA.split('/')[-1]}（窄带对照：长度 600~800）",
                   pack_per_rank(bl, args.ranks, args.steps, args.seed), bl)
            print("\n对照含义：文档长度方差被压掉后，各 rank 的 docs/pack 差异从 ~26% 收敛到 ~0.3%，")
            print("          跨 rank 相等的步占比从 ~10% 升到 ~60%。")
            print("          但**仍不能做到 100% 相等**：整数装箱下 4096/中位长度 不是整数，")
            print("          各 rank 仍会在相邻两个取值之间跳（这里 5 与 6，占比 68% : 32%）。")

    print("\n" + "=" * 96)
    print("复现完成。若要验证这些 pack 在真实训练里的后果，跑 02_validate_on_recorded_run.py")
    print("=" * 96)


if __name__ == "__main__":
    main()
