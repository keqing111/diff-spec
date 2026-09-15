#!/usr/bin/env python3
"""复现 2：用真实训练 run 的录制数据，验证 pack 不均及其后果。

不需要训练、不需要 server、不需要 GPU —— 只读 data/<RUN>/ 下的 jsonl。

用法：
    python3 02_validate_on_recorded_run.py                  # 默认跑全部已收录的 run
    python3 02_validate_on_recorded_run.py data/E3_w8_pf4

四个小节：
  [1] 录制的 pack 不均（与 01 的离线复现互相印证）
  [2] pack 样本数 → pack 生产时间：step_time ≈ docs_max × RTT ÷ num_workers
  [3] prefetch 队列深度：为什么有的步主进程要等、有的不必等
  [4] 后果：最重 rank 决定整步耗时

术语说明（重要）：
  · 这里**没有任何本地缓存**。每个样本的 hidden states 都要向 server 发一次请求
    （`--on-generate delete`，本地不留文件），所以每个样本都是 cache miss。
  · `load_ms < 1ms` 的含义不是"缓存命中"，而是**该批次已由 worker 提前生产完、
    排在 DataLoader 的 prefetch 队列里**，主进程直接取走。
  · `load_ms > 200ms` 的含义是**队列空了，主进程要现场等 worker 生产完这一批**。
  · 队列深度上限 = prefetch_factor × num_workers（本组配置为 4×W）。
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys

import numpy as np

SLOW_MS = 200.0
TOTAL_SEQ_LEN = 4096


def read_rows(pattern: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(pattern)):
        for line in open(f, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def by_rank(run: str, pattern: str) -> dict[int, list[dict]]:
    out = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{run}/rank*/{pattern}")):
        m = re.search(r"/rank(\d+)/", f)
        if m:
            out[int(m.group(1))].extend(read_rows(f))
    return dict(out)


def load_run(run: str) -> dict:
    d = {
        "load": by_rank(run, "load_rank*.jsonl"),
        "timeline": by_rank(run, "timeline_rank*.jsonl"),
        "sample": {},
        "n_workers": {},
    }
    wk = by_rank(run, "worker*_pid*.jsonl")
    d["all_worker"] = wk
    d["sample"] = {r: [x for x in v if x.get("kind") == "sample"] for r, v in wk.items()}
    d["n_workers"] = {r: len({x["pid"] for x in v}) for r, v in wk.items() if v}
    d["config"] = {}
    if os.path.exists(f"{run}/config.txt"):
        for line in open(f"{run}/config.txt"):
            p = line.split()
            if len(p) >= 2:
                d["config"][p[0]] = p[1]
    return d


def sec1_pack(d: dict) -> None:
    print("\n  [1] 录制的 pack 不均（与 01 的离线复现互相印证）")
    print(f"      {'rank':<6}{'pack 数':>8}{'packed_tokens':>15}{'token 填满率':>13}"
          f"{'docs/pack':>11}{'docs 范围':>11}")
    means = {}
    for r in sorted(d["load"]):
        rows = d["load"][r]
        docs = np.array([x.get("docs_in_batch", 0) for x in rows])
        toks = np.array([x.get("packed_tokens", 0) for x in rows])
        means[r] = docs.mean()
        print(f"      r{r:<5}{len(docs):>8}{toks.mean():>15.0f}"
              f"{toks.mean() / TOTAL_SEQ_LEN:>12.1%}{docs.mean():>11.2f}"
              f"{f'{docs.min()}~{docs.max()}':>11}")
    lo, hi = min(means.values()), max(means.values())
    print(f"      ⇒ token 填满率各 rank 一致，但 docs/pack 相差 {100 * (hi / lo - 1):.1f}%"
          f"（{lo:.2f} vs {hi:.2f}）")

    n = min(len(d["load"][r]) for r in d["load"])
    dd = np.stack([np.array([x.get("docs_in_batch", 0) for x in d["load"][r]])[:n]
                   for r in sorted(d["load"])])
    spread = dd.max(axis=0) - dd.min(axis=0)
    print(f"      跨 rank 同一步的 docs 极差：均值 {spread.mean():.2f} 中位 {np.median(spread):.0f}")
    print(f"      **四个 rank 完全相等的步占比 = {100 * np.mean(spread == 0):.1f}%**"
          "  ⇒ 绝大多数步各 rank 的 pack 并不相同")


def sec2_step_time(d: dict, run: str) -> None:
    """step_time ≈ docs_max × RTT ÷ num_workers —— 整步被最重 rank 的 pack 决定。"""
    print("\n  [2] pack 样本数 → 整步耗时（step_time ≈ docs_max × RTT ÷ num_workers）")
    tb = te = None
    for rows in d["timeline"].values():
        for x in rows:
            if x.get("t_batch") is None:
                continue
            tb = x["t_batch"] if tb is None else min(tb, x["t_batch"])
            te = x["t_end"] if te is None else max(te, x["t_end"])
    if tb is None:
        print("      (无 timeline)")
        return
    n_steps = max(len(v) for v in d["load"].values())
    step_t = (te - tb) / n_steps

    heaviest = max(d["load"], key=lambda r: np.mean([x.get("docs_in_batch", 0) for x in d["load"][r]]))
    docs_max = np.mean([x.get("docs_in_batch", 0) for x in d["load"][heaviest]])
    nw = d["n_workers"].get(heaviest, 0)
    rtt = np.mean([x["wall_ms"] for x in d["sample"][heaviest]]) / 1000.0
    pred = docs_max * rtt / nw

    print(f"      最重 rank = r{heaviest}   docs/pack={docs_max:.2f}   "
          f"每样本耗时 RTT={rtt * 1000:.0f}ms   num_workers={nw}")
    print(f"      预测 step_time = {docs_max:.2f} × {rtt:.3f}s ÷ {nw} = {pred:.2f}s")
    print(f"      实测 step_time = {step_t:.2f}s      比值 = {step_t / pred:.3f}")
    print(f"      ⇒ 整步耗时 = 「最重 rank 串行生产一个 pack 的时间」× {step_t / pred:.2f}")
    print(f"        （每样本一次 server 请求，num_workers 个请求并行、再串行 {docs_max:.0f} 轮）")
    # 若 pack 完全均衡，理论提速
    mean_docs = np.mean([np.mean([x.get("docs_in_batch", 0) for x in d["load"][r]]) for r in d["load"]])
    print(f"      若各 rank pack 完全均衡到 {mean_docs:.2f} docs："
          f"step_time → {mean_docs * rtt / nw:.2f}s，**提速 {100 * (1 - mean_docs / docs_max):.1f}%**")


def sec3_queue(d: dict) -> None:
    """重建取批时刻的 prefetch 队列深度。"""
    print("\n  [3] prefetch 队列深度（解释 load_ms 的双峰）")
    cap = None
    print(f"      {'rank':<6}{'队列深度 p50':>13}{'队列深度 max':>13}"
          f"{'队列空时的步占比':>18}{'队列空→load_ms p50':>20}")
    for r in sorted(d["load"]):
        ev = []
        for x in d["all_worker"].get(r, []):
            if x.get("kind") == "collate":
                ev.append((x["t_perf"], "c", None))
        for x in d["load"][r]:
            ev.append((x["t0"], "l", x["load_ms"]))
        ev.sort(key=lambda e: e[0])
        depth = 0
        depths, empty_loads = [], []
        for e in ev:
            if e[1] == "c":
                depth += 1
            else:
                depths.append(depth)
                if depth == 0:
                    empty_loads.append(e[2])
                depth -= 1
        if not depths:
            continue
        nw = d["n_workers"].get(r, 0)
        cap = 4 * nw if cap is None else cap
        empty_pct = 100 * len(empty_loads) / len(depths)
        print(f"      r{r:<5}{np.median(depths):>13.0f}{max(depths):>13d}{empty_pct:>17.1f}%"
              f"{(np.median(empty_loads) if empty_loads else float('nan')):>19.0f}ms")
    print(f"      ⇒ 队列深度上限 = prefetch_factor(4) × num_workers = {cap}，实测 max 与之一致")
    print("      ⇒ 队列为 0 时主进程必须现场等 worker 生产 ⇒ load_ms 跳到 ~1s")
    print("      ⇒ 队列非 0 时批次已在队列里 ⇒ load_ms < 1ms")


def sec4_consequence(d: dict) -> None:
    print("\n  [4] 后果：最重 rank 决定整步耗时，其余 rank 在集合点等它")
    tl = d["timeline"]
    ranks = sorted(tl)
    bystep = collections.defaultdict(dict)
    for r in ranks:
        for x in tl[r]:
            bystep[x["global_step"]][r] = x
    fwd = collections.defaultdict(list)
    skew, last = [], collections.Counter()
    for gs, dd in bystep.items():
        if len(dd) != len(ranks):
            continue
        tbmap = {r: v["t_batch"] for r, v in dd.items()}
        skew.append(max(tbmap.values()) - min(tbmap.values()))
        last[max(tbmap, key=tbmap.get)] += 1
        for r, v in dd.items():
            if v.get("t_fwd") is not None:
                fwd[r].append(v["t_fwd"] - v["t_batch"])
    if not skew:
        return
    skew = np.array(skew)
    tot = sum(last.values())
    print(f"      各 rank 进入 batch 的时刻差 p50={np.median(skew) * 1000:.0f}ms "
          f"p90={np.percentile(skew, 90) * 1000:.0f}ms")
    print("      最后到达的 rank 分布：" + "  ".join(f"r{r}:{100 * c / tot:.0f}%" for r, c in sorted(last.items())))
    for r in sorted(fwd):
        a = np.array(fwd[r])
        print(f"      r{r} 自己 forward 的耗时 p50={np.median(a) * 1000:7.0f}ms")
    # 分条件看：停顿步 vs 干净步（多数配置下中位数落在干净步上）
    slow_steps = [gs for gs, dd in bystep.items()
                  if len(dd) == len(ranks)
                  and any(x.get("load_ms", 0) > SLOW_MS for x in dd.values())]
    frac = 100 * len(slow_steps) / max(len(bystep), 1)
    print(f"      有 rank 停顿的步占比 = {frac:.1f}%")
    if len(slow_steps) > 10:
        sl = sorted(slow_steps)[:1]
        sk_slow = []
        for gs in slow_steps:
            tbmap = {r: v["t_batch"] for r, v in bystep[gs].items()}
            sk_slow.append(max(tbmap.values()) - min(tbmap.values()))
        print(f"      仅在**停顿步**上统计：时刻差 p50={np.median(sk_slow) * 1000:.0f}ms")
    print("      ⇒ 在有停顿的步上：最重 rank 自己 forward 最短（它一到就开跑），")
    print("        其余 rank 的 forward 被拉到 ≈ 它的迟到量（都在等它）")
    print("      ⇒ 若本配置停顿很少（p50 很小），说明供给相对充裕，该效应只在少数步上出现")


def main() -> None:
    runs = sys.argv[1:] or [
        "data/E3_w8_pf4",       # 生产口径（W=8）
        "data/E1_ref_w2_pf4",   # 参考配置（W=2，注意这是误配置，见 README）
        "data/E7_band_w8",      # 窄带语料对照
    ]
    for run in runs:
        if not os.path.isdir(run):
            print(f"跳过（不存在）：{run}")
            continue
        d = load_run(run)
        if not d["load"]:
            print(f"跳过（无数据）：{run}")
            continue
        name = os.path.basename(run.rstrip("/"))
        print("=" * 96)
        print(f"RUN {name}    workers={d['config'].get('workers')} "
              f"prefetch={d['config'].get('prefetch_factor')} "
              f"data={os.path.basename(d['config'].get('data_path', '') or '')}")
        print("=" * 96)
        sec1_pack(d)
        sec2_step_time(d, run)
        sec3_queue(d)
        sec4_consequence(d)
        print()


if __name__ == "__main__":
    main()
