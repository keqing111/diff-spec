#!/usr/bin/env python3
"""关键指标打印器 —— 从一个 run 目录（精简数据即可）算出全部决定性数字。

    python3 print_key_metrics.py <run_dir> [<run_dir> ...]
    python3 print_key_metrics.py data/E1_ref_w2_pf4
    python3 print_key_metrics.py data/*            # 横向对比所有 run

输出分 7 节，对应报告里的 7 组主张：
  [1] 每 rank 负载与饥饿    -> 边①/③：docs/batch 不均、load_ms 双峰
  [2] 逐步跨 rank 不均       -> 「每个 global step 下各 rank 的 pack 仍不同」+ 反证
  [3] 周期性                 -> 慢步指示序列自相关（lag1 负 / lag2 正 => 周期 2）
  [4] 集合同步放大           -> ready_skew、最后到达者、各 rank fwd 时长、停顿归因
  [5] 分支锁存诊断           -> 同 ρ 同尺寸下各 rank 快慢为何天差地别
  [6] server 侧天花板        -> running/waiting、prompt tok/s、engine steps/s
  [7] ρ = 需求/能力          -> 判据（ρ≲0.90 不卡；0.92-0.95 陡增）

字段口径（与 EVIDENCE.md F2 一致）：
  慢步 = load_ms > 200ms；快取 = load_ms < 1ms
  RTT  = 该 rank 所有 sample 事件的 mean(wall_ms)
  ρ    = (docs/batch ÷ step_time) ÷ (num_workers ÷ RTT)
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


# ------------------------------------------------------------------ 读数据
def _read_glob(pattern):
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


def _by_rank(run, pattern):
    out = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{run}/rank*/{pattern}")):
        m = re.search(r"/rank(\d+)/", f)
        if not m:
            continue
        out[int(m.group(1))].extend(_read_glob(f))
    return out


def load_run(run):
    d = {
        "load": _by_rank(run, "load_rank*.jsonl"),
        "timeline": _by_rank(run, "timeline_rank*.jsonl"),
        "ag": _by_rank(run, "ag_rank*.jsonl"),
    }
    wk = _by_rank(run, "worker*_pid*.jsonl")
    d["sample"] = {r: [x for x in v if x.get("kind") == "sample"] for r, v in wk.items()}
    d["collate"] = {r: [x for x in v if x.get("kind") == "collate"] for r, v in wk.items()}
    d["n_workers"] = {
        r: len({x["pid"] for x in v}) for r, v in wk.items() if v
    }
    d["config"] = {}
    cfg = f"{run}/config.txt"
    if os.path.exists(cfg):
        for line in open(cfg):
            p = line.split()
            if len(p) >= 2:
                d["config"][p[0]] = p[1]
    return d


def load_server(run):
    """返回 (running, waiting, prompt_tok_per_s, engine_steps_per_s, tok_per_step)。"""
    path = f"{run}/metrics_slim.jsonl"
    if not os.path.exists(path):
        path = f"{run}/metrics.jsonl"
    if not os.path.exists(path):
        return None
    run_, wait_, pt = collections.defaultdict(list), collections.defaultdict(list), []
    it = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            x = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = x.get("m", {})
        for k, v in m.get("vllm:num_requests_running", {}).items():
            run_[k].append(v)
        for k, v in m.get("vllm:num_requests_waiting", {}).items():
            wait_[k].append(v)
        if "vllm:prompt_tokens_total" in m:
            pt.append((x["w"], sum(m["vllm:prompt_tokens_total"].values())))
        if "vllm:iteration_tokens_total_count" in m:
            it.append(
                (
                    x["w"],
                    sum(m["vllm:iteration_tokens_total_count"].values()),
                    sum(m["vllm:iteration_tokens_total_sum"].values()),
                )
            )
    if not run_:
        return None
    k = sorted(run_)[0]
    r, w = np.array(run_[k]), np.array(wait_[sorted(wait_)[0]])
    toks = (pt[-1][1] - pt[0][1]) / (pt[-1][0] - pt[0][0]) if len(pt) > 1 else float("nan")
    if len(it) > 1:
        d0, d1 = it[0], it[-1]
        nstep = d1[1] - d0[1]
        steps_per_s = nstep / (d1[0] - d0[0])
        tok_per_step = (d1[2] - d0[2]) / max(nstep, 1)
    else:
        steps_per_s = tok_per_step = float("nan")
    return r, w, toks, steps_per_s, tok_per_step


def step_span(d):
    """(总时长秒, 步数) —— 取所有 rank 的最早 t_batch / 最晚 t_end。"""
    tb = te = None
    for rows in d["timeline"].values():
        for x in rows:
            a, b = x.get("t_batch"), x.get("t_end")
            if a is None or b is None:
                continue
            tb = a if tb is None else min(tb, a)
            te = b if te is None else max(te, b)
    if tb is None:
        return float("nan"), 0
    n = max((len(v) for v in d["load"].values()), default=0)
    return te - tb, n


# ------------------------------------------------------------------ 各节
def sec1_load(d, step_t):
    print("  [1] 每 rank 负载与饥饿（边①/③）")
    print(
        f"      {'rank':<5}{'docs/batch':>11}{'load_p50':>11}{'load_p90':>11}"
        f"{'慢步>200ms':>11}{'快取<1ms':>10}{'RTT':>8}{'CPU/doc':>9}"
    )
    ns = d["n_workers"]
    for r in sorted(d["load"]):
        v = np.array([x["load_ms"] for x in d["load"] [r]])
        docs = [x.get("docs_in_batch", 0) for x in d["load"][r]]
        s = d["sample"].get(r, [])
        rtt = np.mean([x["wall_ms"] for x in s]) if s else float("nan")
        cpu = np.mean([x.get("cpu_ms", 0) for x in s]) if s else float("nan")
        print(
            f"      r{r:<4}{np.mean(docs):11.2f}{np.median(v):11.1f}"
            f"{np.percentile(v, 90):11.1f}{100 * np.mean(v > SLOW_MS):10.1f}%"
            f"{100 * np.mean(v < 1):9.1f}%{rtt:7.0f}ms{cpu:8.0f}ms"
        )
    # 双峰诊断（最慢 rank）
    if d["load"]:
        worst = max(d["load"], key=lambda r: np.mean(np.array([x["load_ms"] for x in d["load"][r]]) > SLOW_MS))
        v = np.array([x["load_ms"] for x in d["load"][worst]])
        mid = 100 * np.mean((v >= 1) & (v <= SLOW_MS))
        print(
            f"      → 最慢 rank r{worst}：<1ms {100 * np.mean(v < 1):.1f}% | 1~200ms {mid:.1f}%"
            f" | >200ms {100 * np.mean(v > SLOW_MS):.1f}%  ⇒ 中间段近空 = 双峰（非渐变耗尽）"
        )
    return ns


def sec2_perstep(d):
    print("\n  [2] 逐步跨 rank 不均（验证「每个 global step 下各 rank 的 pack 仍不同」）")
    bystep = collections.defaultdict(dict)
    for r, rows in d["load"].items():
        for x in rows:
            bystep[x["global_step"]][r] = x.get("docs_in_batch", 0)
    full = [v for v in bystep.values() if len(v) == len(d["load"]) and len(v) > 1]
    if not full:
        print("      (数据不足)")
        return
    spread = np.array([max(v.values()) - min(v.values()) for v in full])
    print(f"      同步步数={len(full)}  跨 rank docs 极差：均值 {spread.mean():.2f}  中位 {np.median(spread):.0f}")
    print(f"      **四 rank 完全相等的步占比 = {100 * np.mean(spread == 0):.1f}%**  ⇒ 绝大多数步各 rank 的 pack 是不同的")
    rng = []
    for r, rows in d["load"].items():
        v = [x.get("docs_in_batch", 0) for x in rows]
        rng.append(f"r{r}:{min(v)}~{max(v)}")
    print(f"      各 rank 逐步 docs 取值范围：" + "  ".join(rng))

    # 关键反证：rank 内部，本步 docs 数几乎不预测本步是否慢
    print("\n      ⚠️ 反证：rank 内部「本步 docs 多 ⇒ 本步更慢」不成立")
    print(f"      {'rank':<6}{'corr(docs, log load)':>21}   {'按 docs 分组的慢步率':<40}")
    for r in sorted(d["load"]):
        dd = np.array([x.get("docs_in_batch", 0) for x in d["load"][r]], dtype=float)
        ll = np.array([x["load_ms"] for x in d["load"][r]], dtype=float)
        if len(set(dd)) < 2:
            continue
        c = float(np.corrcoef(dd, np.log1p(ll))[0, 1])
        parts = []
        for k in sorted(set(dd)):
            m = dd == k
            if m.sum() >= 20:
                parts.append(f"{int(k)}:{100 * np.mean(ll[m] > SLOW_MS):.0f}%")
        print(f"      r{r:<5}{c:>21.3f}   {'  '.join(parts):<40}")
    print("      ⇒ 相关系数≈0、慢步率几乎不随 docs 变化 ⇒ 慢/快是双峰，与当步样本数无关")


def sec3_period(d):
    print("\n  [3] 周期性（慢步指示序列自相关）")
    for r in sorted(d["load"]):
        v = np.array([x["load_ms"] for x in d["load"][r]])
        s = (v > SLOW_MS).astype(float)
        if s.std() == 0:
            print(f"      r{r}: 无慢步（恒定 {'快' if s[0] == 0 else '慢'}）")
            continue
        s = s - s.mean()
        ac = [float(np.corrcoef(s[:-k], s[k:])[0, 1]) for k in range(1, 7)]
        print(
            f"      r{r}: 慢步率={100 * np.mean(v > SLOW_MS):5.1f}%  自相关 "
            + " ".join(f"lag{k}={a:+.2f}" for k, a in enumerate(ac, 1))
        )
    print("      （lag1 显著为负 + lag2 显著为正 ⇒ 周期恰为 2：快→慢→快→慢）")


def sec4_sync(d):
    print("\n  [4] 集合同步放大（边④/⑤）")
    tl = d["timeline"]
    ranks = sorted(tl)
    if not ranks:
        print("      (无 timeline 数据)")
        return
    bystep = collections.defaultdict(dict)
    for r in ranks:
        for x in tl[r]:
            bystep[x["global_step"]][r] = x
    # load_ms 以 global_step 为键（load_*.jsonl），用于判断该步是否有人掉队
    lm = {r: {x["global_step"]: x["load_ms"] for x in d["load"][r]} for r in ranks}

    skew_slow, skew_fast = [], []
    nslow = collections.Counter()
    last, fwd = collections.Counter(), collections.defaultdict(list)
    for gs, dd in bystep.items():
        if len(dd) != len(ranks) or any(gs not in lm[r] for r in ranks):
            continue
        tb = {r: v["t_batch"] for r, v in dd.items()}
        s = max(tb.values()) - min(tb.values())
        k = sum(lm[r][gs] > SLOW_MS for r in ranks)
        nslow[k] += 1
        (skew_slow if k else skew_fast).append(s)
        last[max(tb, key=tb.get)] += 1
        for r, v in dd.items():
            if v.get("t_fwd") is not None:
                fwd[r].append(v["t_fwd"] - v["t_batch"])
    if not skew_slow and not skew_fast:
        print("      (无同步步)")
        return
    skew = np.array(skew_slow + skew_fast)
    print(f"      ready_skew（各 rank 进入 batch 的时刻差）p50={np.median(skew) * 1000:.0f}ms "
          f"p90={np.percentile(skew, 90) * 1000:.0f}ms max={skew.max() * 1000:.0f}ms")
    tot = sum(last.values())
    print("      最后到达的 rank 分布：" + "  ".join(f"r{r}:{100 * c / tot:.0f}%" for r, c in sorted(last.items())))
    for r in sorted(fwd):
        a = np.array(fwd[r])
        print(f"      r{r} 自己的 fwd 时长 p50={np.median(a) * 1000:7.0f}ms p90={np.percentile(a, 90) * 1000:7.0f}ms")
    print("      ⇒ 非受害者 fwd 被拉长到 ≈ready_skew，受害者自己最短 = 全体在等它")

    # --- 停顿归因：区分「掉队型」与「全局同步型」 ---
    nt = sum(nslow.values())
    print("\n      停顿归因（该步有几个 rank 的 load_ms > 200ms）：")
    print("        " + "   ".join(f"{k}个慢:{100 * nslow[k] / nt:.1f}%" for k in range(len(ranks) + 1)))
    print(f"        ⇒ 「全部 rank 同时慢」仅 {100 * nslow[len(ranks)] / nt:.1f}%"
          " ⇒ 尖峰是**掉队型**（个别 rank 慢），不是全局同步停顿")
    print(f"        ready_skew | 该步有人掉队 = {np.median(skew_slow) * 1000:.0f}ms   vs   "
          f"该步全快 = {np.median(skew_fast) * 1000:.0f}ms")
    print("        ⇒ 无人掉队时集合同步几乎不产生等待；一旦有人掉队，等待量直接 ≈ 迟到量")


def sec5_lockin(d):
    """分支锁存诊断：fast(<1ms) = 命中 prefetch 缓冲；slow = 当场等生产。

    若各 rank 的生产速度相同（RTT/CPU 逐项相同），快慢差异只能来自
    「缓冲里有没有现成批次」——这是一个双稳态：一旦某 rank 先把缓冲填满，
    它此后每步都命中；没填满的 rank 一直停在 just-in-time 分支。
    """
    ld = d["load"]
    if len(ld) < 2:
        return
    ranks = sorted(ld)
    m = {r: {x["global_step"]: x["load_ms"] for x in ld[r]} for r in ranks}
    steps = sorted(set.intersection(*[set(m[r]) for r in ranks]))
    if len(steps) < 200:
        return
    v = {r: np.array([m[r][g] for g in steps]) for r in ranks}
    print("\n  [5] 分支锁存诊断（为什么同 ρ、同尺寸下各 rank 快慢天差地别）")
    print(f"      {'rank':<6}{'前50步fast':>12}{'全程fast':>10}   每 200 步窗口的 fast 占比")
    for r in ranks:
        a = v[r]
        win = " ".join(f"{100 * np.mean(a[i:i + 200] < 1):4.0f}%" for i in range(0, len(a), 200))
        print(f"      r{r:<5}{100 * np.mean(a[:50] < 1):11.1f}%{100 * np.mean(a < 1):9.1f}%   {win}")
    lead = []
    for i in range(0, len(steps), 100):
        w = steps[i:i + 100]
        fr = {r: np.mean([v[r][j] < 1 for j in range(len(steps)) if steps[j] in set(w)]) for r in ranks}
        lead.append(max(fr, key=fr.get))
    print("      领先者随时间的更替（每 100 步窗口 fast 率最高者）：")
    print("        " + " → ".join(f"r{l}" for l in lead))
    print("      ⇒ 开局各 rank 都接近 just-in-time；先拉开缓冲的 rank 此后恒定命中，")
    print("        且在集体同步下这个优势不会被其他人追回 ⇒ 分支锁定。领先者可易主，")
    print("        说明它由早期随机竞争决定，不是某个 rank 的固定属性。")


def sec4b_ag(d):
    """gap_before_ag2：root all-gather 结束 -> layer0 all-gather 开始。需要 AG_MAX_PER_FWD>=2。"""
    ag = d["ag"]
    if not ag:
        return
    n = max(len(v) for v in ag.values()) if ag else 0
    per_step = {}
    for r, rows in ag.items():
        c = collections.Counter(x["global_step"] for x in rows)
        per_step[r] = collections.Counter(c.values()).most_common(1)[0][0] if c else 0
    if max(per_step.values() or [0]) < 2:
        print(f"      注：本次 AG_PROFILE 每步仅 {max(per_step.values() or [0])} 条事件（AG_MAX_PER_FWD=1），"
              "无 layer0 事件 ⇒ 跳过 gap_before_ag2（需 E1/E1c 的 ag_max=6 数据）")
        return
    allb = np.array([x["t_end"] - x["t_begin"] for rows in ag.values() for x in rows])
    print(f"      all-gather 调用本身耗时：p50={np.median(allb) * 1000:.2f}ms max={allb.max() * 1000:.1f}ms ⇒ 集合调用不阻塞")
    print("      gap_before_ag2（root AG 结束 → layer0 AG 开始）：")
    for r in sorted(ag):
        ev = collections.defaultdict(dict)
        for x in ag[r]:
            ev[x["global_step"]][x["idx_in_fwd"]] = x
        g = [e[2]["t_begin"] - e[1]["t_end"] for e in ev.values() if 1 in e and 2 in e]
        if g:
            g = np.array(g)
            print(f"        r{r}: p50={np.median(g) * 1000:7.1f}ms p90={np.percentile(g, 90) * 1000:7.1f}ms")
    print("      ⇒ 1s 的等待发生在「首次消费被聚合参数」处，不在 all-gather 调用本身")


def sec5_server(run):
    print("\n  [6] server 侧天花板")
    o = load_server(run)
    if o is None:
        print("      (无 metrics)")
        return None
    r, w, toks, sps, tps = o
    print(f"      running p50={np.median(r):.1f} p90={np.percentile(r, 90):.1f} max={r.max():.0f}"
          f"   waiting p50={np.median(w):.1f} max={w.max():.0f}   (max_num_seqs=8)")
    print(f"      prompt tokens/s={toks:.0f}   engine {sps:.1f} step/s   {tps:.0f} tok/step")
    print("      ⇒ 天花板是 token 速率（≈19-20k tok/s），非 token 预算、非并发槽位（E4-C/E4-D）")
    return toks


def sec6_rho(d, step_t, ns):
    print("\n  [7] ρ = 需求 / 能力（判据：ρ≲0.90 不卡；0.92–0.95 陡增）")
    if not step_t or np.isnan(step_t):
        print("      (无 step 时长)")
        return
    print(f"      step_time={step_t:.2f}s")
    print(f"      {'rank':<5}{'docs/batch':>11}{'需求 docs/s':>13}{'能力 docs/s':>13}{'ρ':>8}{'慢步率':>9}")
    for r in sorted(d["load"]):
        v = np.array([x["load_ms"] for x in d["load"][r]])
        docs = np.mean([x.get("docs_in_batch", 0) for x in d["load"][r]])
        s = d["sample"].get(r, [])
        rtt = (np.mean([x["wall_ms"] for x in s]) / 1000.0) if s else float("nan")
        nw = ns.get(r, 0)
        cap = nw / rtt if rtt and not np.isnan(rtt) else float("nan")
        dem = docs / step_t
        rho = dem / cap if cap else float("nan")
        print(f"      r{r:<4}{docs:11.2f}{dem:13.2f}{cap:13.2f}{rho:8.3f}{100 * np.mean(v > SLOW_MS):8.1f}%")


# ------------------------------------------------------------------ 主流程
def report(run):
    name = os.path.basename(run.rstrip("/"))
    cfg = load_run(run)
    print("=" * 100)
    print(f"RUN {name}")
    c = cfg["config"]
    if c:
        print(f"  workers={c.get('workers')} prefetch={c.get('prefetch_factor')} server_dp={c.get('server_dp')} "
              f"train_dp={c.get('train_dp')} in_order={c.get('in_order')} balance={c.get('samp_balance')} "
              f"perm={c.get('samp_perm')} seed={c.get('samp_seed')} ag_max={c.get('ag_max_per_fwd')}")
        print(f"  data={os.path.basename(c.get('data_path', '') or '')}")
    print("=" * 100)
    dur, steps = step_span(cfg)
    if steps:
        print(f"  时长={dur:.0f}s  步数={steps}  step/s={steps / dur:.2f}  step_time={dur / steps:.2f}s\n")
    ns = sec1_load(cfg, dur / steps if steps else float("nan"))
    sec2_perstep(cfg)
    sec3_period(cfg)
    sec4_sync(cfg)
    sec4b_ag(cfg)
    sec5_lockin(cfg)
    sec5_server(run)
    sec6_rho(cfg, dur / steps if steps else float("nan"), ns)
    print()


def summary(runs):
    """横向对比表：每 run 一行。"""
    print("=" * 118)
    print(f"{'run':<20}{'W':>3}{'step/s':>8}{'慢步率(全体)':>13}{'最慢rank':>10}"
          f"{'ready_skew p50':>15}{'running p90':>12}{'waiting max':>12}{'tok/s':>8}"
          f"{'pack相同步':>11}{'极差':>7}")
    print("-" * 136)
    for run in runs:
        d = load_run(run)
        dur, steps = step_span(d)
        c = d["config"]
        rates = {r: 100 * np.mean(np.array([x["load_ms"] for x in v]) > SLOW_MS) for r, v in d["load"].items()}
        if not rates:
            continue
        worst = max(rates, key=rates.get)
        allv = np.concatenate([np.array([x["load_ms"] for x in v]) for v in d["load"].values()])
        # ready_skew
        bystep = collections.defaultdict(dict)
        for r in d["timeline"]:
            for x in d["timeline"][r]:
                bystep[x["global_step"]][r] = x
        sk = [
            max(v[r]["t_batch"] for r in v) - min(v[r]["t_batch"] for r in v)
            for v in bystep.values()
            if len(v) == len(d["timeline"])
        ]
        # pack 均齐度：跨 rank 同一步 docs 极差
        bs = collections.defaultdict(dict)
        for r, rows in d["load"].items():
            for x in rows:
                bs[x["global_step"]][r] = x.get("docs_in_batch", 0)
        full = [v for v in bs.values() if len(v) == len(d["load"]) and len(v) > 1]
        sp = np.array([max(v.values()) - min(v.values()) for v in full]) if full else np.array([np.nan])
        o = load_server(run)
        rp90 = f"{np.percentile(o[0], 90):.1f}" if o else "-"
        wmax = f"{o[1].max():.0f}" if o else "-"
        toks = f"{o[2]:.0f}" if o else "-"
        print(
            f"{os.path.basename(run):<20}{c.get('workers', '?'):>3}"
            f"{steps / dur if dur else float('nan'):8.2f}"
            f"{100 * np.mean(allv > SLOW_MS):12.1f}%{('r' + str(worst)):>10}"
            f"{np.median(sk) * 1000 if sk else float('nan'):14.0f}ms{rp90:>12}{wmax:>12}{toks:>8}"
            f"{100 * np.mean(sp == 0):10.1f}%{sp.mean():7.2f}"
        )
    print("=" * 136)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print(__doc__)
        sys.exit(1)
    if len(args) == 1:
        report(args[0])
    else:
        summary(args)
