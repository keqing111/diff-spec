#!/usr/bin/env python3
"""Root-cause analysis for the periodic FSDP2 forward spike (parts 1-4).

Usage:
  analyze_rootcause.py chain  --raw <RAW_DIR> [--out <OUT_DIR>]
  analyze_rootcause.py cost   --raw <RAW_DIR> [--out <OUT_DIR>]
  analyze_rootcause.py xcorr  --raw <RAW_DIR> [--out <OUT_DIR>] [--rho-server X]
  analyze_rootcause.py queue  --raw <RAW_DIR> [--out <OUT_DIR>]

Input schema (per rank under <RAW_DIR>/rank{R}/):
  load_rank{R}.jsonl      kind=load     : gs, load_ms, t0,t1,w0,w1, docs_in_batch, tokens_in_batch, packed_tokens, batch_seq?
  timeline_rank{R}.jsonl  kind=timeline : gs, docs_in_batch, tokens_in_batch, packed_tokens, load_ms,
                                          t_batch,t_fwd,t_bwd,t_opt,t_end (+ w_*)
  ag_rank{R}.jsonl        kind=ag       : gs, layer, idx_in_fwd, in_bytes,out_bytes, t_begin,t_end
  worker*.jsonl           kind=sample   : worker, index, wall_ms, cpu_ms, seq_len, t_perf, w
                          kind=collate  : worker, n_docs, total_tokens, packed_tokens, wall_ms, t_perf, w
No torch.distributed calls; this is pure offline analysis.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)


# ---------------------------------------------------------------- loading
def _read(paths):
    rows = []
    for f in paths:
        for line in open(f, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def load_all(raw):
    data = {}
    for r in range(16):
        rd = f"{raw}/rank{r}"
        if not os.path.isdir(rd):
            continue
        d = {}
        for kind, pat in (("load", "load_rank*.jsonl"), ("timeline", "timeline_rank*.jsonl"),
                          ("ag", "ag_rank*.jsonl")):
            df = _read(glob.glob(f"{rd}/{pat}"))
            d[kind] = df
        wdf = _read(glob.glob(f"{rd}/worker*.jsonl"))
        if len(wdf):
            d["sample"] = wdf[wdf["kind"] == "sample"].copy()
            d["collate"] = wdf[wdf["kind"] == "collate"].copy()
        else:
            d["sample"] = pd.DataFrame(); d["collate"] = pd.DataFrame()
        bdf = _read(glob.glob(f"{rd}/bins*.jsonl"))
        d["bins"] = bdf
        data[r] = d
    return data


def _steps_df(d):
    """Per-rank per-step table from the timeline events."""
    t = d["timeline"]
    if not len(t):
        return pd.DataFrame()
    t = t.sort_values("global_step").drop_duplicates("global_step").reset_index(drop=True)
    cols = [c for c in ["global_step", "docs_in_batch", "tokens_in_batch", "packed_tokens",
                        "load_ms", "t_batch", "t_fwd", "t_end"] if c in t.columns]
    return t[cols].copy()


def spike_info(d):
    """Per-rank spike definition + late-rank identity from the load events."""
    l = d["load"]
    if not len(l):
        return pd.DataFrame()
    l = l.sort_values("global_step").drop_duplicates("global_step").reset_index(drop=True)
    med = l["load_ms"].median()
    thr = max(200.0, 5.0 * med)
    l["spike"] = l["load_ms"] > thr
    return l, med, thr


# ---------------------------------------------------------------- part 1
def part_chain(raw, out):
    os.makedirs(out, exist_ok=True)
    data = load_all(raw)
    ranks = sorted(data)
    if not ranks:
        print("no data"); return

    # assemble per-step matrix across ranks
    piv, late, skew = {}, {}, {}
    meds = {}
    for r in ranks:
        st = _steps_df(data[r])
        if not len(st):
            continue
        piv[r] = st.set_index("global_step")
        l, med, thr = spike_info(data[r])
        meds[r] = (med, thr)
        late[r] = l.set_index("global_step")["load_ms"]
    common = sorted(set.intersection(*[set(v.index) for v in piv.values()])) if piv else []
    print(f"common steps across ranks: {len(common)}")

    load_mat = pd.DataFrame({r: late[r].reindex(common) for r in ranks})
    docs_mat = pd.DataFrame({r: piv[r]["docs_in_batch"].reindex(common) for r in ranks})
    batch_mat = pd.DataFrame({r: piv[r]["t_batch"].reindex(common) for r in ranks})
    fwd_mat = pd.DataFrame({r: piv[r]["t_fwd"].reindex(common) for r in ranks})

    late_rank = load_mat.idxmax(axis=1)
    sorted_load = np.sort(load_mat.values, axis=1)
    second_over_max = np.divide(sorted_load[:, -2], np.maximum(sorted_load[:, -1], 1e-9))
    ready_skew = (batch_mat.max(axis=1) - batch_mat.min(axis=1)) * 1000.0
    fwd_dur = (fwd_mat - batch_mat) * 1000.0

    any_spike = pd.Series(
        [any(load_mat.loc[s, r] > meds[r][1] for r in ranks) for s in common], index=common)

    print("\n=== Part 1: spike + late-rank statistics ===")
    print(f"  steps={len(common)}  spike steps={int(any_spike.sum())} ({any_spike.mean()*100:.1f}%)")
    if any_spike.sum():
        lr = late_rank[any_spike.values]
        print("  late rank distribution on spike steps:", lr.value_counts().to_dict())
        print(f"  second_latest/latest ratio on spike steps: median={np.nanmedian(second_over_max[any_spike.values]):.2f}")
        frac_multi = np.nanmean(second_over_max[any_spike.values] > 0.7)
        print(f"  fraction of spike steps with >=2 ranks nearly tied (>0.7): {frac_multi*100:.1f}%  "
              f"[H expects ~0]")
    print(f"  ready_skew(ms): p50={ready_skew.median():.0f} p90={ready_skew.quantile(.9):.0f} max={ready_skew.max():.0f}")
    print(f"  per-rank load_ms median: {{{', '.join(f'{r}:{meds[r][0]:.1f}' for r in ranks)}}}")
    print(f"  per-rank fwd duration median (ms): {{{', '.join(f'{r}:{fwd_dur[r].median():.0f}' for r in ranks)}}}")

    # FSDP all-gather part (needs AG_PROFILE)
    ag = {}
    for r in ranks:
        a = data[r]["ag"]
        if len(a):
            a = a[a["idx_in_fwd"] == 1].copy()
            a["gs"] = a["global_step"].astype(int)
            ag[r] = a.sort_values("gs").drop_duplicates("gs").set_index("gs")
    if ag:
        ag_begin = pd.DataFrame({r: ag[r]["t_begin"].reindex(common) for r in ranks})
        ag_end = pd.DataFrame({r: ag[r]["t_end"].reindex(common) for r in ranks})
        ag_block = (ag_end - ag_begin) * 1000.0
        pre_ag = (ag_begin - batch_mat) * 1000.0
        post_ag = (fwd_mat - ag_end) * 1000.0
        print("\n=== Part 1b: FSDP first all-gather decomposition (ms) ===")
        for nm, m in (("pre_ag (t_batch->ag_begin)", pre_ag), ("ag_block (collective)", ag_block),
                      ("post_ag (ag_end->fwd_end)", post_ag)):
            print(f"  {nm}: " + ", ".join(f"r{r}: p50={m[r].median():.0f} p90={m[r].quantile(.9):.0f}" for r in ranks))
        print(f"  spread of ag_end (ms): p50={((ag_end.max(axis=1)-ag_end.min(axis=1))*1000).median():.1f}"
              f"   spread of ag_begin: p50={((ag_begin.max(axis=1)-ag_begin.min(axis=1))*1000).median():.1f}"
              f"   [H expects ag_end to converge, ag_begin not to]")
        # regression: ag_block[r] ~ ready_skew - (t_batch[r]-min t_batch)
        early = [r for r in ranks if r != late_rank.mode().iloc[0]]
        if early:
            target = (ready_skew - (batch_mat[early].sub(batch_mat[early].min(axis=1), axis=0)) * 1000.0)
            x = ag_block[early].stack().values
            y = target.stack().values
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() > 50:
                slope, intercept = np.polyfit(y[m], x[m], 1)
                r2 = np.corrcoef(y[m], x[m])[0, 1] ** 2
                print(f"  regression ag_block ~ (ready_skew - rank_lateness): slope={slope:.2f} R2={r2:.2f}"
                      f"  [H expects slope~1, R2 high]")
        ag_block.to_csv(f"{out}/ag_block.csv")
        pre_ag.to_csv(f"{out}/pre_ag.csv")

    # spike raw evidence
    spikes = [s for s in common if any_spike.loc[s]]
    raw_rows = []
    for s in spikes:
        for r in ranks:
            raw_rows.append({"gs": s, "rank": r, "load_ms": load_mat.loc[s, r],
                             "docs": docs_mat.loc[s, r], "t_batch": batch_mat.loc[s, r],
                             "fwd_dur_ms": fwd_dur.loc[s, r], "is_late": late_rank.loc[s] == r})
    pd.DataFrame(raw_rows).to_csv(f"{out}/chain_spikes.csv", index=False)

    # periodicity
    print("\n=== Part 1c: periodicity of load_ms ===")
    for r in ranks:
        x = load_mat[r].values.astype(float)
        x = x - np.nanmean(x)
        ac = np.correlate(x, x, "full")[len(x)-1:]
        ac = ac / (ac[0] + 1e-12)
        peaks = sorted([l for l in range(2, min(60, len(ac)))
                        if ac[l] > ac[l-1] and ac[l] >= ac[l+1]], key=lambda l: -ac[l])[:3]
        sp = np.diff([s for s in common if load_mat.loc[s, r] > meds[r][1]])
        print(f"  rank{r}: autocorr peaks={[(l, round(float(ac[l]),2)) for l in peaks]}"
              f"  spike count={len(sp)}  median spacing={np.median(sp) if len(sp) else float('nan')}")

    summary = {
        "steps": len(common), "spike_steps": int(any_spike.sum()),
        "late_rank_counts": late_rank[any_spike.values].value_counts().to_dict() if any_spike.sum() else {},
        "second_over_max_median_on_spike": float(np.nanmedian(second_over_max[any_spike.values])) if any_spike.sum() else None,
        "frac_nearly_tied": float(frac_multi) if any_spike.sum() else None,
        "ready_skew_p50_ms": float(ready_skew.median()),
        "ready_skew_p90_ms": float(ready_skew.quantile(.9)),
        "per_rank_load_p50": {r: float(meds[r][0]) for r in ranks},
        "per_rank_fwd_dur_p50": {r: float(fwd_dur[r].median()) for r in ranks},
    }
    with open(f"{out}/chain_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _plot_chain(common, load_mat, docs_mat, fwd_dur, ready_skew, f"{out}/chain.png")
    print(f"\nwrote {out}/chain_summary.json, chain_spikes.csv, chain.png")


def _plot_chain(common, load_mat, docs_mat, fwd_dur, ready_skew, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    ax = axes[0]
    for r in load_mat.columns:
        ax.plot(common, load_mat[r], lw=0.7, label=f"rank{r} load_ms")
    ax.set_ylabel("next(loader) wait (ms)"); ax.legend(fontsize=7, ncol=4); ax.grid(alpha=.25)
    axes[1].plot(common, ready_skew, lw=0.8, color="#d62728")
    axes[1].set_ylabel("ready skew (ms)"); axes[1].grid(alpha=.25)
    for r in docs_mat.columns:
        axes[2].plot(common, docs_mat[r], lw=0.7, label=f"rank{r}")
    axes[2].set_ylabel("docs/batch"); axes[2].legend(fontsize=7, ncol=4); axes[2].grid(alpha=.25)
    for r in fwd_dur.columns:
        axes[3].plot(common, fwd_dur[r], lw=0.7, label=f"rank{r}")
    axes[3].set_ylabel("fwd duration (ms)"); axes[3].set_xlabel("global_step")
    axes[3].legend(fontsize=7, ncol=4); axes[3].grid(alpha=.25)
    fig.suptitle("Causal time chain: per-rank next(loader) wait / ready skew / docs per batch / fwd duration")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)


# ---------------------------------------------------------------- part 2
def part_cost(raw, out):
    """Per-batch production cost vs docs/tokens/request latency.

    Alignment: collate events are production-ordered per rank (sorted by the
    monotonic clock); with in_order=True the first N produced batches are the N
    consumed ones (validated by the P0-4b backlog check), so zipping production
    order with the consumed step order is exact.
    """
    os.makedirs(out, exist_ok=True)
    data = load_all(raw)
    rows = []
    for r in sorted(data):
        d = data[r]
        if not len(d["collate"]) or not len(d["timeline"]):
            continue
        tl = d["timeline"].sort_values("global_step").reset_index(drop=True)
        col = d["collate"].copy()
        col["tprod"] = col["t_perf"] if "t_perf" in col.columns else col["t"]
        col = col.sort_values("tprod").reset_index(drop=True)
        smp = d["sample"].copy()
        if "t_perf" not in smp.columns:
            continue
        # samples grouped per collate, per worker (samples of a batch are
        # consecutive within the worker that produced it)
        per_worker = {}
        for w, gw in smp.groupby("worker"):
            gw = gw.sort_values("t_perf").reset_index(drop=True)
            per_worker[w] = gw
        cursor = {w: 0 for w in per_worker}
        n = min(len(col), len(tl))
        for i in range(n):
            c = col.iloc[i]
            w = int(c["worker"]); nd = int(c.get("n_docs", 0) or 0)
            gw = per_worker.get(w)
            chunk = gw.iloc[cursor[w]: cursor[w] + nd] if gw is not None else None
            if gw is not None:
                cursor[w] += nd
            req_ms = float(chunk["wall_ms"].sum()) if chunk is not None and len(chunk) else np.nan
            row = {
                "rank": r, "step": int(tl.loc[i, "global_step"]),
                "n_docs": nd,
                "total_tokens": c.get("total_tokens", np.nan),
                "packed_tokens": c.get("packed_tokens", np.nan),
                "collate_ms": c.get("wall_ms", np.nan),
                "sample_sum_ms": req_ms,
                "sample_max_ms": float(chunk["wall_ms"].max()) if chunk is not None and len(chunk) else np.nan,
                "sample_mean_tok": float(chunk["seq_len"].mean()) if chunk is not None and "seq_len" in chunk and len(chunk) else np.nan,
                "load_ms": float(tl.loc[i, "load_ms"]),
                "docs_tl": int(tl.loc[i, "docs_in_batch"]) if "docs_in_batch" in tl.columns else -1,
            }
            rows.append(row)
    if not rows:
        print("cost: no usable data"); return
    df = pd.DataFrame(rows)
    print(f"batches joined: {len(df)}")
    ok = (df["n_docs"] == df["docs_tl"]).mean() if "docs_tl" in df else float("nan")
    print(f"  production/consumption alignment check (n_docs == docs_in_batch): {ok*100:.1f}%")
    df["bucket"] = df["n_docs"].apply(lambda k: str(int(k)) if k < 8 else "8+")
    g = df.groupby("bucket").agg(n=("n_docs","count"), docs=("n_docs","mean"),
                                 tok=("total_tokens","mean"),
                                 sample_sum_p50=("sample_sum_ms","median"),
                                 sample_sum_p90=("sample_sum_ms",lambda x: x.quantile(.9)),
                                 collate_p50=("collate_ms","median"),
                                 load_p50=("load_ms","median"),
                                 load_p90=("load_ms",lambda x: x.quantile(.9)))
    print("\n=== Part 2: cost per docs/batch bucket ===")
    print(g.to_string())
    d2 = df.dropna(subset=["load_ms","n_docs"])
    if len(d2) > 50:
        y = d2["load_ms"].values
        def fit(X, k):
            b,*_ = np.linalg.lstsq(X, y, rcond=None)
            rss = float(((y - X@b)**2).sum()); n_, = (len(y),)
            return b, n_*np.log(max(rss,1e-12)/n_) + 2*k, 1-rss/float(((y-y.mean())**2).sum())
        X1 = np.column_stack([np.ones(len(d2)), d2["n_docs"].values])
        X2 = np.column_stack([np.ones(len(d2)), d2["sample_sum_ms"].fillna(d2["sample_sum_ms"].median()).values])
        X3 = np.column_stack([np.ones(len(d2)), d2["n_docs"].values,
                              d2["total_tokens"].fillna(d2["total_tokens"].median()).values])
        b1,a1,r1 = fit(X1,2); b2,a2,r2 = fit(X2,2); b3,a3,r3 = fit(X3,3)
        print(f"\n  C1 = docs only        : slope={b1[1]:.2f} ms/doc   AIC={a1:.0f}  R2={r1:.3f}")
        print(f"  C2 = sum(sample lat)  : slope={b2[1]:.3f}          AIC={a2:.0f}  R2={r2:.3f}")
        print(f"  C3 = docs+tokens      : a={b3[0]:.1f} b={b3[1]:.2f} c={b3[2]:.5f}  AIC={a3:.0f}  R2={r3:.3f}")
        best = min([('docs',a1),('sample_latency',a2),('docs+tokens',a3)], key=lambda t:t[1])
        print(f"  -> best AIC: {best[0]}   [H needs docs/tokens to win; if sample-latency wins, edge 1 is refuted]")
    g.to_csv(f"{out}/cost_buckets.csv"); df.to_csv(f"{out}/cost_batches.csv", index=False)
    print(f"wrote {out}/cost_buckets.csv, cost_batches.csv")


# ---------------------------------------------------------------- part 3
def part_xcorr(raw, out, rho_server=None):
    os.makedirs(out, exist_ok=True)
    data = load_all(raw)
    KS = [2, 4, 8, 16, 32]
    LAGS = list(range(0, 17))
    from scipy.stats import spearmanr
    allrows = []
    for r in sorted(data):
        t = data[r]["timeline"]
        if not len(t) or "docs_in_batch" not in t.columns:
            continue
        t = t.sort_values("global_step").drop_duplicates("global_step").reset_index(drop=True)
        docs = t["docs_in_batch"].astype(float)
        load = t["load_ms"].astype(float)
        for K in KS:
            C = docs.rolling(K, min_periods=1).sum()
            for lag in LAGS:
                x = C.shift(lag)
                m = x.notna() & load.notna()
                if m.sum() < 50:
                    continue
                rho, p = spearmanr(load[m], x[m])
                allrows.append({"rank": r, "K": K, "lag": lag, "n": int(m.sum()),
                                "rho": float(rho), "p": float(p)})
    if not allrows:
        print("xcorr: no data"); return
    res = pd.DataFrame(allrows)
    # Holm correction over the whole grid per rank
    res["p_holm"] = np.nan
    for r, sub in res.groupby("rank"):
        p = sub["p"].values
        order = np.argsort(p)
        adj = np.empty_like(p)
        m = len(p)
        running = 0.0
        for i, oi in enumerate(order):
            val = (m - i) * p[oi]
            running = max(running, val)
            adj[oi] = min(1.0, running)
        res.loc[sub.index, "p_holm"] = adj
    print("\n=== Part 3: cross-correlation load_ms(i) vs C_K(i-lag) (docs/depletion) ===")
    for r, sub in res.groupby("rank"):
        best = sub.sort_values("rho", ascending=False).head(5)
        print(f"\n rank{r}: top (K,lag,rho,p_holm,n):")
        for _, b in best.iterrows():
            print(f"    K={int(b.K):>2} lag={int(b.lag):>2} rho={b.rho:+.3f} p_holm={b.p_holm:.2e} n={int(b.n)}")
        sig = sub[(sub.lag >= 1) & (sub.p_holm < 0.01) & (sub.rho > 0.3)]
        print(f"  -> {'结构成立' if len(sig) else '**无滞后显著结构**'} (lag>=1 & p_holm<0.01 & rho>0.3 的格点数={len(sig)})")
        base = sub[sub.lag == 0].sort_values("rho", ascending=False).head(1)
        if len(base):
            b = base.iloc[0]
            print(f"  (lag=0 最强: K={int(b.K)} rho={b.rho:+.3f})")
    res.to_csv(f"{out}/xcorr.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ranks = sorted(res["rank"].unique())
        fig, axes = plt.subplots(1, len(ranks), figsize=(5 * len(ranks), 4.2), sharey=True)
        if len(ranks) == 1:
            axes = [axes]
        for ax, r in zip(axes, ranks):
            sub = res[res["rank"] == r]
            im = ax.imshow(sub.pivot(index="K", columns="lag", values="rho").values,
                           aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
            ax.set_title(f"rank{r}"); ax.set_xlabel("lag Δ"); ax.set_ylabel("K")
            ax.set_xticks(range(len(LAGS))); ax.set_xticklabels(LAGS)
            ax.set_yticks(range(len(KS))); ax.set_yticklabels(KS)
        fig.colorbar(im, ax=axes, shrink=0.8, label="spearman rho")
        fig.suptitle("Part 3: corr( load_ms(i), sum_{last K} docs )(i-Δ)")
        fig.savefig(f"{out}/xcorr_heatmap.png", dpi=130)
        print(f"wrote {out}/xcorr.csv, xcorr_heatmap.png")
    except Exception as exc:
        print(f"plot failed: {exc}")


# ---------------------------------------------------------------- part 4
def part_queue(raw, out):
    os.makedirs(out, exist_ok=True)
    data = load_all(raw)
    for r in sorted(data):
        d = data[r]
        if not len(d["collate"]) or not len(d["load"]):
            continue
        prod = d["collate"].copy()
        tcol = "t_perf" if "t_perf" in prod.columns else "t"
        prod = prod.sort_values(tcol)
        cons = d["load"].copy()
        # NOTE: must use the *monotonic* clock on both sides (collate.t_perf,
        # load.t1); load.w1 is CLOCK_REALTIME and is not comparable to t_perf.
        ccol = "t1" if "t1" in cons.columns else ("t_perf" if "t_perf" in cons.columns else "t")
        cons = cons.sort_values(ccol)
        rows = []
        pi = 0
        for _, c in cons.iterrows():
            tau = c[ccol]
            while pi < len(prod) and prod.iloc[pi][tcol] <= tau:
                pi += 1
            rows.append({"gs": c["global_step"], "consumed": len(rows) + 1, "produced": pi,
                         "depth": pi - (len(rows) + 1), "load_ms": c.get("load_ms", np.nan)})
        q = pd.DataFrame(rows)
        q.to_csv(f"{out}/depth_rank{r}.csv", index=False)
        print(f"rank{r}: depth p50={q.depth.median():.1f} min={q.depth.min()} max={q.depth.max()} "
              f"(hard cap = num_workers*prefetch_factor)")
    print(f"wrote {out}/depth_rank*.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["chain", "cost", "xcorr", "queue"])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rho-server", type=float, default=None)
    a = ap.parse_args()
    out = a.out or f"/home/y50063564/dataloader_diag/analysis/{a.part}"
    {"chain": lambda: part_chain(a.raw, out),
     "cost": lambda: part_cost(a.raw, out),
     "xcorr": lambda: part_xcorr(a.raw, out, a.rho_server),
     "queue": lambda: part_queue(a.raw, out)}[a.part]()


if __name__ == "__main__":
    main()
