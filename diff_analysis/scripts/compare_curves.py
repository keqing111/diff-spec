#!/usr/bin/env python3
"""多份 DSpark 训练日志曲线对比绘图(基于 reconstruct_metrics.py 的解析逻辑)。

用法:
    python3 compare_curves.py <log1> <log2> [...] \
        [--tags tag1,tag2,...] [-o <输出目录>] [--smooth N] [--zoom <step>]

把多份日志(如 baseline 与 dspark_diff)的 loss / accept_len / accept_rate /
full_acc / ce_loss 曲线画到同一张图上。默认同时输出:
    compare_<metric>.png          — 完整 global_step 范围
    compare_zoom_<metric>.png     — 限制 x 轴到 [0, min(各日志最大step)]
                                   (便于对比被提前中断的短 run 与 baseline 早期)
不改动原 reconstruct_metrics.py。
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# 数值列(转 float); 其余的按 str 保留 (与 reconstruct_metrics.py 一致)
FLOAT_KEYS = {
    "train/confidence_loss", "train/confidence_abs_error",
    "train/confidence_pred_mean", "train/confidence_cumprod_bias",
    "train/loss", "train/ce_loss", "train/tv_loss",
    "train/accept_rate", "train/accept_len", "train/full_acc",
    *(f"train/position_{k}_acc" for k in range(1, 8)),
    "profile/fetch_ms", "profile/fwd_ms", "profile/bwd_ms",
    "profile/opt_ms", "profile/step_ms", "profile/tokens_per_s",
    "profile/fetch_frac", "lr/Muon", "lr/AdamW",
    "val/confidence_loss_epoch", "val/confidence_abs_error_epoch",
    "val/confidence_pred_mean_epoch", "val/confidence_cumprod_bias_epoch",
    "val/loss_epoch", "val/ce_loss_epoch", "val/tv_loss_epoch",
    "val/accept_rate_epoch", "val/accept_len_epoch", "val/full_acc_epoch",
    *(f"val/position_{k}_acc_epoch" for k in range(1, 8)),
}
INT_KEYS = {"global_step", "epoch"}

START_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] INFO\s+(train|val)/")
CONT_RE = re.compile(r"^\s+\S+=")
KV_RE = re.compile(r"([\w./]+)=([^,\s]+)")


def parse_log(log_path: str) -> list[dict]:
    """把日志解析成指标记录列表(每个 train 步 / 每个 val epoch 一条)。"""
    blocks: list[dict] = []
    current: dict | None = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = START_RE.match(line)
            if m:
                if current:
                    blocks.append(current)
                current = {"_ts": m.group(1), "_type": m.group(2), "_text": line}
                continue
            if current is not None and CONT_RE.match(line):
                current["_text"] += line
                continue
            if current is not None:
                blocks.append(current)
                current = None
    if current:
        blocks.append(current)

    records = []
    for b in blocks:
        pairs = dict(KV_RE.findall(b["_text"]))
        rec: dict = {"_ts": b["_ts"], "_type": b["_type"]}
        for k, v in pairs.items():
            if k in INT_KEYS:
                try:
                    rec[k] = int(v)
                except ValueError:
                    continue
            elif k in FLOAT_KEYS:
                try:
                    rec[k] = float(v)
                except ValueError:
                    continue
            else:
                rec[k] = v
        records.append(rec)
    return records


def to_frame(records: list[dict], typ: str) -> pd.DataFrame:
    recs = [r for r in records if r.get("_type") == typ]
    df = pd.DataFrame(recs)
    if "global_step" in df:
        df = df.sort_values("global_step").reset_index(drop=True)
    return df


COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def plot_series(ax, df, metric, label, color, smooth, ylims=None):
    """画一条(含平滑)曲线; 返回该序列值范围。"""
    if metric not in df or df[metric].isna().all():
        return None
    s = pd.to_numeric(df[metric], errors="coerce")
    x = df["global_step"] if "global_step" in df else df.index
    ax.plot(x, s, color=color, alpha=0.22, lw=0.8, label=f"{label} (raw)")
    if smooth > 1 and len(s) > smooth:
        sm = s.rolling(smooth, center=True, min_periods=1).mean()
        ax.plot(x, sm, color=color, lw=1.8, label=f"{label} (smooth)")
    else:
        ax.plot(x, s, color=color, lw=1.8, label=label)
    return s.min(), s.max()


def steps_per_epoch(df: pd.DataFrame) -> float:
    """第一处 epoch 从 0 -> 1 时的 global_step, 作为单个 epoch 的步数估计。"""
    if "epoch" not in df or "global_step" not in df:
        return float(df["global_step"].max()) or 1.0
    prev = None
    for _, r in df.iterrows():
        if r["epoch"] != prev:
            if prev == 0 and r["epoch"] == 1:
                return float(r["global_step"])
            prev = r["epoch"]
    return float(df["global_step"].max()) or 1.0


def add_epoch_lines(ax, df, max_step):
    """对每个 >1 epoch 的序列在 epoch 边界画轻竖线。"""
    if "epoch" not in df or "global_step" not in df:
        return
    prev = None
    for _, r in df.iterrows():
        if r["epoch"] != prev:
            if prev is not None and r["global_step"] <= max_step:
                ax.axvline(r["global_step"], color="gray", ls="--", lw=0.6, alpha=0.35)
            prev = r["epoch"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="训练日志路径(>=2 份)")
    ap.add_argument("--tags", default="", help="逗号分隔标签, 与日志一一对应(默认取文件名)")
    ap.add_argument("-o", "--outdir", default=None, help="输出目录(默认=第一个日志同目录)")
    ap.add_argument("--smooth", type=int, default=201, help="平滑窗口")
    ap.add_argument("--zoom", type=int, default=None,
                    help="缩放图的 x 上限(默认=各日志最小最大 step)")
    ap.add_argument("--epochs", action="store_true",
                    help="用 epoch 数作横轴(每份日志按其 steps_per_epoch 归一化), "
                    "便于对比不同 world_size 的训练速率")
    ap.add_argument("--samples", type=float, default=None,
                    help="数据集总样本数; 设置后横轴 = 已见样本数 "
                    "(global_step / steps_per_epoch * samples)")
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    if not tags:
        tags = [Path(p).stem for p in args.logs]
    if len(tags) != len(args.logs):
        tags = [Path(p).stem for p in args.logs]

    outdir = Path(args.outdir) if args.outdir else Path(args.logs[0]).parent
    outdir.mkdir(parents=True, exist_ok=True)

    frames, vframes = [], []
    print(f"解析 {len(args.logs)} 份日志 ...")
    for log, tag in zip(args.logs, tags):
        recs = parse_log(log)
        tr = to_frame(recs, "train")
        va = to_frame(recs, "val")
        frames.append((tag, tr))
        vframes.append((tag, va))
        print(f"  [{tag}] train {len(tr)} 条, val {len(va)} 条, "
              f"step {tr['global_step'].min()}-{tr['global_step'].max()}, "
              f"epoch {sorted(tr['epoch'].unique())}")

    # x 轴单位: --samples 时横轴=已见样本数; --epochs 时=epoch 数; 否则=global_step。
    spe = {tag: steps_per_epoch(tr) for tag, tr in frames}

    def x_of(tag, tr):
        gs = tr["global_step"] if "global_step" in tr else tr.index
        if args.samples:
            return (gs / spe[tag] * args.samples).astype(float)
        if args.epochs:
            return (gs / spe[tag]).astype(float)
        return gs

    if args.samples:
        xlabel = "samples seen"
    elif args.epochs:
        xlabel = "epochs"
    else:
        xlabel = "global_step"
    common_max = min(x_of(tag, fr).max() for tag, fr in frames if not fr.empty)
    zoom_max = args.zoom if args.zoom else common_max
    print(f"\n公共窗口 = {xlabel} [0, {common_max:.0f}]\n")

    metrics = [
        ("train/loss", "loss"),
        ("train/ce_loss", "ce_loss"),
        ("train/tv_loss", "tv_loss"),
        ("train/accept_len", "accept_len"),
        ("train/accept_rate", "accept_rate"),
        ("train/full_acc", "full_acc"),
    ]

    rows = []
    for tag, tr in frames:
        row = {"run": tag, "steps": len(tr),
               "last_step": tr["global_step"].iloc[-1],
               "last_epoch": int(tr["epoch"].iloc[-1]),
               "steps/epoch": round(spe[tag], 1)}
        for m, short in metrics:
            if m in tr:
                w = tr[x_of(tag, tr) <= common_max]
                row[f"{short}_mean[0:{common_max:.2f}]"] = w[m].mean()
                row[f"{short}_last"] = tr[m].iloc[-1]
        rows.append(row)
    cmp = pd.DataFrame(rows)
    print(cmp.round(4).to_string(index=False))
    cmp.to_csv(outdir / "compare_summary.csv", index=False)
    print(f"\n已写 {outdir/'compare_summary.csv'}")

    # 每个 metric: 一张全范围图 + 一张缩放图
    for m, short in metrics:
        if all(m not in fr or fr[m].isna().all() for _, fr in frames):
            continue
        for kind, xmax in (("", None), ("zoom_", zoom_max)):
            fig, ax = plt.subplots(figsize=(12, 5))
            for i, (tag, tr) in enumerate(frames):
                color = COLORS[i % len(COLORS)]
                x = x_of(tag, tr)
                if args.samples or args.epochs:
                    ax.plot(x, tr[m], color=color, alpha=0.22, lw=0.8,
                            label=f"{tag} (raw)")
                    if args.smooth > 1 and len(tr[m]) > args.smooth:
                        sm = tr[m].rolling(args.smooth, center=True, min_periods=1).mean()
                        ax.plot(x, sm, color=color, lw=1.8, label=f"{tag} (smooth)")
                    else:
                        ax.plot(x, tr[m], color=color, lw=1.8, label=tag)
                else:
                    plot_series(ax, tr, m, tag, color, args.smooth)
                add_epoch_lines(ax, tr, xmax or tr["global_step"].max())
            ax.axvline(common_max, color="gray", ls=":", lw=1, alpha=0.7)
            ax.text(common_max, ax.get_ylim()[1], f"min-max({common_max:.2f})",
                    ha="left", va="top", fontsize=8, color="gray")
            ax.set_xlabel(xlabel)
            if xmax:
                ax.set_xlim(0, xmax)
            ax.set_ylabel(short)
            ax.set_title(f"compare {short} vs {xlabel}")
            ax.legend(loc="upper right" if "loss" in short or "ce" in short else "lower right")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / f"compare_{kind}{short}.png", dpi=150)
            plt.close(fig)
            print(f"已写 compare_{kind}{short}.png")


if __name__ == "__main__":
    main()
