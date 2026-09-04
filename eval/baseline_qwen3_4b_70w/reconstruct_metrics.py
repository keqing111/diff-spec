#!/usr/bin/env python3
"""从 DSpark 训练原始日志还原指标并绘图。

用法:
    python3 reconstruct_metrics.py <train_logs.log> [-o <输出目录>] [--tag <标签>]

输入: 训练的原始 stdout 日志(52MB 级, loguru 换行包装格式)。
      每个指标块以一行 "[HH:MM:SS] INFO  train/..." 或 "val/..._epoch=..." 开头,
      后续为缩进的 key=value 续行, 到下一个带时间戳的日志行为止。

输出(写到输出目录, 默认 = 日志同目录):
    train_metrics.csv   — 每步 train 指标 (global_step, epoch, loss, accept_len, ...)
    val_metrics.csv     — 每个 epoch 末的 val 指标 (epoch, loss_epoch, accept_len_epoch, ...)
    loss_curve.png      — loss 随 global_step 的变化 (train 曲线 + 平滑 + val 点)
    accept_len_curve.png — accept_len 随 global_step 的变化
    epoch_summary.png   — 每个 epoch 的 accept_len / loss 汇总对比 (train均值 vs val)

对比 baseline 与创新时: 分别跑本脚本 (加 --tag 区分), 再对 CSV 作图即可。
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# 数值列(转 float); 其余的按 str 保留
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
    if not df.empty and "_ts" in df:
        df = df.sort_values("_ts").reset_index(drop=True)
    # 用 global_step 排序(更可靠)
    if "global_step" in df:
        df = df.sort_values("global_step").reset_index(drop=True)
    return df


def plot_curve(ax, steps, values, label, color, window=201):
    ax.plot(steps, values, color=color, alpha=0.25, lw=0.8, label=f"{label} (raw)")
    if window > 1 and len(values) > window:
        s = pd.Series(values).rolling(window, center=True, min_periods=1).mean()
        ax.plot(steps, s, color=color, lw=1.8, label=f"{label} (smooth w={window})")
    else:
        ax.plot(steps, values, color=color, lw=1.8, label=label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="训练日志路径")
    ap.add_argument("-o", "--outdir", default=None, help="输出目录(默认=日志同目录)")
    ap.add_argument("--tag", default="", help="标签(叠加到图标题/文件名)")
    ap.add_argument("--smooth", type=int, default=201, help="平滑窗口(默认 201)")
    args = ap.parse_args()

    log_path = Path(args.log)
    outdir = Path(args.outdir) if args.outdir else log_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f" [{args.tag}]" if args.tag else ""

    print(f"解析 {log_path} ...")
    records = parse_log(str(log_path))
    train = to_frame(records, "train")
    val = to_frame(records, "val")
    print(f"train 记录 {len(train)} 条, val 记录 {len(val)} 条")
    if train.empty and val.empty:
        print("未解析到任何指标块, 请检查日志格式。", file=sys.stderr)
        sys.exit(1)

    # 保存 CSV
    train.to_csv(outdir / "train_metrics.csv", index=False)
    val.to_csv(outdir / "val_metrics.csv", index=False)
    print(f"已写 {outdir/'train_metrics.csv'} 和 {outdir/'val_metrics.csv'}")

    # ---- epoch 边界(global_step 变化处) ----
    epoch_bounds = []
    if "epoch" in train and "global_step" in train:
        prev_e = None
        for _, r in train.iterrows():
            if r["epoch"] != prev_e:
                epoch_bounds.append((r["global_step"], r["epoch"]))
                prev_e = r["epoch"]
        if len(epoch_bounds) > 1:
            print("epoch 边界(global_step, epoch):", epoch_bounds)

    def add_epoch_lines(ax):
        for gs, ep in epoch_bounds[1:]:  # 跳过第 0 个
            ax.axvline(gs, color="gray", ls="--", lw=0.8, alpha=0.7)
            ax.text(gs, ax.get_ylim()[1], f"ep{ep}", ha="left", va="top",
                    fontsize=8, color="gray")

    # ---- 图 1: accept_len ----
    if "train/accept_len" in train:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = train["global_step"] if "global_step" in train else train.index
        plot_curve(ax, x, train["train/accept_len"], "accept_len", "#1f77b4", args.smooth)
        if not val.empty and "val/accept_len_epoch" in val:
            vx = val["global_step"] if "global_step" in val else val.index
            ax.plot(vx, val["val/accept_len_epoch"], "o-", color="crimson",
                    lw=1.5, label="val accept_len (epoch end)")
            for _, r in val.iterrows():
                ep = int(r.get("epoch", -1))
                ax.annotate(f"ep{ep}: {r['val/accept_len_epoch']:.3f}",
                            (r.get("global_step", r.name), r["val/accept_len_epoch"]),
                            textcoords="offset points", xytext=(0, 8),
                            fontsize=8, color="crimson")
        add_epoch_lines(ax)
        ax.set_xlabel("global_step"); ax.set_ylabel("accept_len")
        ax.set_title(f"accept_len vs step{tag}")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(outdir / "accept_len_curve.png", dpi=150)
        plt.close(fig)
        print("已写 accept_len_curve.png")

    # ---- 图 2: loss ----
    if "train/loss" in train:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = train["global_step"] if "global_step" in train else train.index
        plot_curve(ax, x, train["train/loss"], "loss", "#d62728", args.smooth)
        if not val.empty and "val/loss_epoch" in val:
            vx = val["global_step"] if "global_step" in val else val.index
            ax.plot(vx, val["val/loss_epoch"], "s-", color="darkorange",
                    lw=1.5, label="val loss (epoch end)")
        add_epoch_lines(ax)
        ax.set_xlabel("global_step"); ax.set_ylabel("loss")
        ax.set_title(f"loss vs step{tag}")
        ax.legend(loc="upper right"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(outdir / "loss_curve.png", dpi=150)
        plt.close(fig)
        print("已写 loss_curve.png")

    # ---- 图 3: 每个 epoch 的 accept_len / loss 汇总 ----
    if "epoch" in train and not val.empty:
        g = train.groupby("epoch")[["train/accept_len", "train/loss"]].mean()
        cols = []
        for _, r in val.iterrows():
            ep = int(r.get("epoch", -1))
            cols.append({
                "epoch": ep,
                "val_accept_len": r.get("val/accept_len_epoch", float("nan")),
                "val_loss": r.get("val/loss_epoch", float("nan")),
            })
        vdf = pd.DataFrame(cols).set_index("epoch")
        summ = g.join(vdf, how="outer").sort_index()

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        a1.plot(summ.index, summ["train/accept_len"], "o-", label="train mean")
        if "val_accept_len" in summ:
            a1.plot(summ.index, summ["val_accept_len"], "s-", label="val (epoch end)")
        a1.set_xlabel("epoch"); a1.set_ylabel("accept_len")
        a1.set_title(f"accept_len per epoch{tag}"); a1.legend(); a1.grid(alpha=0.3)
        a2.plot(summ.index, summ["train/loss"], "o-", label="train mean")
        if "val_loss" in summ:
            a2.plot(summ.index, summ["val_loss"], "s-", label="val (epoch end)")
        a2.set_xlabel("epoch"); a2.set_ylabel("loss")
        a2.set_title(f"loss per epoch{tag}"); a2.legend(); a2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(outdir / "epoch_summary.png", dpi=150)
        plt.close(fig)
        print("已写 epoch_summary.png")
        print("\n=== 每个 epoch 汇总(均值) ===")
        print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
