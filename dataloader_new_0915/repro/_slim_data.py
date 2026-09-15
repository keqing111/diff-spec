#!/usr/bin/env python3
"""把 dataloader_diag/exp 下 17 个决定性 run 精简复制到本文件夹 data/。

精简规则（只保留分析脚本真正读取的字段/文件）：
  rank*/load_rank*.jsonl      全字段保留（每行 ~300B）
  rank*/timeline_rank*.jsonl  全字段保留
  rank*/ag_rank*.jsonl        全字段保留（边⑤的定位证据）
  rank*/worker*.jsonl         kind=sample/collate 保留
  rank*/bins_*.jsonl          仅 E1 / E1c 保留（analyze_buffer.py 用）
  rank*/agwait_*.jsonl        仅 E1c 保留
  metrics.jsonl (70MB)   ->   metrics_slim.jsonl   只留 6 个 counter
  sys.jsonl     (8MB)    ->   丢弃（E7 混淆审计结论已固化在 EVIDENCE.md I 节）

用法：python3 _slim_data.py [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys

SRC = "/home/y50063564/dataloader_diag/exp"
DST = "/home/y50063564/dataloader_new_0915/data"

RUNS = [
    "E1_ref_w2_pf4",
    "E1c_agwait_ref",
    "E3_w1_pf4",
    "E3_w4_pf4",
    "E3_w6_pf4",
    "E3_w8_pf4",
    "E3_w12_pf4",
    "E4C_w8_tok16384",
    "E4C_w8_tok32768",
    "E4D_w8_seqs16",
    "E4D_w8_seqs32",
    "E5_balance_docs",
    "E5b_skew_docs",
    "E6_perm_rev",
    "E7_band_w4",
    "E7_band_w8",
    "E7b_band_w2",
]

# analyze_rootcause / run_E3 / run_E4C / run_E4D / run_E7 只读这几项
KEEP_METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_waiting_by_reason",
    "vllm:prompt_tokens_total",
    "vllm:iteration_tokens_total_count",
    "vllm:iteration_tokens_total_sum",
}

BINS_RUNS = {"E1_ref_w2_pf4", "E1c_agwait_ref"}
AGWAIT_RUNS = {"E1c_agwait_ref"}


def slim_metrics(src: str, dst: str) -> tuple[int, int]:
    n_in = n_out = 0
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            n_in += 1
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = x.get("m", {})
            keep = {k: v for k, v in m.items() if k in KEEP_METRICS}
            if not keep:
                continue
            fo.write(json.dumps({"w": x["w"], "mono": x.get("mono"), "m": keep}) + "\n")
            n_out += 1
    return n_in, n_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = 0
    for run in RUNS:
        src, dst = f"{SRC}/{run}", f"{DST}/{run}"
        if not os.path.isdir(src):
            print(f"!! missing {src}", file=sys.stderr)
            continue
        if not args.dry_run:
            os.makedirs(dst, exist_ok=True)
        run_bytes = 0

        for name in ("config.txt", "run.log"):
            p = f"{src}/{name}"
            if os.path.exists(p):
                if not args.dry_run:
                    shutil.copy2(p, f"{dst}/{name}")
                run_bytes += os.path.getsize(p)

        for rankdir in sorted(glob.glob(f"{src}/rank*")):
            rk = os.path.basename(rankdir)
            if not args.dry_run:
                os.makedirs(f"{dst}/{rk}", exist_ok=True)
            for f in sorted(glob.glob(f"{rankdir}/*.jsonl")):
                base = os.path.basename(f)
                if base.startswith("bins_") and run not in BINS_RUNS:
                    continue
                if base.startswith("agwait_") and run not in AGWAIT_RUNS:
                    continue
                if base.startswith("metrics"):
                    continue
                if not args.dry_run:
                    shutil.copy2(f, f"{dst}/{rk}/{base}")
                run_bytes += os.path.getsize(f)

        m_src = f"{src}/metrics.jsonl"
        if os.path.exists(m_src):
            if args.dry_run:
                run_bytes += os.path.getsize(m_src) * 0.02
            else:
                n_in, n_out = slim_metrics(m_src, f"{dst}/metrics_slim.jsonl")
                run_bytes += os.path.getsize(f"{dst}/metrics_slim.jsonl")
                print(f"  {run}: metrics {n_in} -> {n_out} 行")

        total += run_bytes
        print(f"{run:<20} {run_bytes / 1e6:8.1f} MB")

    print(f"\n合计 {total / 1e6:.0f} MB" + ("  (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
