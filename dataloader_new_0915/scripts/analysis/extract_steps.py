#!/usr/bin/env python3
"""Tolerant extractor of per-step train metrics from a speculators console log.

A "block" runs from a line beginning with a timestamp up to the next such line,
so tqdm/progress lines inside a block are harmless. Key=value pairs anywhere in
the block are collected; a block is a training record iff it has global_step and
train/loss. Used for the P0 neutrality comparison.
"""
import argparse
import re
import sys

START = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]")
KV = re.compile(r"([\w./]+)=([^,\s]+)")
KEYS = ["train/loss", "train/accept_len", "train/ce_loss", "train/tv_loss"]


def extract(path):
    """One training record = the key=value pairs accumulated since the previous
    `global_step=` line, closed by that line (the logger prints global_step last)."""
    recs = {}
    acc: list[tuple[str, str]] = []
    for line in open(path, encoding="utf-8", errors="replace"):
        for k, v in KV.findall(line):
            if k == "global_step":
                kv = dict(acc)
                try:
                    gs = int(v)
                except ValueError:
                    acc = []
                    continue
                if "train/loss" in kv:
                    recs[gs] = [kv.get(key, "NA") for key in KEYS]
                acc = []
            else:
                acc.append((k, v))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", action="store_true",
                    help="compare all logs bit-for-bit on the common global_steps")
    a = ap.parse_args()
    parsed = {p: extract(p) for p in a.logs}
    if a.out:
        first = parsed[a.logs[0]]
        with open(a.out, "w") as f:
            for gs in sorted(first):
                f.write("\t".join([str(gs)] + first[gs]) + "\n")
        print(f"wrote {a.out} ({len(first)} steps)")
    for p, r in parsed.items():
        print(f"{p}: {len(r)} steps  gs {min(r) if r else '-'}..{max(r) if r else '-'}")
    if a.compare and len(a.logs) >= 2:
        base = parsed[a.logs[0]]
        ok_all = True
        for p in a.logs[1:]:
            other = parsed[p]
            common = sorted(set(base) & set(other))
            bad = [gs for gs in common if base[gs] != other[gs]]
            print(f"compare {a.logs[0].split('/')[-2]} vs {p.split('/')[-2]}: "
                  f"common={len(common)} mismatches={len(bad)} "
                  f"-> {'**逐位一致 ✓**' if not bad else '**差异 ✗**'}")
            for gs in bad[:5]:
                print(f"   gs={gs}: {base[gs]} vs {other[gs]}")
            ok_all = ok_all and not bad
        print("OVERALL:", "IDENTICAL" if ok_all else "DIFFERENT")


if __name__ == "__main__":
    main()
