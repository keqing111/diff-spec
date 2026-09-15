#!/usr/bin/env python3
"""Poll the vLLM /metrics endpoint every second and append raw scrapes as JSONL.

Usage: scrape_metrics.py --url http://host:port/metrics --out <dir>/metrics.jsonl [--interval 1]
Each line: {"w": wall_time, "mono": perf_counter, "m": {family: {labelstr: value}}}
Counters are kept raw (cumulative); interval rates are computed offline.
"""
import argparse
import json
import re
import time
import urllib.request

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE.+-]+|NaN|Inf|-Inf)$')


def parse(text):
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = LINE.match(line.strip())
        if not m:
            continue
        fam, labels, val = m.group(1), m.group(2) or "", m.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        out.setdefault(fam, {})[labels] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with open(a.out, "a", buffering=1) as f:
        while True:
            t0 = time.perf_counter()
            try:
                with opener.open(a.url, timeout=5) as r:
                    body = r.read().decode("utf-8", "replace")
                f.write(json.dumps({"w": time.time(), "mono": t0, "m": parse(body)}) + "\n")
            except Exception as exc:
                f.write(json.dumps({"w": time.time(), "mono": t0, "error": str(exc)}) + "\n")
            dt = a.interval - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
