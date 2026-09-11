#!/usr/bin/env python3
"""Poll /proc for the processes of interest every second and append JSONL.

Usage: scrape_sys.py --out <dir>/sys.jsonl [--interval 1] [--pattern speculators/scripts/train.py] [--pattern vllm]
Per line: {"w": wall, "mono": perf, "procs": {pid: {...}}, "vmstat": {...}, "pressure": {...}}
"""
import argparse
import glob
import json
import os
import time


def pids_matching(patterns):
    out = []
    for p in glob.glob("/proc/[0-9]*"):
        try:
            cmd = open(f"{p}/cmdline", "rb").read().decode("utf-8", "replace")
        except OSError:
            continue
        if any(pat in cmd for pat in patterns):
            out.append(int(os.path.basename(p)))
    return sorted(set(out))


def read_proc(pid):
    rec = {}
    try:
        st = open(f"/proc/{pid}/stat").read().split()
        # utime(14) stime(15) minflt(10) majflt(12) rss(24) in clock ticks/pages
        rec["utime"] = int(st[13]); rec["stime"] = int(st[14])
        rec["minflt"] = int(st[9]); rec["majflt"] = int(st[11])
        rec["rss_pages"] = int(st[23]); rec["threads"] = int(st[19])
    except Exception:
        pass
    try:
        sched = open(f"/proc/{pid}/schedstat").read().split()
        rec["run_ns"] = int(sched[0]); rec["wait_ns"] = int(sched[1]); rec["slices"] = int(sched[2])
    except Exception:
        pass
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith(("VmRSS:", "voluntary_ctxt_switches:", "nonvoluntary_ctxt_switches:")):
                k, v = line.split(":", 1)
                rec[k] = v.strip()
    except Exception:
        pass
    return rec


def read_vmstat():
    out = {}
    try:
        for line in open("/proc/vmstat"):
            k, v = line.split()
            if k in ("pswpin", "pswpout", "pgmajfault", "pgfault", "nr_free_pages"):
                out[k] = int(v)
    except Exception:
        pass
    return out


def read_pressure():
    out = {}
    for name in ("cpu", "io", "memory"):
        try:
            for line in open(f"/proc/pressure/{name}"):
                kind = line.split()[0]
                vals = dict(kv.split("=") for kv in line.split()[1:])
                out[f"{name}.{kind}"] = vals
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--pattern", action="append", default=[])
    a = ap.parse_args()
    pats = a.pattern or ["speculators/scripts/train.py", "vllm"]
    with open(a.out, "a", buffering=1) as f:
        while True:
            t0 = time.perf_counter()
            rec = {"w": time.time(), "mono": t0, "procs": {}, "vmstat": read_vmstat(),
                   "pressure": read_pressure()}
            for pid in pids_matching(pats):
                rec["procs"][str(pid)] = read_proc(pid)
            f.write(json.dumps(rec) + "\n")
            dt = a.interval - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
