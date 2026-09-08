#!/usr/bin/env python3
import json
from pathlib import Path
BASE = Path(__file__).parent

v1 = json.load(open(BASE / "results_exp1/exp1_vertical.json"))
h1 = json.load(open(BASE / "results_exp1/exp1_horizontal.json"))
j1 = json.load(open(BASE / "results_exp1/exp1_jscan.json"))
s2 = json.load(open(BASE / "results_exp2/exp2_summary.json"))
s3 = json.load(open(BASE / "results_exp3/exp3_summary.json"))

print("=== exp1 纵向: 关键token注意力% (答案槽)  行=(L,pct) 列 L0(b0+b1)/L0(diff)/L1-L4 + P ===")
print(f"{'cell':12s} " + " ".join(f"{c:>7s}" for c in ["L0b", "L0d", "L1", "L2", "L3", "L4", "P"]))
for r in v1:
    l = r["layer"]
    vals = [l["0"]["both_raw"], l["0"]["diff_raw"]] + [l[str(k)]["both_raw"] for k in range(1, 5)]
    p = r["p_first_subword"]
    print(f"L{r['length']}p{r['position_pct']:<3d}   " + " ".join(f"{x*100:7.2f}" for x in vals)
          + f"  {p:5.3f}  pred={r['argmax_token']!r}")

print("\n=== exp1 横向 512/50: 每 pair b0/b1/diff 对关键token的注意力 (raw, 16 pairs) ===")
print("pair   b0      b1      diff")
for rr in h1["L512_p50"]:
    print(f"{rr['pair']:3d}  {rr['b0']*100:6.2f}  {rr['b1']*100:6.2f}  {rr['diff']*100:+6.2f}")

print("\n=== exp1 j-scan (512/50): 关键token注意力% vs j (L0b=b0+b1, L0d=diff, L1-L4) ===")
js = {int(k): v for k, v in j1["L512_p50"].items()}
print(f"{'j':>2s} " + " ".join(f"{c:>7s}" for c in ["L0b", "L0d", "L1", "L2", "L3", "L4"]))
for j in sorted(js):
    row = js[j]
    vals = [row["0"]["both_raw"] * 100, row["0"]["diff_raw"] * 100] + \
           [row[str(k)]["both_raw"] * 100 for k in range(1, 5)]
    print(f"{j:2d} " + " ".join(f"{x:7.2f}" for x in vals))

print("\n=== exp2 summary ===")
print("n | TT TF FT FF | P(true|tgt_ok) hit(all) hit(tgt_ok) | #tgtX_draftOK")
for n in sorted(s2, key=int):
    c = s2[n]["cells"]
    print(f"{int(n):2d} | {c['TT']:2d} {c['TF']:2d} {c['FT']:2d} {c['FF']:2d} | "
          f"{'-' if s2[n]['p_true_tc'] is None else round(s2[n]['p_true_tc'],3)} "
          f"{round(s2[n]['hit_top1_all'],2)} "
          f"{round(s2[n]['hit_top1_tc'],2) if s2[n]['hit_top1_tc'] is not None else '-'} | {c['FT']}")

print("\n=== exp3 summary (variant/query) ===")
for k in s3:
    c = s3[k]["cells"]
    print(f"{k:16s} n={s3[k]['n']:2d} TT/TF/FT/FF={c['TT']}/{c['TF']}/{c['FT']}/{c['FF']}  "
          f"pExp|tgtOK={'-' if s3[k]['p_exp|tc'] is None else round(s3[k]['p_exp|tc'],3)}  "
          f"attnNew={s3[k]['attn_new-mean']:.2f} attnOld={s3[k]['attn_old-mean']:.2f}")
