"""导出 diff 层 layer0 在最后 query 的逐 base-token 注意力数值(非热力图)。

读取 attn_plots/layer0_diff_branches.pt, 打印:
  1. 各 pair 在最后 query 对每个 base token 的 branch0/branch1/相减 注意力
  2. 逐 token 完整表(可指定 pair) + 零值结构统计
用法: python scripts/show_attn_per_token.py [--pair 0] [--all]
"""
import argparse
import torch

PT = "/home/y50063564/processed_data/dspark_data/attn_plots/layer0_diff_branches.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--pair", type=int, default=-1, help="只打印某个 pair 的完整逐token表; -1=全部摘要")
ap.add_argument("--all", action="store_true", help="所有 pair 都打印完整逐token表")
args = ap.parse_args()

d = torch.load(PT)
b0, b1, sub = d["branch0"], d["branch1"], d["subtracted"]
tokens = d["tokens"]
seq = d["kv_len"] - d["q_len"]
q = d["q_len"] - 1  # 最后 query

def fmt(v):
    return f"{v:.4f}" if abs(v) >= 1e-4 else f"{v:.1e}"

print(f"最后 query q={q}, base tokens={seq}")
pairs = range(16)

# 1) 全 pair 摘要: 每个 pair 在每个 base token 上的注意力, 看零值结构
print("\n=== 各 pair 在 q=%d 的注意力零值结构 ===" % q)
print("pair | b0总和 | b1总和 | b0非零(>1e-6) | b1非零 | diff总和 | diff<0个数")
for p in pairs:
    t0 = b0[p, q, :seq]; t1 = b1[p, q, :seq]; ts = sub[p, q, :seq]
    print(f"{p:4d} | {t0.sum():.3f} | {t1.sum():.3f} | {(t0>1e-6).sum():2d} | "
          f"{(t1>1e-6).sum():2d} | {ts.sum():+.3f} | {(ts<0).sum():2d}")

# 2) 完整逐 token 表
def print_table(p):
    t0 = b0[p, q, :seq]; t1 = b1[p, q, :seq]; ts = sub[p, q, :seq]
    print(f"\n=== pair{p} 逐 base token 注意力 (q={q}) ===")
    print("pos | token        |   branch0 |   branch1 |      diff")
    for i in range(seq):
        print(f"{i:3d} | {tokens[i]:<12s} | {fmt(t0[i]):>9s} | {fmt(t1[i]):>9s} | {fmt(ts[i]):>9s}")

if args.all:
    for p in pairs:
        print_table(p)
elif args.pair >= 0:
    print_table(args.pair)
else:
    # 默认: 打印 pair0 和 pair15(用户提到最后一个 pair 不同) 全表
    print_table(0)
    print_table(15)
print("\ndone")
