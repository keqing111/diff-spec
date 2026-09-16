#!/usr/bin/env python3
"""Unit-check the ctx-only mask fix on a synthetic packed sequence.

Verifies, for ctx-only (include_same_block=False):
  legacy (anchor_in_context=True) : the anchor's own base column IS visible
  fixed  (anchor_in_context=False): it is NOT, and the visible set is still non-empty
and that the normal (include_same_block=True) path is unchanged.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/home/y50063564/dspark_project/speculators/src")
from speculators.models.dflash.attention import (  # noqa: E402
    create_anchor_block_mask_mod,
)

TOTAL = 32
BLOCK = 4
ANCHOR = 8
WINDOW = 2048
device = torch.device("cpu")

document_ids = torch.zeros(TOTAL, dtype=torch.long, device=device)  # ONE document of 32 tokens
anchor_positions = torch.tensor([ANCHOR], device=device)


def visible_rows(**kw):
    mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
        document_ids=document_ids,
        total_seq_len=TOTAL,
        anchor_positions=anchor_positions,
        block_size=BLOCK,
        sliding_window=WINDOW,
        sliding_window_non_causal=False,
        **kw,
    )
    q_idx = torch.arange(q_len, device=device, dtype=torch.long)[:, None]
    kv_idx = torch.arange(kv_len, device=device, dtype=torch.long)[None, :]
    m = mask_mod(torch.tensor(0), torch.tensor(0), q_idx, kv_idx)
    return m, q_len, kv_len


def report(name, m, q_len, kv_len):
    print(f"--- {name} ---")
    print(f"    q_len={q_len} kv_len={kv_len} (kv>= {TOTAL} 是 synthetic 段)")
    for q in range(q_len):
        cols = torch.nonzero(m[q], as_tuple=False).squeeze(-1).tolist()
        n_synth = sum(1 for c in cols if c >= TOTAL)
        has_anchor = ANCHOR in cols
        print(
            f"    q{q}: 可见 {len(cols):3d} 列 | synthetic {n_synth} 个 | "
            f"含 anchor({ANCHOR})={has_anchor} | 可见列={cols}"
        )
    return m


print("=" * 78)
m_norm, q, k = visible_rows(include_same_block=True)
report("普通掩码 include_same_block=True（应保持不变）", m_norm, q, k)

print()
m_leg, q, k = visible_rows(include_same_block=False, anchor_in_context=True)
report("ctx-only 旧行为 anchor_in_context=True（应有 anchor 泄漏）", m_leg, q, k)

print()
m_fix, q, k = visible_rows(include_same_block=False, anchor_in_context=False)
report("ctx-only 修复后 anchor_in_context=False（应无 anchor）", m_fix, q, k)

print()
print("=" * 78)
print("断言")
print("=" * 78)

ok = True


def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


# 普通掩码：q0 只看 kv<8 无 synthetic；qi 还能看到自己 block 的 0..i
check("普通: q0 不含 anchor 列", not m_norm[0][ANCHOR])
check("普通: q0 看到自己的 synthetic 槽位(1 个)", int(m_norm[0][TOTAL:].sum()) == 1)
check("普通: q3 能看到 block 的 0..3", m_norm[3][TOTAL : TOTAL + 4].all())
check("普通: q0 能看到 base[0..7] 共 8 列", int(m_norm[0][:ANCHOR].sum()) == ANCHOR)
check("普通: q3 看不到 block 的第 4 个（因果）", not m_norm[3][TOTAL + 4:].any())

# 旧行为：能看到 anchor 列，且看不到 synthetic
check("旧行为: 每一行都含 anchor 列", all(m_leg[i][ANCHOR] for i in range(4)))
check("旧行为: 完全看不到 synthetic", not m_leg[:, TOTAL:].any())

# 修复后：看不到 anchor 列，也看不到 synthetic，但仍然非空
check("修复后: 每一行都不含 anchor 列", all(not m_fix[i][ANCHOR] for i in range(4)))
check("修复后: 完全看不到 synthetic", not m_fix[:, TOTAL:].any())
check("修复后: 每一行可见集非空（否则 softmax 全 -inf → NaN）",
      all(m_fix[i].any() for i in range(4)))
check("修复后: 行内容 = base[ANCHOR-WINDOW .. ANCHOR-1]",
      all(m_fix[i][:ANCHOR].all() and not m_fix[i][ANCHOR:].any() for i in range(4)))

# 文档首 token 作 anchor 时，修复后为空 —— 正是 select_anchors 必须排除的情况
anchor0 = torch.tensor([0], device=device)
mask_mod0, q0, k0 = create_anchor_block_mask_mod(
    document_ids=document_ids, total_seq_len=TOTAL, anchor_positions=anchor0,
    block_size=BLOCK, sliding_window=WINDOW, sliding_window_non_causal=False,
    include_same_block=False, anchor_in_context=False,
)
qi = torch.arange(q0, device=device, dtype=torch.long)[:, None]
ki = torch.arange(k0, device=device, dtype=torch.long)[None, :]
m0 = mask_mod0(torch.tensor(0), torch.tensor(0), qi, ki)
check("anchor=文档首 token 且修复后：可见集为空（所以必须靠 select_anchors 排除）",
      not m0.any())

print()
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
