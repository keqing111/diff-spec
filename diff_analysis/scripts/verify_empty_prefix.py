"""验证 context-only mask 是否会让某些 query 没有合法注意力目标(空 base-prefix)。
从训练数据取样本, 构造 context-only(include_same_block=False) 的 mask, 统计
每个 query 可关注的 base token 数, 找出为 0 的 query。
用法: ASCEND_RT_VISIBLE_DEVICES=11 python scripts/verify_empty_prefix.py
"""
import sys

import torch

sys.path.insert(0, "/home/y50063564/dspark_project/speculators/src")
torch.manual_seed(0)

from torch.nn.attention.flex_attention import create_mask  # noqa: E402

from speculators.models.dflash.attention import create_anchor_block_mask_mod  # noqa: E402
from speculators.models.dflash.utils import select_anchors  # noqa: E402

def lengths_to_doc(lengths, total):
    doc = torch.full((total,), -1, dtype=torch.long)
    doc[: lengths.sum()] = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return doc

# 模拟打包序列: 多个 doc
for doc_lengths in [[8, 8, 8], [10, 6], [3, 12], [16]]:
    total = sum(doc_lengths)
    doc = lengths_to_doc(torch.tensor(doc_lengths), total)
    loss_mask = torch.ones(1, total)
    anchors, valid = select_anchors(loss_mask, num_anchors=2, block_size=4)
    for incl_block in (True, False):
        m, q_len, kv_len = create_anchor_block_mask_mod(
            document_ids=doc, total_seq_len=total, anchor_positions=anchors,
            block_size=4, sliding_window=None, include_same_block=incl_block)
        mask = create_mask(m, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=torch.device("cpu"))
        # 每个 query 可关注的 base token 数(0..total)
        n_per_query = mask[0, 0, :, :total].sum(dim=-1)
        empty = (n_per_query == 0).sum().item()
        tag = "含block" if incl_block else "context-only"
        if empty:
            print(f"doc_len={doc_lengths} anchors={anchors.tolist()} [{tag}]: {empty}/{q_len} 个 query 无任何 base 目标 ← NaN根源")
        else:
            print(f"doc_len={doc_lengths} anchors={anchors.tolist()} [{tag}]: 所有 query 都有 base 目标")
