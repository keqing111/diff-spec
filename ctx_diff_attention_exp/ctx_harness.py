#!/usr/bin/env python3
"""ctx-only diff 模型 (ctxonly/checkpoints/2) 的采集 harness。

复用 archive eval/attention_experiment/draft_exp_common.py 的:
  - SpeculatorModel 加载 + 强制末尾锚点 + _last_anchors 记录
  - prepare_inputs / build_vocab / find_span / first_subword_draft_id / answer_slot
区别:
  - CKPT 指向 ctxonly/checkpoints/2 (diff@layer0 + context_only)
  - forward 后额外抓 diff 层 (layer0) 的 _debug_branch0/_debug_branch1/_debug_attn,
    以及各 GQA 层 (layer1-4) 的 _debug_attn (需要 capture_attention=True)
"""
import sys
from pathlib import Path

CKPT = "/home/y50063564/processed_data/dspark_data/dspark_diff_l0_5l_ctxonly/checkpoints/2"
COMMON = "/home/y50063564/processed_data/archive_dspark_20260904/eval/attention_experiment"
SRC = "/home/y50063564/dspark_project/speculators/src"
for p in (COMMON, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import torch_npu  # noqa: F401  (注册 npu 后端)

import draft_exp_common as dec  # noqa: E402

dec.CKPT = CKPT


def load(device: str = "npu:0"):
    """返回 (verifier, tokenizer, draft, vocab)。draft 已强制末尾锚点, 各层可 capture。"""
    verifier, tokenizer, draft = dec.load_models(device)
    vocab = dec.build_vocab(tokenizer)          # 排序后的 draft 词表 (list[int])
    # 打开所有层 self_attn 的注意力捕获 (diff 层会额外给 branch0/1)
    for layer in draft.layers:
        layer.self_attn.capture_attention = True
    return verifier, tokenizer, draft, vocab


def run_one(draft, aux, ids, loss_mask, vlast, doc_ids):
    """前向并返回 (logits, layer_debug)。

    layer_debug[l] = dict(keys):
        l==0(diff): 'b0','b1','attn'   (num_pairs x q x kv)
        l>=1(GQA) : 'attn'             (num_heads x q x kv)
    同时返回 logits [1, q, draft_vocab] (capture backbone_forward 的 r[1])。
    """
    from speculators.models.dflash.core import DFlashDraftModel

    cap_logits = {}
    orig_bb = DFlashDraftModel._backbone_forward
    def wbb(self, *a, **kw):
        r = orig_bb(self, *a, **kw)
        cap_logits["logits"] = r[1].detach().float().cpu()
        return r
    DFlashDraftModel._backbone_forward = wbb
    try:
        with torch.no_grad():
            draft(hidden_states=aux, input_ids=ids, loss_mask=loss_mask,
                  verifier_last_hidden_states=vlast, document_ids=doc_ids, max_anchors=256)
    finally:
        DFlashDraftModel._backbone_forward = orig_bb

    debug = {}
    for li, layer in enumerate(draft.layers):
        sa = layer.self_attn
        if hasattr(sa, "_debug_branch0") and sa._debug_branch0 is not None:
            debug[li] = {"b0": sa._debug_branch0[0].float(),
                         "b1": sa._debug_branch1[0].float(),
                         "attn": sa._debug_attn[0].float()}
        elif hasattr(sa, "_debug_attn") and sa._debug_attn is not None:
            debug[li] = {"attn": sa._debug_attn[0].float()}
        else:
            debug[li] = {}
    return cap_logits["logits"], debug


# ---- 共享度量函数 ----

def answer_row(q_len: int, block_size: int) -> int:
    """forced 锚点位于最末 anchor 元素; 答案=其 block 的 offset1。"""
    num_anchors = q_len // block_size
    return (num_anchors - 1) * block_size + 1


def _span_attn_single(a2d, row: int, span, T: int) -> float:
    """a2d: [units, q, kv]; 返回所有 unit 在该 row 对 key span(base区间) 注意力总和/unit 数。"""
    return float(a2d[:, row, span].sum() / a2d.shape[0])


def layer_key_share(dbg, li: int, row: int, span, T: int) -> dict:
    """逐层: 答案槽 row 对关键 token span 的注意力占比(与 exp1 的 head-mean 同口径)。
    L0(diff): 返回 branch(b0+b1)/32 与 diff(b0-b1)/16 两条。L1-4: GQA head-mean/32。
    """
    T = int(T)
    if li == 0:
        b0 = dbg[0]["b0"]; b1 = dbg[0]["b1"]; sub = dbg[0]["attn"]
        both = float((b0[:, row, span].sum() + b1[:, row, span].sum()) / (2 * b0.shape[0]))
        diff = float(sub[:, row, span].sum() / sub.shape[0])
        # 归一化版本(占该 row 对全部 base token 的注意力)
        b0tot = float(b0[:, row, :T].sum()); b1tot = float(b1[:, row, :T].sum())
        subtot = float(sub[:, row, :T].sum())
        return {"both_raw": both, "diff_raw": diff,
                "both_share": both / max((b0tot + b1tot) / (2 * b0.shape[0]), 1e-9),
                "diff_share": diff / max(subtot / sub.shape[0], 1e-9),
                "b0base_frac": b0tot / b0[:, row].sum() if b0[:, row].sum() > 0 else float("nan"),
                "b1base_frac": b1tot / b1[:, row].sum() if b1[:, row].sum() > 0 else float("nan")}
    a = dbg[li]["attn"]          # [heads, q, kv]
    raw = _span_attn_single(a, row, span, T)
    total = float(a[:, row, :T].sum() / a.shape[0])
    return {"both_raw": raw, "both_share": raw / max(total, 1e-9)}


def per_pair_key_attn(dbg, row: int, span, T: int) -> list[dict]:
    """横向: diff层(L0) 16 pair 在 row 对关键 span 的 b0/b1/(b0-b1) 注意力。"""
    b0 = dbg[0]["b0"]; b1 = dbg[0]["b1"]; sub = dbg[0]["attn"]
    out = []
    for p in range(b0.shape[0]):
        s0 = float(b0[p, row, span].sum()); s1 = float(b1[p, row, span].sum())
        ssub = float(sub[p, row, span].sum())
        out.append({"pair": p, "b0": s0, "b1": s1, "diff": ssub,
                    "b0_basefrac": float(b0[p, row, :T].sum()),
                    "b1_basefrac": float(b1[p, row, :T].sum())})
    return out


def jscan_key_share(dbg, q_len, block_size, span, T, j0=1, j1=None) -> dict:
    """答案槽 j=1..block-1 每位置: 每层关键token share (与纵向同口径)。"""
    a = q_len // block_size
    start = (a - 1) * block_size
    if j1 is None:
        j1 = block_size - 1
    res = {}
    for j in range(j0, j1 + 1):
        row = start + j
        if row >= q_len:
            continue
        res[j] = {li: layer_key_share(dbg, li, row, span, T) for li in range(5)}
    return res


if __name__ == "__main__":
    # 冒烟: 单样本, 验证加载/前向/形状
    device = "npu:0"
    torch.manual_seed(0)
    verifier, tokenizer, draft, vocab = load(device)
    print("draft type:", type(draft).__name__,
          "| n_layers:", len(draft.layers),
          "| block_size:", getattr(draft, "block_size", "?"))
    sa0 = draft.layers[0].self_attn
    print("layer0 self_attn:", type(sa0).__name__)

    text = ("The access code for Velora Station is Alpha. The old harbor once handled thousands of ships "
            "before the canal was built. A researcher on the expedition recorded the temperature. "
            "The capital of France is Paris and many tourists visit the Eiffel tower every year. "
            "The access code for Velora Station is")
    ids, aux, vlast, lm, doc, toks, T = dec.prepare_inputs(verifier, tokenizer, text, device)
    logits, dbg = run_one(draft, aux, ids, lm, vlast, doc)
    print("T(seq)=", T, "logits", tuple(logits.shape))
    for li in sorted(dbg):
        ks = {k: tuple(v.shape) for k, v in dbg[li].items()}
        print(f" layer{li} debug keys:", ks)
    # anchors 是 forward 时记录的
    ans_first = dec.first_subword_draft_id(tokenizer, vocab, "Alpha")
    print("Alpha first-subword draft_id =", ans_first,
          "tok =", tokenizer.decode([vocab[ans_first]]) if ans_first is not None else "?")
    q_len = logits.shape[1]
    num_anchors = q_len // draft.block_size          # 分配的 anchor 数
    print("num_anchors(alloc) =", num_anchors)
    # 校准: forced 锚点在最末 anchor 元素(索引 num_anchors-1), 答案 pos=seq -> 其 block 的 offset1
    a = num_anchors - 1
    # 扫描该 block 全部 offset 及邻域, 看 P(Alpha)/argmax
    for off in range(-1, draft.block_size + 2):
        r = a * draft.block_size + off
        if not (0 <= r < q_len):
            continue
        pr = torch.softmax(logits[0, r], dim=-1)
        pA = float(pr[ans_first]) if ans_first is not None else None
        arg = int(pr.argmax())
        print(f"  row={r} (off={off}): P(Alpha)={pA:.4f} argmax={tokenizer.decode([vocab[arg]])!r} (p={pr[arg]:.3f})")
    print("SMOKE OK")
