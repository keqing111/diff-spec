#!/usr/bin/env python3
"""实验3变体: 更新推理的深入探究。

维度:
  A) 上下文长度: 短(只留关键句) vs 长(512)
  B) 旧码: Beta(单token, 词表内)  [与 Delta 可比]
  C) 新码出现次数: 用不同句式说 "现在是 Delta" 1/2/4 次

对每个配置: 问新码/问旧码, 报 target 与 draft 的 P(Delta)/P(Beta) 及每层注意力。

用法:
    ASCEND_RT_VISIBLE_DEVICES=15 python3 exp3_variations.py
"""
import torch
import torch_npu  # noqa: F401
from draft_exp_common import (load_models, build_vocab, run_forward, answer_slot,
                             find_span, prepare_inputs, first_subword_draft_id, FILLER_POOL)

OLD, NEW = "Beta", "Delta"

NEW_STMTS = [
    "After the security update, the access code for Velora Station was changed to Delta.",
    "The current access code for Velora Station is now Delta.",
    "The access code for Velora Station has been updated to Delta.",
    "Velora Station now uses the access code Delta.",
    "From today on, the access code for Velora Station is Delta.",
    "The new access code for Velora Station is Delta.",
]
OLD_STMT = "The access code for Velora Station was Beta."
NEW_COMP = "After the security update, the access code for Velora Station is"
OLD_COMP = "Before the security update, the access code for Velora Station was"


def build_prompt(n_new: int, long_ctx: bool, tokenizer) -> str:
    stmts = [OLD_STMT] + NEW_STMTS[:n_new]
    prompt = " ".join(stmts)
    if long_ctx:
        fi = 0
        while len(tokenizer.encode(prompt + " " + NEW_COMP)) < 512:
            prompt += " " + FILLER_POOL[fi % len(FILLER_POOL)]
            fi += 1
    return prompt  # 补全在 run 时拼接


def run_case(verifier, tokenizer, draft, vocab, prompt, comp, expected, device):
    p_full = prompt + " " + comp
    ids, aux, vlast, lm, doc, toks, T = prepare_inputs(verifier, tokenizer, p_full, device)
    # target
    with torch.no_grad():
        tlogits = verifier(ids).logits[0, -1]
    t_arg = int(tlogits.argmax().item())
    t_tok = tokenizer.decode([t_arg])
    eid = tokenizer.encode(" " + expected)[-1]
    t_p = torch.softmax(tlogits, dim=-1)[eid].item()
    # draft
    cap = run_forward(draft, aux, ids, lm, vlast, doc)
    slot = answer_slot(draft)
    d_probs = torch.softmax(cap["logits"][0, slot], dim=-1)
    d_arg = int(d_probs.argmax().item())
    d_tok = tokenizer.decode([vocab[d_arg]]) if d_arg < len(vocab) else "?"
    new_did = first_subword_draft_id(tokenizer, vocab, NEW)
    old_did = first_subword_draft_id(tokenizer, vocab, OLD)
    d_p_new = d_probs[new_did].item() if new_did is not None else None
    d_p_old = d_probs[old_did].item() if old_did is not None else None
    # 每层注意力(新/旧码)
    new_span = find_span(toks, NEW)
    old_span = find_span(toks, OLD)
    per = {"new": [], "old": []}
    for l in range(5):
        a = cap["layer_attn"][l][0, :, slot, :T].mean(0)
        per["new"].append(a[new_span].sum().item() if new_span else 0.0)
        per["old"].append(a[old_span].sum().item() if old_span else 0.0)
    return {"target_argmax": t_tok, "target_p_expected": t_p,
            "draft_argmax": d_tok, "draft_p_new": d_p_new, "draft_p_old": d_p_old,
            "new_attn": per["new"], "old_attn": per["old"], "T": T}


def main():
    torch.manual_seed(42)
    device = "npu:0"
    verifier, tokenizer, draft = load_models(device)
    vocab = build_vocab(tokenizer)

    print(f"{'配置':<22} {'问':<4} {'target':<14} {'draft':<12} {'P(Delta)':>8} {'P(Beta)':>8} | 每层: 新码/旧码")
    configs = [
        ("short_n1",  1, False), ("short_n2", 2, False), ("short_n4", 4, False),
        ("long_n1",   1, True),  ("long_n4",  4, True),
    ]
    for name, n_new, long_ctx in configs:
        prompt = build_prompt(n_new, long_ctx, tokenizer)
        for qname, comp, expected in [("new", NEW_COMP, NEW), ("old", OLD_COMP, OLD)]:
            r = run_case(verifier, tokenizer, draft, vocab, prompt, comp, expected, device)
            t_ok = "✓" if expected in r["target_argmax"] else "✗"
            d_ok = "✓" if expected in r["draft_argmax"] else "✗"
            na = " ".join(f"{x*100:.0f}" for x in r["new_attn"])
            oa = " ".join(f"{x*100:.0f}" for x in r["old_attn"])
            print(f"{name:22s} {qname:4s} {r['target_argmax']!r:14s} {r['draft_argmax']!r:12s} "
                  f"{r['draft_p_new'] if r['draft_p_new'] is None else round(r['draft_p_new'],3):>8} "
                  f"{r['draft_p_old'] if r['draft_p_old'] is None else round(r['draft_p_old'],3):>8} | "
                  f"new[{na}] old[{oa}]")
        print()


if __name__ == "__main__":
    main()
