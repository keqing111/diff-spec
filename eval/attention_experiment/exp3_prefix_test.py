#!/usr/bin/env python3
"""验证假设: draft 靠前缀匹配选码。新码句前缀与补全一致 vs 不一致。

补全: "After the security update, the access code for Velora Station is"
旧码: Beta。新码: Delta。
配置: aligned(前缀一致) / misaligned(was changed to) / 组合。

用法:
    ASCEND_RT_VISIBLE_DEVICES=15 python3 exp3_prefix_test.py
"""
import torch
import torch_npu  # noqa: F401
from draft_exp_common import (load_models, build_vocab, run_forward, answer_slot,
                             find_span, prepare_inputs, first_subword_draft_id)

OLD, NEW = "Beta", "Delta"
COMP = "After the security update, the access code for Velora Station is"
ALIGNED = "After the security update, the access code for Velora Station is Delta."
MISALIGNED = "After the security update, the access code for Velora Station was changed to Delta."
OLD_STMT = "The access code for Velora Station was Beta."


def run(verifier, tokenizer, draft, vocab, prompt, device):
    p_full = prompt + " " + COMP
    ids, aux, vlast, lm, doc, toks, T = prepare_inputs(verifier, tokenizer, p_full, device)
    cap = run_forward(draft, aux, ids, lm, vlast, doc)
    slot = answer_slot(draft)
    probs = torch.softmax(cap["logits"][0, slot], dim=-1)
    darg = int(probs.argmax().item())
    d_tok = tokenizer.decode([vocab[darg]]) if darg < len(vocab) else "?"
    ndid = first_subword_draft_id(tokenizer, vocab, NEW)
    obid = first_subword_draft_id(tokenizer, vocab, OLD)
    p_new = probs[ndid].item() if ndid is not None else None
    p_old = probs[obid].item() if obid is not None else None
    nsp = find_span(toks, NEW); osp = find_span(toks, OLD)
    na, oa = [], []
    for l in range(5):
        a = cap["layer_attn"][l][0, :, slot, :T].mean(0)
        na.append(a[nsp].sum().item() if nsp else 0.0)
        oa.append(a[osp].sum().item() if osp else 0.0)
    return d_tok, p_new, p_old, na, oa


def main():
    torch.manual_seed(42)
    device = "npu:0"
    verifier, tokenizer, draft = load_models(device)
    vocab = build_vocab(tokenizer)

    configs = [
        ("aligned(前缀一致)",            OLD_STMT + " " + ALIGNED),
        ("misaligned(was changed)",      OLD_STMT + " " + MISALIGNED),
        ("aligned+aligned",              OLD_STMT + " " + ALIGNED + " " + ALIGNED),
        ("aligned+misaligned",           OLD_STMT + " " + ALIGNED + " " + MISALIGNED),
        ("misaligned+aligned",           OLD_STMT + " " + MISALIGNED + " " + ALIGNED),
        ("only_aligned(无旧码句)",        ALIGNED),
    ]
    print(f"补全: {COMP!r}")
    print(f"{'配置':<24} {'draft':<10} {'P(Delta)':>8} {'P(Beta)':>8} | 每层注意力 新码/旧码")
    for name, prompt in configs:
        d_tok, p_new, p_old, na, oa = run(verifier, tokenizer, draft, vocab, prompt, device)
        ok = "✓" if d_tok == " Delta" else "✗"
        print(f"{name:24s} {d_tok!r:<10}{ok} {p_new if p_new is None else round(p_new,3):>8} "
              f"{p_old if p_old is None else round(p_old,3):>8} | new[{(' '.join(f'{x*100:.0f}' for x in na))}] old[{(' '.join(f'{x*100:.0f}' for x in oa))}]")


if __name__ == "__main__":
    main()
