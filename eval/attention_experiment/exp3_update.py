#!/usr/bin/env python3
"""实验3: 更新推理 — 旧码 KeQing → 更新 → 新码 MX-409。

上下文: "The access code for Velora Station was KeQing ... After the security update,
the access code for Velora Station was changed to MX-409"
问新码: 末尾 "After the security update, the access code for Velora Station is" → 应预测 MX-409
问旧码: 末尾 "Before the security update, the access code for Velora Station was" → 应预测 KeQing
额外: 检查 target(verifier) 在答案位置的 argmax 是否正确。

用法:
    ASCEND_RT_VISIBLE_DEVICES=14 python3 exp3_update.py [-o outdir]
"""
import argparse
import json
from pathlib import Path

import torch

from draft_exp_common import (load_models, build_vocab, run_forward, answer_slot,
                             find_span, prepare_inputs, first_subword_draft_id, FILLER_POOL)

CONTEXT = ("The access code for Velora Station was KeQing. "
           "After the security update, the access code for Velora Station was changed to Delta.")
NEW_COMPLETION = "After the security update, the access code for Velora Station is"
OLD_COMPLETION = "Before the security update, the access code for Velora Station was"
NEW_CODE, OLD_CODE = "Delta", "KeQing"
TARGET_LEN = 512


def build_prompt(completion: str, tokenizer) -> str:
    prompt = CONTEXT
    comp = " " + completion
    fi = 0
    while len(tokenizer.encode(prompt + comp)) < TARGET_LEN:
        prompt += " " + FILLER_POOL[fi % len(FILLER_POOL)]
        fi += 1
    return prompt + comp


def target_check(verifier, tokenizer, prompt, device, expected):
    """verifier 在答案位置(最后 token)的 argmax 是否以 expected 开头。"""
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        logits = verifier(ids).logits[0, -1]
    arg = int(logits.argmax().item())
    arg_tok = tokenizer.decode([arg])
    # expected 的 verifier id
    exp_ids = tokenizer.encode(" " + expected)
    hit = arg == exp_ids[0]
    p = torch.softmax(logits, dim=-1)[exp_ids[0]].item()
    return arg_tok, hit, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default=Path(__file__).parent / "exp3_results")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    device = "npu:0"
    verifier, tokenizer, draft = load_models(device)
    vocab = build_vocab(tokenizer)
    new_id = first_subword_draft_id(tokenizer, vocab, NEW_CODE)
    old_id = first_subword_draft_id(tokenizer, vocab, OLD_CODE)

    rows = []
    for name, completion, expected, exp_id in [
        ("new", NEW_COMPLETION, NEW_CODE, new_id),
        ("old", OLD_COMPLETION, OLD_CODE, old_id),
    ]:
        prompt = build_prompt(completion, tokenizer)
        # target 检查
        arg_tok, hit, p_target = target_check(verifier, tokenizer, prompt, device, expected)
        # draft
        ids, aux, vlast, lm, doc, toks, T = prepare_inputs(verifier, tokenizer, prompt, device)
        cap = run_forward(draft, aux, ids, lm, vlast, doc)
        slot = answer_slot(draft)
        p_draft = torch.softmax(cap["logits"][0, slot], dim=-1)[exp_id].item() if exp_id is not None else None
        darg = int(cap["logits"][0, slot].argmax().item())
        darg_tok = tokenizer.decode([vocab[darg]]) if darg < len(vocab) else "?"
        # 新旧码在上下文的位置
        new_span = find_span(toks, NEW_CODE)
        old_span = find_span(toks, OLD_CODE)
        per_layer = {"new": [], "old": []}
        for l in range(5):
            a = cap["layer_attn"][l][0, :, slot, :T].mean(0)
            per_layer["new"].append(a[new_span].sum().item() if new_span else 0.0)
            per_layer["old"].append(a[old_span].sum().item() if old_span else 0.0)
        rows.append({"ask": name, "T": T, "expected": expected,
                     "target_argmax": arg_tok, "target_correct": hit, "target_p": p_target,
                     "draft_argmax": darg_tok, "draft_p_expected": p_draft,
                     "key_attn_by_layer": per_layer,
                     "new_span": new_span, "old_span": old_span})
        print(f"[{name}] 问{expected}: target预测={arg_tok!r} 正确={hit} (P={p_target:.3f}) | "
              f"draft预测={darg_tok!r} P(期望)={p_draft if p_draft is None else round(p_draft,4)}")
        print(f"    draft 每层注意力: 新码MX-409={[round(x*100,1) for x in per_layer['new']]}  "
              f"旧码KeQing={[round(x*100,1) for x in per_layer['old']]}")

    (out / "exp3_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\n已存 {out}/exp3_results.json")


if __name__ == "__main__":
    main()
