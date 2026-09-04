#!/usr/bin/env python3
"""测试 sliding_window 2048 → 4096 对关键 token 注意力的影响。

在长度 2048、关键句位置 10%/25%/50% 下, 对比 2048 vs 4096 窗口。
用法:
    ASCEND_RT_VISIBLE_DEVICES=15 python3 test_sliding_window.py
"""
import torch
import torch_npu  # noqa: F401
from draft_exp_common import load_models, build_vocab, run_forward, answer_slot, find_span, prepare_inputs, first_subword_draft_id
import exp1_length_position as e1

ANSWER = "Alpha"


def run_with_window(verifier, tokenizer, draft, vocab, prompt, device, window):
    # 设置 sliding_window
    for layer in draft.layers:
        layer.self_attn.sliding_window = window
    draft.sliding_window = window
    ids, aux, vlast, lm, doc, toks, T = prepare_inputs(verifier, tokenizer, prompt, device)
    cap = run_forward(draft, aux, ids, lm, vlast, doc)
    slot = answer_slot(draft)
    span = find_span(toks, ANSWER)
    per = []
    for l in range(5):
        a = cap["layer_attn"][l][0, :, slot, :T].mean(0)
        per.append(a[span].sum().item())
    aid = first_subword_draft_id(tokenizer, vocab, ANSWER)
    p = torch.softmax(cap["logits"][0, slot], dim=-1)[aid].item() if aid is not None else None
    return per, p


def main():
    torch.manual_seed(42)
    device = "npu:0"
    verifier, tokenizer, draft = load_models(device)
    vocab = build_vocab(tokenizer)
    filler = e1.make_filler_pool(42)

    print("=== 长度 2048, 关键句位置 10/25/50%, sliding_window 2048 vs 4096 ===")
    print(f"{'pos':>4} {'window':>6} | 每层关键token注意力 [L0..L4]         | P(Alpha)")
    for p in [10, 25, 50]:
        prompt = e1.build_prompt(2048, p, tokenizer, filler)
        for win in [2048, 4096]:
            per, pp = run_with_window(verifier, tokenizer, draft, vocab, prompt, device, win)
            print(f"{p:>4}% {win:>6} | {' '.join(f'{x*100:5.1f}' for x in per)} | {pp:.3f}")


if __name__ == "__main__":
    main()
