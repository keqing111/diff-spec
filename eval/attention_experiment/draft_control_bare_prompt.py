#!/usr/bin/env python3
"""控制实验: 只发裸 prompt(无上下文答案), 测 draft 的先验 P(Tokyo)。

对比: 完整样本(含上下文 Tokyo)的 P(Tokyo) 与 裸 prompt 的 P(Tokyo)。
若裸 prompt 也高 → Tokyo 在 draft 知识库里; 若低 → 信息来自上下文。
用法:
    ASCEND_RT_VISIBLE_DEVICES=15 python3 draft_control_bare_prompt.py
"""
import torch
import torch_npu  # noqa: F401

from draft_dilution_experiment import load_models, build_vocab_mapping, run_forward, p_tokyo

MODEL = "/home/y50063564/Qwen3-4B"
TARGET_LAYERS = [1, 9, 17, 25, 33]

PROMPTS = {
    "bare_capital_japan": "The capital city of Japan is",
    "bare_capital": "The capital city is",
    "bare_japan": "Japan is a country in Asia. The capital city of Japan is",
    "bare_smalltalk": "I am writing a short sentence about world capitals now. The capital city of Japan is",
}

FULL_SAMPLE = ("The capital city of Japan is Tokyo. The capital city of France is Paris, a city famous for its art and cuisine. "
"The Japanese capital is the seat of the national government and home to the Imperial Palace, and it sits on the eastern coast of Honshu, the largest of Japan's islands. "
"The capital of the United States is Washington D.C., where the White House and the Capitol building are located on the Potomac River. "
"Paris lies on the Seine in northern France, while Washington D.C. is far to the west, near the Atlantic coast of the United States. "
"The capital city of Japan is")


def measure(verifier, tokenizer, draft, vocab, tokyo_id, prompt, device) -> tuple[float, str]:
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    T = ids.shape[1]
    with torch.no_grad():
        hs = verifier(ids, output_hidden_states=True).hidden_states
    aux = torch.cat([hs[i].float() for i in TARGET_LAYERS], dim=-1).to(torch.bfloat16)
    vlast = hs[-1]
    loss_mask = torch.ones_like(ids).float()
    doc = torch.zeros_like(ids)
    cap = run_forward(draft, aux, ids, loss_mask, vlast, doc)
    bs = draft.block_size
    na = draft._last_anchors.shape[0]
    ans_slot = (na - 1) * bs + 1
    p = p_tokyo(cap["logits"], ans_slot, tokyo_id)
    arg = int(cap["logits"][0, ans_slot].argmax().item())
    arg_tok = tokenizer.decode([vocab[arg]]) if arg < len(vocab) else "?"
    return p, arg_tok, T


def main() -> None:
    torch.manual_seed(42)
    device = "npu:0"
    verifier, tokenizer, draft = load_models(device)
    vocab, tokyo_id = build_vocab_mapping(tokenizer)

    print("=== 完整样本(含上下文答案 + 混淆句) ===")
    p_full, arg_full, T = measure(verifier, tokenizer, draft, vocab, tokyo_id, FULL_SAMPLE, device)
    print(f"  P(Tokyo) = {p_full:.4f}  (预测 {arg_full!r}, T={T})")

    print("\n=== 裸 prompt(无上下文答案) ===")
    for name, prompt in PROMPTS.items():
        p, arg, T = measure(verifier, tokenizer, draft, vocab, tokyo_id, prompt, device)
        print(f"  {name:16s} T={T:3d}  P(Tokyo) = {p:.4f}  (预测 {arg!r})")


if __name__ == "__main__":
    main()
