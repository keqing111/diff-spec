#!/usr/bin/env python3
"""把所有实验的 prompt 存成文本文件, 并对实验2跑 target(verifier)正确性检查。

输出:
  exp1_results_alpha/prompts/L{L}_pos{p}.txt
  exp2_results_alpha/prompts/n{n}.txt          (+ target 是否正确)
  exp3_results_v2/prompts/{new,old}.txt
"""
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import exp1_length_position as e1
import exp2_distractors as e2
import exp3_update as e3


def main():
    device = "npu:0"
    # 只需 tokenizer + verifier(不加载 draft)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from draft_exp_common import MODEL
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    verifier = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    vocab_verifier = tokenizer

    # ---- 实验1: 存全部 25 个 prompt ----
    out1 = Path("exp1_results_alpha/prompts"); out1.mkdir(parents=True, exist_ok=True)
    filler = e1.make_filler_pool(42)
    for L in e1.LENGTHS:
        for p in e1.POSITIONS:
            prompt = e1.build_prompt(L, p, tokenizer, filler)
            (out1 / f"L{L}_pos{p}.txt").write_text(prompt)
    print(f"实验1 prompts 已存 {out1} (25 个)")

    # ---- 实验2: 存 prompt + target 检查 ----
    out2 = Path("exp2_results_alpha/prompts"); out2.mkdir(parents=True, exist_ok=True)
    # Alpha 的 verifier id(首子词)
    alpha_vid = tokenizer.encode(" Alpha")[-1]
    rows = []
    for n in e2.DISTRACTOR_COUNTS:
        prompt = e2.build_prompt(n, tokenizer)
        (out2 / f"n{n}.txt").write_text(prompt)
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            logits = verifier(ids).logits[0, -1]
        arg = int(logits.argmax().item())
        arg_tok = tokenizer.decode([arg])
        p_alpha = torch.softmax(logits, dim=-1)[alpha_vid].item()
        rows.append({"n": n, "target_argmax": arg_tok, "target_p_alpha": p_alpha,
                     "target_correct": arg == alpha_vid})
        print(f"  n={n:>2}: target预测={arg_tok!r} P(Alpha)={p_alpha:.4f} 正确={arg==alpha_vid}")
    (out2 / "target_check.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"实验2 prompts 已存 {out2} + target_check.json")

    # ---- 实验3: 存 prompt ----
    out3 = Path("exp3_results_v2/prompts"); out3.mkdir(parents=True, exist_ok=True)
    (out3 / "new.txt").write_text(e3.build_prompt(e3.NEW_COMPLETION, tokenizer))
    (out3 / "old.txt").write_text(e3.build_prompt(e3.OLD_COMPLETION, tokenizer))
    print(f"实验3 prompts 已存 {out3}")


if __name__ == "__main__":
    main()
