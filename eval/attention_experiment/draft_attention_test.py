#!/usr/bin/env python3
"""用 baseline checkpoint_best 的 DSpark draft 在样本上推理, 抓取 draft 5 层 GQA 注意力。

先测单样本验证流程; 之后跑全部样本。
"""
import torch
import torch_npu  # noqa: F401

MODEL = "/home/y50063564/Qwen3-4B"
CKPT = "/home/y50063564/checkpoint_best"
TARGET_LAYERS = [1, 9, 17, 25, 33]


def main() -> None:
    device = "npu:0"
    print(f"设备: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("加载 verifier (Qwen3-4B)...")
    verifier = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    verifier.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    print("加载 draft (checkpoint_best)...")
    from speculators.model import SpeculatorModel

    draft = SpeculatorModel.from_pretrained(CKPT, trust_remote_code=True, verifier=MODEL)
    draft.to(device).eval()
    # 强制 eager 注意力 + 稠密 float 掩码, 以拿到注意力权重(BlockMask 与 eager 不兼容)
    from speculators.models.attention import create_float_mask
    draft._create_mask_fn = create_float_mask
    for layer in draft.layers:
        layer.self_attn.config._attn_implementation = "eager"
    print(f"draft 层数: {len(draft.layers)}")

    # 样本 s01
    prompt = ("The capital city of Japan is Tokyo. "
              "The weather today is pleasant and the sky is clear across the whole region. "
              "Many people enjoy taking a quiet walk in the park during the afternoon. "
              "The capital city of Japan is")
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    T = ids.shape[1]
    print(f"样本 T={T}")

    # verifier hidden states
    with torch.no_grad():
        out = verifier(ids, output_hidden_states=True)
    hs = out.hidden_states
    print(f"hidden_states 数: {len(hs)} (期望 37 = embed + 36 层)")
    aux = torch.cat([hs[i].float() for i in TARGET_LAYERS], dim=-1).to(torch.bfloat16)
    vlast = hs[-1]
    print(f"aux shape: {tuple(aux.shape)}, vlast: {tuple(vlast.shape)}")

    # draft 输入
    loss_mask = torch.ones_like(ids).float()
    doc_ids = torch.zeros_like(ids)

    # hook 抓 5 层注意力
    captured = {}
    handles = []
    for i, layer in enumerate(draft.layers):
        def mk_hook(idx):
            def h(mod, inp, out):
                captured[idx] = out[1].detach().float().cpu()  # attn_weights
            return h
        handles.append(layer.self_attn.register_forward_hook(mk_hook(i)))

    print("draft 前向...")
    with torch.no_grad():
        draft_tokens, loss, metrics = draft(
            hidden_states=aux, input_ids=ids, loss_mask=loss_mask,
            verifier_last_hidden_states=vlast, document_ids=doc_ids,
            max_anchors=256,
        )
    for h in handles:
        h.remove()

    for i in range(len(draft.layers)):
        a = captured.get(i)
        print(f"L{i}: attn {tuple(a.shape)}" if a is not None else f"L{i}: 未捕获")
    print("done")


if __name__ == "__main__":
    main()
