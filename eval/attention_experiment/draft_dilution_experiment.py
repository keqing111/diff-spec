#!/usr/bin/env python3
"""关键 token 稀释实验(用 baseline checkpoint_best 的 DSpark draft)。

样本: 自然段落(答案 Tokyo + 混淆句 Paris/Washington, 塞在不同位置), 末尾 query 预测 Tokyo。
流程:
  1. 完整 forward: 采集 draft 逐层 hidden_states + target 压缩 hidden_states(fc_output)
     + 最后一层注意力; 强制末尾锚点让 draft 预测答案; baseline P(Tokyo)。
  2. top-8 高注意力 context token, 分别做 3 种替换(归零/序列均值/norm相近), 每次一个,
     重跑 forward, 记 P(Tokyo) 变化。

关键修复:
  - t2d/d2t 是"布尔掩码 + 偏移量", 需用 token_freq.pt 重建真实映射。
  - select_anchors 排除末尾 block_size 位, 需强制在 T-1 加锚点才能预测答案。

用法:
    ASCEND_RT_VISIBLE_DEVICES=15 python3 draft_dilution_experiment.py [-o outdir]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

MODEL = "/home/y50063564/Qwen3-4B"
CKPT = "/home/y50063564/checkpoint_best"
TOKEN_FREQ = "/home/y50063564/data/open_perfectblend_qwen3_4b_700k/token_freq.pt"
TARGET_LAYERS = [1, 9, 17, 25, 33]
DRAFT_VOCAB = 32000
ANSWER_WORD = "Tokyo"
SEED = 42

SAMPLE = (
    "The capital city of Japan is Tokyo. "
    "The capital city of France is Paris, a city famous for its art and cuisine. "
    "The Japanese capital is the seat of the national government and home to the "
    "Imperial Palace, and it sits on the eastern coast of Honshu, the largest of "
    "Japan's islands. "
    "The capital of the United States is Washington D.C., where the White House and "
    "the Capitol building are located on the Potomac River. "
    "Paris lies on the Seine in northern France, while Washington D.C. is far to the "
    "west, near the Atlantic coast of the United States. "
    "The capital city of Japan is"
)


def build_vocab_mapping(tokenizer) -> tuple[list[int], int]:
    """重建 draft 词表(与训练一致): 返回 verifier_id 列表(索引=draft_id) 和 Tokyo draft id。"""
    tf = torch.load(TOKEN_FREQ, map_location="cpu", weights_only=True)
    sorted_tokens = sorted(tf, key=lambda tid: (-tf[tid], tid))[:DRAFT_VOCAB]
    sorted_tokens.sort()
    vid = tokenizer.encode(f" {ANSWER_WORD}")[-1]
    tokyo_id = sorted_tokens.index(vid)
    return sorted_tokens, tokyo_id


def load_models(device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    verifier = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    verifier.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    from speculators.model import SpeculatorModel
    from speculators.models.attention import create_float_mask
    import speculators.models.dflash.core as core_mod

    draft = SpeculatorModel.from_pretrained(CKPT, trust_remote_code=True, verifier=MODEL)
    draft.to(device).eval()
    draft._create_mask_fn = create_float_mask
    for layer in draft.layers:
        layer.self_attn.config._attn_implementation = "eager"

    # 强制末尾锚点: patch select_anchors
    orig_select = core_mod.select_anchors
    def forced_select(loss_mask, num_anchors, block_size):
        anchors, valid = orig_select(loss_mask, num_anchors, block_size)
        anchors = anchors.clone(); valid = valid.clone()
        anchors[-1] = loss_mask.shape[1] - 1
        valid[-1] = True
        return anchors, valid
    core_mod.select_anchors = forced_select
    draft._restore_select = lambda: setattr(core_mod, "select_anchors", orig_select)

    # 抓 anchor_positions
    orig_build = draft._build_attention_mask
    def wrapper(loss_mask, max_anchors, document_ids, device):
        out = orig_build(loss_mask, max_anchors, document_ids, device)
        draft._last_anchors = out[2].cpu()
        return out
    draft._build_attention_mask = wrapper
    return verifier, tokenizer, draft


def run_forward(draft, aux, ids, loss_mask, vlast, doc_ids, fc_override=None):
    from speculators.models.dflash.core import DFlashDraftModel
    captured = {}
    handles = []
    for i in range(5):
        def mk(idx):
            def h(mod, inp, out):
                captured[f"layer_{idx}"] = out[0].detach().float().cpu()
            return h
        handles.append(draft.layers[i].register_forward_hook(mk(i)))
    def attn_hook(mod, inp, out):
        captured["last_attn"] = out[1].detach().float().cpu()
    handles.append(draft.layers[4].self_attn.register_forward_hook(attn_hook))

    # 直接替换 hidden_norm.forward(保证 override 生效)
    orig_nf = draft.hidden_norm.forward
    def new_nf(inp):
        out = orig_nf(inp)
        captured["fc_output"] = out.detach().cpu()  # 保持原 dtype
        return fc_override if fc_override is not None else out
    draft.hidden_norm.forward = new_nf

    bb = DFlashDraftModel._backbone_forward
    def wbb(self, *a, **kw):
        r = bb(self, *a, **kw)
        captured["logits"] = r[1].detach().float().cpu()
        return r
    DFlashDraftModel._backbone_forward = wbb

    with torch.no_grad():
        draft(hidden_states=aux, input_ids=ids, loss_mask=loss_mask,
              verifier_last_hidden_states=vlast, document_ids=doc_ids,
              max_anchors=256)
    DFlashDraftModel._backbone_forward = bb
    draft.hidden_norm.forward = orig_nf
    for h in handles:
        h.remove()
    return captured


def p_tokyo(logits, noise_pos, tokyo_id):
    return torch.softmax(logits[0, noise_pos], dim=-1)[tokyo_id].item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default=Path(__file__).parent / "dilution_results")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    device = "npu:0"
    verifier, tokenizer, draft = load_models(device)
    vocab, tokyo_id = build_vocab_mapping(tokenizer)
    print(f"Tokyo draft id = {tokyo_id} (verifier id = {vocab[tokyo_id]})")

    ids = tokenizer(SAMPLE, return_tensors="pt")["input_ids"].to(device)
    T = ids.shape[1]
    print(f"样本 T={T}")
    toks = [tokenizer.decode([i]) for i in ids[0].tolist()]
    aw_pos = [i for i, t in enumerate(toks) if ANSWER_WORD.lower() in t.replace(" ", "").lower()]

    with torch.no_grad():
        hs = verifier(ids, output_hidden_states=True).hidden_states
    aux = torch.cat([hs[i].float() for i in TARGET_LAYERS], dim=-1).to(torch.bfloat16)
    vlast = hs[-1]
    loss_mask = torch.ones_like(ids).float()
    doc_ids = torch.zeros_like(ids)

    # 第一次完整 forward
    cap = run_forward(draft, aux, ids, loss_mask, vlast, doc_ids)
    fc = cap["fc_output"]  # [1, T, 2560]
    logits = cap["logits"]  # [1, noise_len, 32000]
    bs = draft.block_size
    ans_slot = (draft._last_anchors.shape[0] - 1) * bs + 1  # 强制末尾锚点块的 j=1
    p0 = p_tokyo(logits, ans_slot, tokyo_id)
    # 该槽位 argmax 是不是 Tokyo
    arg = int(logits[0, ans_slot].argmax().item())
    arg_tok = tokenizer.decode([vocab[arg]]) if arg < len(vocab) else "?"
    print(f"答案槽位 noise={ans_slot}, 预测 {arg_tok!r} (draft_id={arg}), P(Tokyo)={p0:.4f}")

    # top-8 context token: 最后一层注意力对 context 部分聚合
    attn = cap["last_attn"]  # [1, H, Nq, T+N]
    ctx = attn[0, :, :, :T].sum(dim=(0, 1))  # [T] 各 context token 收到注意力
    top8 = ctx.argsort(descending=True)[:8].tolist()
    print(f"top-8 context token: {[(p, toks[p].strip(), round(ctx[p].item(),3)) for p in top8]}")

    # baseline 信息
    rec = {
        "sample": SAMPLE, "T": T, "answer_positions": aw_pos,
        "answer_slot": ans_slot, "tokyo_draft_id": tokyo_id,
        "baseline_p_tokyo": p0, "baseline_argmax_token": arg_tok,
        "top8_tokens": [{"pos": p, "text": toks[p].strip(), "attn": ctx[p].item()} for p in top8],
    }
    (out / "baseline.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    torch.save({"fc_output": fc, "per_layer_hidden": {i: cap[f"layer_{i}"] for i in range(5)},
                "last_attn": attn, "logits": logits, "ids": ids.cpu(), "tokens": toks,
                "answer_slot": ans_slot, "tokyo_id": tokyo_id}, out / "baseline.pt")

    # ===== 替换实验 =====
    fc_base = fc.clone()  # [1, T, 2560]
    # 三种操作
    ops = {
        "zero": lambda fc, i: fc.clone().index_fill(1, torch.tensor([i]), 0.0),
        "mean": lambda fc, i: fc.clone().index_copy(1, torch.tensor([i]), fc.mean(dim=1, keepdim=True)),
        "norm_similar": None,
    }
    norms = fc_base[0].norm(dim=-1)  # [T]
    results = []
    for pos in top8:
        row = {"pos": pos, "text": toks[pos].strip()}
        for op_name, fn in ops.items():
            if op_name == "norm_similar":
                # 挑 norm 最接近的其它 token
                d = (norms - norms[pos]).abs(); d[pos] = float("inf")
                j = int(d.argmin().item())
                mod = fc_base.clone(); mod[0, pos] = fc_base[0, j]
                note = f"<- pos{j} {toks[j].strip()} norm={norms[j].item():.2f}"
            else:
                mod = fn(fc_base, pos)
                note = ""
            p = p_tokyo(run_forward(draft, aux, ids, loss_mask, vlast, doc_ids, fc_override=mod.to(device))["logits"], ans_slot, tokyo_id)
            row[op_name] = p
            row[f"{op_name}_delta"] = p - p0
            row[f"{op_name}_note"] = note
        results.append(row)
        print(f"pos{pos} {toks[pos].strip()!r}: zero→{row['zero']:.4f}(Δ{row['zero_delta']:+.4f}) "
              f"mean→{row['mean']:.4f}(Δ{row['mean_delta']:+.4f}) "
              f"norm→{row['norm_similar']:.4f}(Δ{row['norm_similar_delta']:+.4f})")

    (out / "replacement_results.json").write_text(json.dumps({
        "baseline_p_tokyo": p0, "results": results}, indent=2, ensure_ascii=False))
    print(f"\nbaseline P(Tokyo)={p0:.4f}, 结果存于 {out}/replacement_results.json")


if __name__ == "__main__":
    main()
