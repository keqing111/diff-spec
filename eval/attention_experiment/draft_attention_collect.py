#!/usr/bin/env python3
"""用 baseline checkpoint_best 的 DSpark draft 在 10 个样本上推理, 分析其 5 层 GQA 注意力。

Draft 注意力结构: query = noise 位置(max_anchors*block_size), key = [context(T) ; noise(N)]。
本脚本对每个样本每层计算:
  - 头平均 + query平均 后, 注意力分给 context(BOS/答案词/query/其余) vs noise 的占比
  - 活跃 query 比例(注意力非均匀的 query, 用于排除 padding 稀释)

用法:
    ASCEND_RT_VISIBLE_DEVICES=15 python3 draft_attention_collect.py [-o outdir]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

MODEL = "/home/y50063564/Qwen3-4B"
CKPT = "/home/y50063564/checkpoint_best"
TARGET_LAYERS = [1, 9, 17, 25, 33]
RAW = Path(__file__).parent / "attention_raw"


def load_models(device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    verifier = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    verifier.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    from speculators.model import SpeculatorModel
    from speculators.models.attention import create_float_mask

    draft = SpeculatorModel.from_pretrained(CKPT, trust_remote_code=True, verifier=MODEL)
    draft.to(device).eval()
    draft._create_mask_fn = create_float_mask
    for layer in draft.layers:
        layer.self_attn.config._attn_implementation = "eager"
    return verifier, tokenizer, draft


def run_sample(verifier, tokenizer, draft, prompt, device, max_anchors=256):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    T = ids.shape[1]
    with torch.no_grad():
        hs = verifier(ids, output_hidden_states=True).hidden_states
    aux = torch.cat([hs[i].float() for i in TARGET_LAYERS], dim=-1).to(torch.bfloat16)
    vlast = hs[-1]
    loss_mask = torch.ones_like(ids).float()
    doc_ids = torch.zeros_like(ids)

    captured = {}
    handles = []
    for i, layer in enumerate(draft.layers):
        def mk(idx):
            def h(mod, inp, out):
                captured[idx] = out[1].detach().float().cpu()
            return h
        handles.append(layer.self_attn.register_forward_hook(mk(i)))
    with torch.no_grad():
        draft(hidden_states=aux, input_ids=ids, loss_mask=loss_mask,
              verifier_last_hidden_states=vlast, document_ids=doc_ids,
              max_anchors=max_anchors)
    for h in handles:
        h.remove()
    return T, captured


def analyze(attn, T, bos, aw, qpos):
    """attn: [L, heads, Nq, T+N]. 返回每层 {bos, answer, query, other_ctx, noise, active_frac}。"""
    a = attn.float()  # [L, H, Nq, T+N]
    L, H, Nq, K = a.shape
    per_layer = []
    for l in range(L):
        al = a[l]  # [H, Nq, K]
        # 活跃 query: 最大注意力 > 2 * 均匀(即非 padding 的均匀行)
        uniform = 1.0 / K
        maxv = al.max(dim=-1).values  # [H, Nq]
        active = (maxv > uniform * 2.0).float()
        active_frac = active.mean().item()
        # 头平均 → [Nq, K], 再对活跃 query 平均
        am = al.mean(dim=0)  # [Nq, K]
        ctx = am[:, :T]
        noise = am[:, T:].sum(dim=-1)
        bos_v = ctx[:, bos]
        aw_v = ctx[:, aw].sum(dim=-1)
        q_v = ctx[:, qpos]
        other = ctx.sum(dim=-1) - bos_v - aw_v - q_v
        # 对活跃 query 平均(避免 padding 稀释)
        mask = active.mean(dim=0) > 0.5  # [Nq]
        if mask.sum() == 0:
            mask = torch.ones_like(mask, dtype=torch.bool)
        sel = lambda v: v[mask].mean().item()
        per_layer.append({
            "bos": sel(bos_v), "answer": sel(aw_v), "query": sel(q_v),
            "other_ctx": sel(other), "noise": sel(noise), "active_frac": active_frac,
        })
    return per_layer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default=Path(__file__).parent)
    args = ap.parse_args()
    out = Path(args.outdir)

    device = "npu:0"
    verifier, tokenizer, draft = load_models(device)
    samples = sorted(RAW.glob("*.pt"))

    rows = []
    all_share = {f"L{l}": [] for l in range(5)}
    for f in samples:
        r = torch.load(f, weights_only=False)
        prompt = r["prompt_text"]
        aw = r["answer_word_positions"]
        T, captured = run_sample(verifier, tokenizer, draft, prompt, device)
        # 把 5 层注意力叠成 [5, H, Nq, T+N] (去掉 batch 维)
        attn = torch.stack([captured[l].squeeze(0) for l in range(5)])
        bos, qpos = 0, T - 1
        per = analyze(attn, T, bos, aw, qpos)
        rows.append({"id": r["id"], "T": T, "answer_word": r["answer_word"]} | {
            f"L{l}_{k}": per[l][k] for l in range(5) for k in per[l]
        })
        for l in range(5):
            all_share[f"L{l}"].append(per[l])
        print(f"{r['id']}: T={T} " +
              " | ".join(f"L{l} ans={per[l]['answer']*100:.1f}% bos={per[l]['bos']*100:.1f}% "
                         f"noise={per[l]['noise']*100:.1f}% act={per[l]['active_frac']:.2f}" for l in range(5)))

    (out / "draft_attention_summary.csv").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False)
    )

    # 均值
    print("\n=== 10 样本均值(每 draft 层) ===")
    print(f"{'层':>3} {'答案词%':>7} {'BOS%':>6} {'query%':>7} {'其余ctx%':>8} {'noise%':>7} {'活跃query':>8}")
    for l in range(5):
        m = {k: np.mean([p[l][k] for p in all_share[f"L{l}"]]) for k in all_share[f"L{l}"][0]}
        print(f"{l:>3} {m['answer']*100:>7.2f} {m['bos']*100:>6.1f} {m['query']*100:>7.2f} "
              f"{m['other_ctx']*100:>8.2f} {m['noise']*100:>7.1f} {m['active_frac']:>8.2f}")


if __name__ == "__main__":
    main()
