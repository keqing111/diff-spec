#!/usr/bin/env python3
"""per-layer 分析: draft 5 层中, 答案槽位(query)对各 context token 的注意力占比。

重点: 每层答案槽位给 "Tokyo"(关键 token) 的注意力比例, 看是否随层被"稀释";
同时给 BOS / 其它 context / noise 的占比, 与 target 压缩 hidden_states 中 Tokyo 的区分度对比。

用法:
    ASCEND_RT_VISIBLE_DEVICES=15 python3 draft_per_layer_analysis.py
"""
import torch
import torch_npu  # noqa: F401

from draft_dilution_experiment import load_models, build_vocab_mapping, TARGET_LAYERS, MODEL, SAMPLE


def main() -> None:
    torch.manual_seed(42)
    device = "npu:0"
    verifier, tokenizer, draft = load_models(device)
    vocab, tokyo_id = build_vocab_mapping(tokenizer)

    ids = tokenizer(SAMPLE, return_tensors="pt")["input_ids"].to(device)
    T = ids.shape[1]
    toks = [tokenizer.decode([i]) for i in ids[0].tolist()]
    aw_pos = [i for i, t in enumerate(toks) if "tokyo" in t.replace(" ", "").lower()][0]
    with torch.no_grad():
        hs = verifier(ids, output_hidden_states=True).hidden_states
    aux = torch.cat([hs[i].float() for i in TARGET_LAYERS], dim=-1).to(torch.bfloat16)
    vlast = hs[-1]
    loss_mask = torch.ones_like(ids).float()
    doc = torch.zeros_like(ids)

    # 抓 5 层注意力 + fc_output + logits
    from speculators.models.dflash.core import DFlashDraftModel
    cap = {}
    handles = []
    for i in range(5):
        def mk(idx):
            def h(mod, inp, out):
                cap[f"attn_{idx}"] = out[1].detach().float().cpu()
            return h
        handles.append(draft.layers[i].self_attn.register_forward_hook(mk(i)))
    orig_nf = draft.hidden_norm.forward
    def nf(inp):
        o = orig_nf(inp)
        cap["fc"] = o.detach().float().cpu()
        return o
    draft.hidden_norm.forward = nf
    bb = DFlashDraftModel._backbone_forward
    def wbb(self, *a, **kw):
        r = bb(self, *a, **kw)
        cap["logits"] = r[1].detach().float().cpu()
        return r
    DFlashDraftModel._backbone_forward = wbb
    with torch.no_grad():
        draft(hidden_states=aux, input_ids=ids, loss_mask=loss_mask,
              verifier_last_hidden_states=vlast, document_ids=doc, max_anchors=256)
    DFlashDraftModel._backbone_forward = bb
    draft.hidden_norm.forward = orig_nf
    for h in handles:
        h.remove()

    bs = draft.block_size
    na = draft._last_anchors.shape[0]
    ans_slot = (na - 1) * bs + 1  # 答案槽位
    print(f"答案槽位 noise={ans_slot}, 预测 {tokenizer.decode([vocab[int(cap['logits'][0,ans_slot].argmax())]])!r}")

    # target 压缩 hidden_states 里 Tokyo 的区分度(与序列均值的余弦)
    fc = cap["fc"][0]  # [T, 2560]
    fc_n = fc / fc.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tokyo_v = fc_n[aw_pos]
    cos_to_mean = (tokyo_v * fc_n.mean(dim=0)).sum().item()
    cos_to_tokyo = (tokyo_v * fc_n).sum(dim=-1)  # [T] 每个 token 与 Tokyo 的余弦
    print(f"\ntarget(fc_output): Tokyo token 与序列均值余弦={cos_to_mean:.3f}")
    print(f"  Tokyo 与其它 token 的最大余弦(top3): "
          + ", ".join(f"{toks[p].strip()}={cos_to_tokyo[p].item():.2f}"
                      for p in cos_to_tokyo.argsort(descending=True)[:3]))

    # 每层: 答案槽位注意力分布
    print(f"\n{'层':>3} {'→Tokyo%':>8} {'→BOS%':>6} {'→其它ctx%':>9} {'→noise%':>8} {'top3被关注token'}")
    for L in range(5):
        a = cap[f"attn_{L}"][0, :, ans_slot, :]  # [H, T+N]
        am = a.mean(dim=0)  # [T+N]
        tokyo = am[aw_pos].item()
        bos = am[0].item()
        other_ctx = am[:T].sum().item() - tokyo - bos
        noise = am[T:].sum().item()
        # 被关注最多的 context token(top3)
        ctx_ord = am[:T].argsort(descending=True)[:3]
        top3 = ", ".join(f"{toks[p].strip()}={am[p].item():.3f}" for p in ctx_ord.tolist())
        print(f"{L:>3} {tokyo*100:>7.1f}% {bos*100:>5.1f}% {other_ctx*100:>8.1f}% {noise*100:>7.1f}%  {top3}")


if __name__ == "__main__":
    main()
