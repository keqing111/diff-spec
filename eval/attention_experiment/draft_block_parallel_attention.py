#!/usr/bin/env python3
"""DSpark block 并行生成分析(答案后有真实续文)。

样本: "…Japan is Tokyo, a vibrant metropolis located on the eastern coast…"(答案 Tokyo 后跟 ~20 个真实 token)
anchor 放在答案前一个 "is" 上, block 预测: j=1="Tokyo", j=2..7=真实续词。
用纯 backbone logits(无 Markov)greedy; 分析各 j 对 context 的关注, 及去掉关键 token 的影响。

用法: ASCEND_RT_VISIBLE_DEVICES=13 python3 draft_block_parallel_attention.py
"""
import torch
import torch_npu  # noqa: F401
import speculators.models.dflash.core as core_mod
from draft_exp_common import load_models, build_vocab, prepare_inputs, find_span

SAMPLE = ("The weather in many countries varies a lot, but geography is always an interesting "
          "subject to study. The capital city of Japan is Tokyo, a vibrant metropolis located "
          "on the eastern coast of Honshu island. Tokyo is home to the Imperial Palace and "
          "hosts millions of visitors every year, and its historic temples draw tourists from "
          "around the world.")
KEY = "Tokyo"


def setup_anchor_at(draft, tokenizer, anchor_pos: int):
    """重新 patch select_anchors, 把最后一个 anchor 强制放到 anchor_pos。"""
    orig = core_mod.select_anchors
    def forced(loss_mask, num_anchors, block_size):
        anchors, valid = orig(loss_mask, num_anchors, block_size)
        anchors = anchors.clone(); valid = valid.clone()
        anchors[-1] = anchor_pos
        valid[-1] = True
        return anchors, valid
    core_mod.select_anchors = forced
    draft._restore_anchor = lambda: setattr(core_mod, "select_anchors", orig)


def single_forward(draft, aux, ids, lm, vlast, doc, fc_override):
    from speculators.models.dflash.core import DFlashDraftModel
    cap = {}
    handles = []
    for i in range(5):
        def mk(idx):
            def h(mod, inp, out):
                cap.setdefault("layer_attn", {})[idx] = out[1].detach().float().cpu()
            return h
        handles.append(draft.layers[i].self_attn.register_forward_hook(mk(i)))
    orig_nf = draft.hidden_norm.forward
    def nf(inp):
        o = orig_nf(inp); cap["fc"] = o.detach().float().cpu()
        return fc_override if fc_override is not None else o
    draft.hidden_norm.forward = nf
    bb = DFlashDraftModel._backbone_forward
    def wbb(self, *a, **kw):
        r = bb(self, *a, **kw); cap["backbone_logits"] = r[1].detach().float().cpu()
        return r
    DFlashDraftModel._backbone_forward = wbb
    with torch.no_grad():
        draft(hidden_states=aux, input_ids=ids, loss_mask=lm,
              verifier_last_hidden_states=vlast, document_ids=doc, max_anchors=256)
    DFlashDraftModel._backbone_forward = bb
    draft.hidden_norm.forward = orig_nf
    for h in handles: h.remove()
    return cap


def main():
    torch.manual_seed(42)
    device = "npu:0"
    verifier, tokenizer, draft = load_models(device)
    vocab = build_vocab(tokenizer)

    ids, aux, vlast, lm, doc, toks, T = prepare_inputs(verifier, tokenizer, SAMPLE, device)
    # 答案位置 A; anchor 放 A-1("is"), 使 block 预测 A, A+1, ...
    A = find_span(toks, KEY)[0]
    anchor_pos = A - 1
    print(f"答案 {KEY}@{A} = {toks[A]!r}; anchor 放在 {anchor_pos} = {toks[anchor_pos]!r}; T={T}")
    # 验证 anchor+1..+7 是真实文本
    print(f"block 预测的真实续文: {[toks[A+k].strip() for k in range(8)]}")
    setup_anchor_at(draft, tokenizer, anchor_pos)

    bs = draft.block_size

    def run(fc_override, title):
        cap = single_forward(draft, aux, ids, lm, vlast, doc, fc_override)
        na_idx = draft._last_anchors.shape[0] - 1  # 前向后才可用
        logits = cap["backbone_logits"][0]
        print(f"\n=== {title} ===")
        print(f"{'j':>2} {'greedy(backbone)':>16} | {'真实token':>12} | {'L4 top-3关注context':>24} | 对{KEY}%")
        for j in range(1, bs):
            noise_pos = na_idx * bs + j
            gid = int(logits[noise_pos].argmax().item())
            gtok = tokenizer.decode([vocab[gid]]) if gid < len(vocab) else "?"
            real = toks[anchor_pos + j].strip() if anchor_pos + j < T else "?"
            a = cap["layer_attn"][4][0, :, noise_pos, :T].mean(0)
            key_pct = a[A].item() * 100
            top3 = a.argsort(descending=True)[:3]
            top3s = ", ".join(f"{toks[p].strip()}" for p in top3.tolist())
            ok = "✓" if gtok.strip() == real else " "
            print(f"{j:>2} {gtok!r:>16}{ok} | {real:>12} | {top3s:>24} | {key_pct:>6.1f}%")

    # 基线
    run(None, "基线(关键token存在)")
    # 去掉关键 token
    fc = single_forward(draft, aux, ids, lm, vlast, doc, None)["fc"]
    fc_mod = fc.clone(); fc_mod[0, A] = 0.0
    run(fc_mod.to(device), "去掉关键token(fc清零)")


if __name__ == "__main__":
    main()
