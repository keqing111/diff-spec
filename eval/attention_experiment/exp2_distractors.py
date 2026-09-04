#!/usr/bin/env python3
"""实验2: semantic distractor 递增 — 固定长度512, distractor 数量 0/1/2/4/8/16/32 递增。

追踪: 答案槽位每层对 关键token(KeQing) 与 distractor code token 的注意力;
看随 distractor 增加, 关键token注意力是否严重下滑。

用法:
    ASCEND_RT_VISIBLE_DEVICES=13 python3 exp2_distractors.py [-o outdir]
"""
import argparse
import json
from pathlib import Path

import torch

from draft_exp_common import (load_models, build_vocab, run_forward, answer_slot,
                             find_span, prepare_inputs, first_subword_draft_id, FILLER_POOL)

KEY = "The access code for Velora Station is Alpha."
COMPLETION = "The access code for Velora Station is"
ANSWER = "Alpha"
TARGET_LEN = 512
DISTRACTOR_COUNTS = [0, 1, 2, 4, 8, 16, 32]

# distractor 池(与关键句句型相似, 站名/code 不同)
DISTRACTORS = [
    "The access code for Melora Station is QX-124.",
    "The maintenance code used by Velora Station is liyue.",
    "The access code for Aurora Station is ZY-889.",
    "The backup code for Celora Station is PL-556.",
    "The entry code for Dorava Station is NM-310.",
    "The relay code for Felora Station is OR-777.",
    "The access code for Helora Station is TS-245.",
    "The protocol code for Kalora Station is WQ-913.",
    "The access code for Silara Station is JK-068.",
    "The recovery code for Velora Station is BD-402.",
    "The access code for Umora Station is GC-553.",
    "The security code for Nalora Station is HW-871.",
    "The access code for Ivelora Station is RT-336.",
    "The dispatch code for Velora Station is FP-220.",
    "The access code for Xelora Station is VN-665.",
    "The calibration code for Durala Station is AZ-194.",
    "The access code for Ovelora Station is KJ-887.",
    "The diagnostic code for Velora Station is TR-508.",
    "The access code for Pilara Station is MB-732.",
    "The routing code for Falora Station is XQ-419.",
    "The access code for Zorala Station is CW-265.",
    "The override code for Velora Station is GT-144.",
    "The access code for Belora Station is EH-693.",
    "The activation code for Gilara Station is SD-358.",
    "The access code for Jorava Station is LF-871.",
    "The passphrase code for Velora Station is KP-507.",
    "The access code for Teralo Station is UY-224.",
    "The bootstrap code for Laroma Station is ZN-630.",
    "The access code for Korava Station is DA-416.",
    "The sync code for Velora Station is WR-908.",
    "The access code for Samara Station is BK-772.",
    "The handshake code for Volara Station is QM-539.",
]


def build_prompt(n_dist: int, tokenizer) -> str:
    """关键句固定在 ~50% 位置, distractor 均匀分布在前半/后半, 避免位置混淆。"""
    comp = " " + COMPLETION
    half_target = TARGET_LEN // 2 - len(tokenizer.encode(KEY)) // 2
    dists = DISTRACTORS[:n_dist]
    d_half = n_dist // 2
    before_dists, after_dists = dists[:d_half], dists[d_half:]

    # 前半: 填充 + 前半 distractor(交错), 到 ~half_target
    before, fi, di = [], 0, 0
    while len(tokenizer.encode(" ".join(before) + " " + KEY)) < half_target:
        if di < len(before_dists) and len(before) % 4 == 0:  # 每4句插1个distractor
            before.append(before_dists[di]); di += 1
        else:
            before.append(FILLER_POOL[fi % len(FILLER_POOL)]); fi += 1
    # 关键句
    head = " ".join(before) + " " + KEY
    # 后半: 剩余 distractor + 填充到 TARGET_LEN
    after = []
    for d in after_dists:
        after.append(d)
    while len(tokenizer.encode(head + " " + " ".join(after) + comp)) < TARGET_LEN:
        after.append(FILLER_POOL[fi % len(FILLER_POOL)]); fi += 1
    return head + " " + " ".join(after) + comp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default=Path(__file__).parent / "exp2_results")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    device = "npu:0"
    verifier, tokenizer, draft = load_models(device)
    vocab = build_vocab(tokenizer)
    ke_id = first_subword_draft_id(tokenizer, vocab, ANSWER)

    rows = []
    for n in DISTRACTOR_COUNTS:
        prompt = build_prompt(n, tokenizer)
        ids, aux, vlast, lm, doc, toks, T = prepare_inputs(verifier, tokenizer, prompt, device)
        cap = run_forward(draft, aux, ids, lm, vlast, doc)
        slot = answer_slot(draft)
        key_span = find_span(toks, ANSWER)
        # distractor code spans: 每个 distractor 的 code 词(QX-124 等)
        dist_attn_by_layer = [0.0] * 5
        dist_spans = []
        for d in DISTRACTORS[:n]:
            code = d.split(" is ")[-1].rstrip(".").split(" ")[-1]
            sp = find_span(toks, code)
            dist_spans.append(sp)
        per_layer_key = []
        per_layer_dist = []
        for l in range(5):
            a = cap["layer_attn"][l][0, :, slot, :T].mean(0)
            per_layer_key.append(a[key_span].sum().item())
            if dist_spans:
                dsum = sum(a[sp].sum().item() for sp in dist_spans)
                per_layer_dist.append(dsum / len(dist_spans))  # 平均每个 distractor code
            else:
                per_layer_dist.append(0.0)
        p_ke = torch.softmax(cap["logits"][0, slot], dim=-1)[ke_id].item() if ke_id is not None else None
        arg = int(cap["logits"][0, slot].argmax().item())
        arg_tok = tokenizer.decode([vocab[arg]]) if arg < len(vocab) else "?"
        rows.append({"n_distractors": n, "T": T,
                     "key_attn_by_layer": per_layer_key,
                     "dist_attn_by_layer": per_layer_dist,
                     "mean_key_attn": float(sum(per_layer_key)/5),
                     "mean_dist_attn": float(sum(per_layer_dist)/5),
                     "p_first_subword": p_ke, "argmax_token": arg_tok})
        print(f"n={n:2d}  T={T:4d}  P(Ke)={p_ke if p_ke is None else round(p_ke,4)}  预测={arg_tok!r}  "
              f"关键token注意力(层均值)={sum(per_layer_key)/5*100:5.2f}%  每层={[round(x*100,1) for x in per_layer_key]}  "
              f"distractor平均注意力={sum(per_layer_dist)/5*100:5.2f}%")

    (out / "exp2_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\n已存 {out}/exp2_results.json")


if __name__ == "__main__":
    main()
