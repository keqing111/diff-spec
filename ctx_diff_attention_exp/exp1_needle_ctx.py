#!/usr/bin/env python3
"""实验1(ctx): Velora/Alpha needle, 长度[128,512,2048] x 位置[10,50,90]% 代表格。
纵向逐层关键token占比(L0 diff: b0+b1 与 b0-b1; L1-4 GQA head-mean) + draft P。
横向: 代表格下 diff 层每 pair b0/b1/diff; 答案槽 j=1..8 扫描。
用法: ASCEND_RT_VISIBLE_DEVICES=11 python3 exp1_needle_ctx.py [-o outdir]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/y50063564/processed_data/archive_dspark_20260904/eval/attention_experiment")
import draft_exp_common as dec          # noqa: E402
from exp1_length_position import make_filler_pool, build_prompt   # noqa: E402
import ctx_harness as ch                # noqa: E402

ANSWER = "Alpha"
LENGTHS = [128, 512, 2048]
POSITIONS = [10, 50, 90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default="results_exp1")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    device = "npu:0"
    verifier, tokenizer, draft, vocab = ch.load(device)
    ans_first = dec.first_subword_draft_id(tokenizer, vocab, ANSWER)
    print(f"答案 {ANSWER!r} 首子词 draft_id = {ans_first}")

    filler = make_filler_pool(42)
    vertical, horizontals, jscans = [], {}, {}
    for L in LENGTHS:
        for pct in POSITIONS:
            prompt = build_prompt(L, pct, tokenizer, filler)
            ids, aux, vlast, lm, doc, toks, T = dec.prepare_inputs(verifier, tokenizer, prompt, device)
            logits, dbg = ch.run_one(draft, aux, ids, lm, vlast, doc)
            q_len = logits.shape[1]
            row = ch.answer_row(q_len, draft.block_size)
            span = dec.find_span(toks, ANSWER)
            if not span:
                print(f"  L={L} pos={pct} 未找到 span, 跳过"); continue
            lay = {li: ch.layer_key_share(dbg, li, row, span, T) for li in range(5)}
            pr = torch.softmax(logits[0, row], dim=-1)
            p = float(pr[ans_first]) if ans_first is not None else None
            arg = int(pr.argmax()); arg_tok = tokenizer.decode([vocab[arg]]) if arg < len(vocab) else "?"
            rec = {"length": L, "position_pct": pct, "T": int(T), "row": int(row),
                   "span": span, "p_first_subword": p, "argmax_token": arg_tok,
                   "layer": {str(li): lay[li] for li in range(5)}}
            vertical.append(rec)
            print(f"L={L:5d} pos={pct:3d}% T={T:5d} row={row}: "
                  f"L0 both={lay[0]['both_raw']*100:.2f}% diff={lay[0]['diff_raw']*100:+.2f}% "
                  f"L4={lay[4]['both_raw']*100:.2f}%  P(Alpha)={p if p is None else round(p,3)}  pred={arg_tok!r}")
            # 横向 + j 扫描: 只挑代表格
            if (L, pct) in [(512, 50), (2048, 90)]:
                horizontals[f"L{L}_p{pct}"] = ch.per_pair_key_attn(dbg, row, span, T)
                jscans[f"L{L}_p{pct}"] = ch.jscan_key_share(dbg, q_len, draft.block_size, span, T)

    def to_py(o):
        if torch.is_tensor(o):
            return float(o)
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                pass
        raise TypeError(f"not serializable: {type(o)}")
    (out / "exp1_vertical.json").write_text(json.dumps(vertical, indent=2, default=to_py))
    (out / "exp1_horizontal.json").write_text(json.dumps(horizontals, indent=2, default=to_py))
    (out / "exp1_jscan.json").write_text(json.dumps(jscans, indent=2, default=to_py))
    print(f"\n已存 {out}/exp1_vertical.json 等")


if __name__ == "__main__":
    main()
