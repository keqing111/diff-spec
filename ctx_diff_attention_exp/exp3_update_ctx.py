#!/usr/bin/env python3
"""实验3(ctx): 更新追踪 — 旧码→更新→新码。单token 码。
旧句: "The access code for Velora Station was {OLD}."
更新句(变体): U0 "After the security update, the access code for Velora Station was changed to {NEW}."
               U1 "After the security update, the access code for Velora Station is {NEW}."
问 NEW: 补全 "The access code for Velora Station is"
问 OLD: 补全 "The access code for Velora Station was"
每个 (变体,问法,seed) 记 target/draft 是否答对 + P + 答案槽对新码/旧码 span 的注意力。
用法: ASCEND_RT_VISIBLE_DEVICES=12 python3 exp3_update_ctx.py -o results_exp3
"""
import argparse
import json
import random
from pathlib import Path

import torch
import sys
sys.path.insert(0, "/home/y50063564/processed_data/archive_dspark_20260904/eval/attention_experiment")
import draft_exp_common as dec
from exp1_length_position import make_filler_pool
import ctx_harness as ch

CAND = ["Delta", "Beta", "Gamma", "Omega", "Kappa", "Theta", "Zulu", "Vega",
        "Echo", "Nova", "Tango", "Oscar", "Juliet", "Foxtrot", "Lima", "Milo"]
NSEEDS = 16
TARGET_LEN = 220


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default="results_exp3")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    device = "npu:0"
    verifier, tokenizer, draft, vocab = ch.load(device)

    def single(word):
        ids = tokenizer.encode(" " + word)
        return len(ids) == 1
    ok = [w for w in CAND if single(w)]
    assert len(ok) >= 2, f"单token码不足: {ok}"
    OLD, NEW = ok[0], ok[1]
    old_id = tokenizer.encode(" " + OLD)[-1]; new_id = tokenizer.encode(" " + NEW)[-1]
    old_did = dec.first_subword_draft_id(tokenizer, vocab, OLD)
    new_did = dec.first_subword_draft_id(tokenizer, vocab, NEW)
    print(f"OLD={OLD} NEW={NEW} | qwen_ids=({old_id},{new_id}) draft_ids=({old_did},{new_did})")

    UPDATES = {
        "was_changed": f"After the security update, the access code for Velora Station was changed to {NEW}.",
        "is_now":      f"After the security update, the access code for Velora Station is {NEW}.",
    }
    COMPL = {"NEW": "The access code for Velora Station is", "OLD": "The access code for Velora Station was"}
    EXPECT = {"NEW": NEW, "OLD": OLD}

    rows = []
    for uv, upd in UPDATES.items():
        for qname, comp in COMPL.items():
            ans = EXPECT[qname]
            exp_id = new_id if qname == "NEW" else old_id
            exp_did = new_did if qname == "NEW" else old_did
            for seed in range(NSEEDS):
                rng = random.Random(seed * 13 + len(uv) + (0 if qname == "NEW" else 7))
                filler = make_filler_pool(seed * 3 + (1 if qname == "NEW" else 2))
                fi = 0
                old_s = f"The access code for Velora Station was {OLD}."
                # 布局: 旧码句 ~15%, 更新句 ~ 随机中段
                parts = []
                # 少量前置 filler, 旧码句, 1-2 filler, 更新句紧贴, 然后补全
                for _ in range(rng.randint(2, 3)):
                    parts.append(filler[fi % len(filler)]); fi += 1
                parts.append(old_s)
                for _ in range(rng.randint(1, 2)):
                    parts.append(filler[fi % len(filler)]); fi += 1
                parts.append(upd)
                comps = " " + comp
                def tokn(s): return len(tokenizer.encode(s if s.startswith(" ") else " " + s))
                while tokn(" ".join(parts) + comps) < TARGET_LEN - 8:
                    parts.insert(0, filler[fi % len(filler)]); fi += 1
                prompt = " ".join(parts) + comps

                ids, aux, vlast, lm, doc, toks, T = dec.prepare_inputs(verifier, tokenizer, prompt, device)
                with torch.no_grad():
                    vl = verifier(ids).logits[0, T - 1]
                target_ok = bool(int(vl.argmax()) == exp_id)
                logits, dbg = ch.run_one(draft, aux, ids, lm, vlast, doc)
                qlen = logits.shape[1]; row = ch.answer_row(qlen, draft.block_size)
                pr = torch.softmax(logits[0, row], dim=-1)
                draft_ok = bool(int(pr.argmax()) == exp_did)
                p_exp = float(pr[exp_did]) if exp_did is not None else None
                p_other = float(pr[old_did] if qname == "NEW" else new_did) if (old_did if qname=="NEW" else new_did) is not None else None
                # span 注意力
                sp_new = dec.find_span(toks, NEW); sp_old = dec.find_span(toks, OLD)
                def attn_on(sp):
                    if not sp: return None
                    tot = 0.0
                    for li in range(5):
                        d = dbg[li]
                        a = d.get("attn")
                        if a is not None:
                            tot += float(a[:, row, sp].sum())
                        else:
                            b0 = d["b0"]; b1 = d["b1"]
                            tot += float((b0[:, row, sp].sum() + b1[:, row, sp].sum()))
                    return tot
                rows.append({"variant": uv, "query": qname, "seed": seed, "T": int(T),
                             "target_ok": target_ok, "draft_ok": draft_ok,
                             "draft_p_exp": p_exp, "draft_p_other": p_other,
                             "attn_new": attn_on(sp_new), "attn_old": attn_on(sp_old),
                             "span_new": sp_new, "span_old": sp_old})
    def to_py(o):
        if torch.is_tensor(o):
            return float(o)
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                pass
        raise TypeError(f"not serializable: {type(o)}")
    (out / "exp3_samples.json").write_text(json.dumps(rows, indent=2, default=to_py))

    summ = {}
    for uv in UPDATES:
        for qname in ("NEW", "OLD"):
            sub = [r for r in rows if r["variant"] == uv and r["query"] == qname]
            cells = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}
            for r in sub:
                k = ("T" if r["target_ok"] else "F") + ("T" if r["draft_ok"] else "F")
                cells[k] += 1
            tc = [r for r in sub if r["target_ok"]]
            summ[f"{uv}/{qname}"] = {
                "n": len(sub), "cells": cells,
                "p_exp|tc": (sum(r["draft_p_exp"] for r in tc) / len(tc)) if tc else None,
                "attn_new-mean": (sum(r["attn_new"] or 0 for r in sub) / len(sub)),
                "attn_old-mean": (sum(r["attn_old"] or 0 for r in sub) / len(sub)),
                "n_tgtX_draftOK": cells["FT"]}
    (out / "exp3_summary.json").write_text(json.dumps(summ, indent=2))
    for k, s in summ.items():
        print(f"{k:16s} n={s['n']:2d} cells={s['cells']} p_exp|tgtOK={('-' if s['p_exp|tc'] is None else round(s['p_exp|tc'],3))} "
              f"attnNew={s['attn_new-mean']:.2f} attnOld={s['attn_old-mean']:.2f} tgtX/draftOK={s['n_tgtX_draftOK']}")
    print(f"\n已存 {out}/exp3_summary.json & exp3_samples.json")


if __name__ == "__main__":
    main()
