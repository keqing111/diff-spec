#!/usr/bin/env python3
"""实验2(ctx): 多语义相似 distractor。
- 关键句 "The access code for Velora Station is Alpha." ~50% 位置, 长度 ~512;
- distractor 数 0/1/2/4/8/16/32, 每个 n 跑多个 seed (不同填充/顺序)。
- 对每个样本: 记录 target(verifier) 与 draft 是否答对 + P(真码);
  以及 L3+L4 在答案槽注意力把"真关键句 span"排在所有候选句 span 第几 (定位命中)。
- 全部样本存下来 → 汇总 target×draft 2x2、target答对时 P(真码)/定位命中率,
  并单独找 "target错但draft对" 的样本。
用法: ASCEND_RT_VISIBLE_DEVICES=12 python3 exp2_distractors_ctx.py -o results_exp2
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

KEY_SENT = "The access code for Velora Station is Alpha."
COMPLETION = "The access code for Velora Station is"
ANSWER = "Alpha"
LEVELS = [0, 1, 2, 4, 8, 16, 32]
NSEEDS = 12
TARGET_LEN = 512

# (station, code) 池; code 不一定单token (定位按整句 span 比), 只有 ANSWER 需单token。
STATIONS = ["Velora", "Melora", "Xerion", "Cantara", "Yasmin", "Ontar", "Karli",
            "Nestro", "Belora", "Rynor", "Tavril", "Ondel", "Saron", "Kethar",
            "Umbral", "Zorath", "Pellum", "Draxis", "Hollis", "Fendral",
            "Quorin", "Vendra", "Lithra", "Marvox", "Sylvan", "Onera",
            "Corvane", "Estrel", "Mirella", "Voska", "Aurell", "Trebis",
            "Nymora", "Ilande", "Ferrow", "Callun", "Drast", "Veyra"]
CODES = ["Delta", "Beta", "Gamma", "Omega", "Sigma", "Kappa", "Theta", "Zulu",
         "Vega", "Iota", "Echo", "Lima", "Oscar", "Zed", "Nova", "Rex",
         "Tango", "Juliet", "Yankee", "Hotel", "Foxtrot", "India", "Mike",
         "November", "Quebec", "Sierra", "Uniform", "Whiskey", "Xray",
         "Papa", "Charlie", "Bravo", "Alfa", "Kilo", "Lima9", "Que", "ZedX", "Milo"]


def build_one(rng, tokenizer, filler, n):
    """构造长度 ~TARGET_LEN 的 prompt, 返回 (prompt, cands)。
    cands: list of (label, needle_text); label= 'KEY'|'DIST'."""
    stations = rng.sample([s for s in STATIONS if s != "Velora"], min(n, len(STATIONS) - 1))
    codes = rng.sample(CODES, min(n, len(CODES)))
    dist_sents = []
    for s, c in zip(stations, codes):
        dist_sents.append(f"The access code for {s} Station is {c}.")
    before_half = dist_sents[:(n + 1) // 2]
    after_half = dist_sents[(n + 1) // 2:]
    fi = 0
    parts = []
    # 前半: 随机穿插 filler 与 before distractor
    for d in before_half:
        parts.append(d)
        for _ in range(rng.randint(0, 2)):
            parts.append(filler[fi % len(filler)]); fi += 1
    parts.append(KEY_SENT)
    for d in after_half:
        parts.append(d)
    # 填充到 ~TARGET_LEN token
    comp = " " + COMPLETION
    def tok(s): return len(tokenizer.encode(" " + s if not s.startswith(" ") else s))
    while tok(" ".join(parts) + comp) < TARGET_LEN - 5:
        parts.append(filler[fi % len(filler)]); fi += 1
    prompt = " ".join(parts) + comp
    cands = [("KEY", KEY_SENT.rstrip("."))]
    for d in dist_sents:
        cands.append(("DIST", d.rstrip(".")))
    return prompt, cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default="results_exp2")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    device = "npu:0"
    verifier, tokenizer, draft, vocab = ch.load(device)
    ans_first = dec.first_subword_draft_id(tokenizer, vocab, ANSWER)
    ans_id = tokenizer.encode(" " + ANSWER)[-1]          # verifier(qwen) token id
    print(f"ANSWER={ANSWER}: qwen_id={ans_id} draft_id={ans_first}")

    rows = []
    for n in LEVELS:
        for seed in range(NSEEDS):
            rng = random.Random(seed * 1000 + n)
            filler = make_filler_pool(seed * 7 + n)
            prompt, cands = build_one(rng, tokenizer, filler, n)
            ids, aux, vlast, lm, doc, toks, T = dec.prepare_inputs(verifier, tokenizer, prompt, device)
            # target 预测 (verifier logits at T-1 -> token T)
            with torch.no_grad():
                vl = verifier(ids).logits[0, T - 1]
            target_ok = bool(int(vl.argmax()) == ans_id)
            # draft
            logits, dbg = ch.run_one(draft, aux, ids, lm, vlast, doc)
            q = logits.shape[1]; row = ch.answer_row(q, draft.block_size)
            pr = torch.softmax(logits[0, row], dim=-1)
            draft_ok = bool(int(pr.argmax()) == ans_first)
            p_true = float(pr[ans_first]) if ans_first is not None else None
            # 候选句 span 定位 (L3+L4 注意力, 答案槽 row)
            spans = []
            for lab, needle in cands:
                sp = dec.find_span(toks, needle)
                if not sp:
                    sp = []
                spans.append((lab, sp))
            a3 = dbg[3]["attn"]; a4 = dbg[4]["attn"]
            mass = {}
            for lab, sp in spans:
                if sp:
                    m = float(a3[:, row, sp].sum()) + float(a4[:, row, sp].sum())
                else:
                    m = float("-inf")
                mass[lab] = m
            key_mass = mass.get("KEY", float("-inf"))
            dist_masses = [v for k, v in mass.items() if k != "KEY" and v > float("-inf")]
            rank_key = 1 + sum(1 for v in dist_masses if v > key_mass)  # 1=最高
            hit_top1 = rank_key == 1
            rows.append({"n": n, "seed": seed, "T": int(T),
                         "target_ok": target_ok, "draft_ok": draft_ok,
                         "draft_p_true": p_true, "rank_key": rank_key, "hit_top1": hit_top1,
                         "spans": {lab: sp for lab, sp in spans}})
    (out / "exp2_samples.json").write_text(json.dumps(rows, indent=2))

    # 汇总
    summ = {}
    for n in LEVELS:
        sub = [r for r in rows if r["n"] == n]
        cells = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}
        for r in sub:
            k = ("T" if r["target_ok"] else "F") + ("T" if r["draft_ok"] else "F")
            cells[k] += 1
        tc = [r for r in sub if r["target_ok"]]
        summ[n] = {"n_samples": len(sub), "cells": cells,
                   "p_true_tc": (sum(r["draft_p_true"] for r in tc) / len(tc)) if tc else None,
                   "hit_top1_all": sum(r["hit_top1"] for r in sub) / len(sub),
                   "hit_top1_tc": (sum(r["hit_top1"] for r in tc) / len(tc)) if tc else None,
                   "n_target_wrong_draft_right": cells["FT"]}
    (out / "exp2_summary.json").write_text(json.dumps(summ, indent=2))
    print("n | TT TF FT FF | P(true|tgt_ok) | hit_top1(all) hit_top1(tgt_ok) | #tgtX_draftOK")
    for n in LEVELS:
        s = summ[n]
        c = s["cells"]
        print(f"{n:2d} | {c['TT']:3d} {c['TF']:3d} {c['FT']:3d} {c['FF']:3d} | "
              f"{('-' if s['p_true_tc'] is None else round(s['p_true_tc'],3)):>6} | "
              f"{s['hit_top1_all']:.2f} {s['hit_top1_tc'] if s['hit_top1_tc'] is not None else float('nan'):.2f} | {c['FT']}")
    print(f"\n已存 {out}/exp2_summary.json & exp2_samples.json")


if __name__ == "__main__":
    main()
