#!/usr/bin/env python3
"""实验1: 长度×位置矩阵 — 关键token注意力是否随 长度/位置 保持。

长度 [128,256,512,1024,2048] × 关键句位置 [10%,25%,50%,75%,90%] = 25 组合, 每组合1次。
度量: 答案槽位每层对 关键token(KeQing 所有子词) 的注意力占比 + 答案首子词 P。

用法:
    ASCEND_RT_VISIBLE_DEVICES=15 python3 exp1_length_position.py [-o outdir]
"""
import argparse
import json
import random
from pathlib import Path

import torch

from draft_exp_common import (load_models, build_vocab, run_forward, answer_slot,
                             find_span, prepare_inputs, first_subword_draft_id)

KEY = "The access code for Velora Station is Alpha."
COMPLETION = "The access code for Velora Station is"
ANSWER = "Alpha"
LENGTHS = [128, 256, 512, 1024, 2048]
POSITIONS = [10, 25, 50, 75, 90]

# 填充句模板(程序化生成, 多样不重复)
_T = [
    "The {a} {b} was {c} before the {d} {e}.",
    "A {f} {g} the {h} near the {i} every {j}.",
    "During the {k}, the {l} {m} across the {n}.",
    "The {o} on the {p} {q} the {r} to the {s}.",
    "Historians {t} that the {u} was {v} in the {w} century.",
    "Scientists {x} the {y} of the {z} under controlled conditions.",
    "The {aa} {bb} the {cc} with a {dd} before the {ee}.",
    "A {ff} in the {gg} {hh} the {ii} for the {jj}.",
]
_W = {
    "a": ["old harbor", "young chemist", "medieval bridge", "distant galaxy", "quiet village"],
    "b": ["was built", "recorded data", "stood firm", "lay frozen", "opened early"],
    "c": ["constructed", "analyzed", "repaired", "observed", "surveyed"],
    "d": ["main road", "local council", "research team", "winter storm", "autumn festival"],
    "e": ["arrived", "decided", "reported", "gathered", "completed"],
    "f": ["researcher", "gardener", "engineer", "librarian", "sailor"],
    "g": ["examined", "watered", "inspected", "cataloged", "navigated"],
    "h": ["sample", "garden", "bridge", "archive", "channel"],
    "i": ["lab", "river", "hill", "library", "coast"],
    "j": ["morning", "evening", "season", "decade", "year"],
    "k": ["winter", "harvest", "conference", "migration", "excavation"],
    "l": ["birds", "farmers", "delegates", "herds", "volunteers"],
    "m": ["migrated", "gathered", "traveled", "returned", "marched"],
    "n": ["plain", "valley", "border", "field", "square"],
    "o": ["clock", "weather vane", "old sign", "bronze statue", "wooden gate"],
    "p": ["tower", "roof", "wall", "piazza", "fence"],
    "q": ["pointed to", "overlooked", "guarded", "showed", "marked"],
    "r": ["direction", "valley", "entrance", "time", "boundary"],
    "s": ["north", "east", "river", "town", "forest"],
    "t": ["believe", "disagree", "conclude", "estimate", "recall"],
    "u": ["fortress", "temple", "aqueduct", "market", "settlement"],
    "v": ["erected", "abandoned", "expanded", "burned", "founded"],
    "w": ["third", "ninth", "twelfth", "fifteenth", "nineteenth"],
    "x": ["measured", "estimated", "replicated", "classified", "plotted"],
    "y": ["density", "volume", "temperature", "pressure", "composition"],
    "z": ["gas", "alloy", "crystal", "liquid", "solution"],
    "aa": ["mechanic", "archivist", "beekeeper", "translator", "cartographer"],
    "bb": ["cleaned", "restored", "fed", "copied", "drew"],
    "cc": ["engine", "manuscript", "hives", "documents", "maps"],
    "dd": ["new wrench", "soft brush", "smoker", "scanner", "compass"],
    "ee": ["long trip", "exhibition", "winter", "deadline", "survey"],
    "ff": ["farmer", "student", "potter", "weaver", "guard"],
    "gg": ["north field", "workshop", "studio", "loom", "watchtower"],
    "hh": ["planted", "hammered", "glazed", "wove", "patrolled"],
    "ii": ["wheat", "metal", "clay", "thread", "gate"],
    "jj": ["market", "frame", "kiln", "blanket", "watch"],
}


def make_filler_pool(seed: int, n: int = 200) -> list[str]:
    rng = random.Random(seed)
    seen = set()
    pool = []
    while len(pool) < n:
        t = rng.choice(_T)
        s = t.format(**{k: rng.choice(v) for k, v in _W.items()})
        if s not in seen:
            seen.add(s); pool.append(s)
    return pool


def build_prompt(L: int, pct: int, tokenizer, filler: list[str]) -> str:
    completion = " " + COMPLETION
    q_tokens = len(tokenizer.encode(completion))
    key_tokens = len(tokenizer.encode(" " + KEY))
    avail = L - q_tokens - key_tokens
    before_target = int(avail * pct / 100)
    before, used = [], 0
    fi = 0
    while used < before_target:
        s = filler[fi % len(filler)]; fi += 1
        before.append(s); used += len(tokenizer.encode(" " + s))
    after, used2 = [], 0
    while len(tokenizer.encode(" ".join(before) + " " + KEY + " " + " ".join(after) + completion)) < L:
        s = filler[fi % len(filler)]; fi += 1
        after.append(s)
    return " ".join(before) + " " + KEY + " " + " ".join(after) + completion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default=Path(__file__).parent / "exp1_results")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    device = "npu:0"
    verifier, tokenizer, draft = load_models(device)
    vocab = build_vocab(tokenizer)
    ans_first_id = first_subword_draft_id(tokenizer, vocab, ANSWER)
    print(f"答案 {ANSWER!r} 首子词 draft_id = {ans_first_id}")

    filler = make_filler_pool(42)
    rows = []
    for L in LENGTHS:
        for pct in POSITIONS:
            prompt = build_prompt(L, pct, tokenizer, filler)
            ids, aux, vlast, lm, doc, toks, T = prepare_inputs(verifier, tokenizer, prompt, device)
            cap = run_forward(draft, aux, ids, lm, vlast, doc)
            slot = answer_slot(draft)
            span = find_span(toks, ANSWER)
            # 每层答案槽位对关键 token 跨度的注意力
            attn_share = []
            for l in range(5):
                a = cap["layer_attn"][l][0, :, slot, :T].mean(dim=0)  # [T]
                attn_share.append(a[span].sum().item())
            p = torch.softmax(cap["logits"][0, slot], dim=-1)[ans_first_id].item() if ans_first_id is not None else None
            arg = int(cap["logits"][0, slot].argmax().item())
            arg_tok = tokenizer.decode([vocab[arg]]) if arg < len(vocab) else "?"
            rows.append({"length": L, "position_pct": pct, "actual_T": T,
                         "key_span": span, "key_attn_by_layer": attn_share,
                         "mean_key_attn": float(sum(attn_share)/5), "p_first_subword": p,
                         "argmax_token": arg_tok})
            print(f"L={L:5d} pos={pct:3d}%  T={T:5d}  关键token注意力(5层均值)={sum(attn_share)/5*100:5.1f}%  "
                  f"P(首子词)={p if p is None else round(p,4)}  预测={arg_tok!r}")

    (out / "exp1_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\n已存 {out}/exp1_results.json")


if __name__ == "__main__":
    main()
