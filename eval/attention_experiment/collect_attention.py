#!/usr/bin/env python3
"""采集 Qwen3-4B 生成答案 token 时各层各头的注意力原始数据(只采集,不分析)。

方法:
- 本地 Qwen3-4B + torch_npu, eager 注意力(output_attentions=True 才返回权重)。
- 手动自回归循环,greedy 解码,seed=42。
- 每个生成步抓取 32 层 × 32 头的注意力(取最后 query 位置对全上下文的权重)。
- 每样本存一个 .pt(含注意力张量 + token 元数据 + 关键位置索引),另有 manifest.json。

用法:
    python3 collect_attention.py  [-o <输出目录>]
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch

# 依赖安装: pip install torch_npu transformers

MODEL_PATH = "/home/y50063564/Qwen3-4B"
SEED = 42
MAX_NEW_TOKENS = 8

# 中性填充句(循环使用,用于把样本 padding 到目标长度)
FILLER = [
    "The weather today is pleasant and the sky is clear across the whole region.",
    "Many people enjoy taking a quiet walk in the park during the afternoon.",
    "Reading books is a popular way to spend a peaceful evening at home.",
    "The local library opens early in the morning and stays open until late.",
    "A small cafe near the station serves fresh coffee and warm bread.",
]

# 10 个样本: 答案明确写在上下文里; filler 插在答案句与 query 之间,
# 使长样本中答案离 query 更远(正好测"关键 token 远离 query"的场景)。
SAMPLES = [
    {"id": "s01_statement_tokyo_short", "target_len": 60, "style": "statement",
     "answer_word": "Tokyo",
     "answer_sentence": "The capital city of Japan is Tokyo.",
     "lead": "",
     "query": "The capital city of Japan is"},

    {"id": "s02_statement_paris_short", "target_len": 150, "style": "statement",
     "answer_word": "Paris",
     "answer_sentence": "The capital city of France is Paris.",
     "lead": "",
     "query": "The capital city of France is"},

    {"id": "s03_qa_italy_short", "target_len": 200, "style": "qa",
     "answer_word": "Rome",
     "answer_sentence": "The capital of Italy is Rome.",
     "lead": "Some people wonder about geography. ",
     "query": "So, what is the capital of Italy?"},

    {"id": "s04_statement_sky_short", "target_len": 300, "style": "statement",
     "answer_word": "blue",
     "answer_sentence": "The sky appears blue because blue light scatters more strongly than other colors.",
     "lead": "This is called Rayleigh scattering. ",
     "query": "Therefore, the color of the sky is"},

    {"id": "s05_math_medium", "target_len": 700, "style": "statement",
     "answer_word": "42",
     "answer_sentence": "It is a well-known fact that the product of 6 and 7 is 42.",
     "lead": "",
     "query": "The product of 6 and 7 is"},

    {"id": "s06_qa_australia_far", "target_len": 1100, "style": "qa",
     "answer_word": "Canberra",
     "answer_sentence": "The capital of Australia is Canberra.",
     "lead": ("Some people mistakenly think the capital of Australia is Sydney, "
              "because Sydney is the largest city. But Sydney is not the capital. "
              "Melbourne also served as the capital before Canberra. "),
     "query": "So, what is the capital of Australia?"},

    {"id": "s07_statement_sun_medium", "target_len": 900, "style": "statement",
     "answer_word": "east",
     "answer_sentence": "The Sun rises in the east every morning and sets in the west every evening.",
     "lead": "This happens because the Earth rotates from west to east. ",
     "query": "So, the Sun rises in the"},

    {"id": "s08_statement_jupiter_long", "target_len": 1600, "style": "statement",
     "answer_word": "Jupiter",
     "answer_sentence": "The largest planet in the solar system is Jupiter.",
     "lead": ("The solar system contains eight planets, each with different sizes. "
              "Some planets are small and rocky, while others are large gas giants. "),
     "query": "The largest planet in the solar system is"},

    {"id": "s09_statement_moon_long", "target_len": 1900, "style": "statement",
     "answer_word": "Neil",
     "answer_sentence": ("In 1969, humans first set foot on the Moon during the Apollo 11 "
                         "mission, and the commander of that mission was Neil Armstrong."),
     "lead": ("Space exploration has a long and fascinating history that people study "
              "around the world. Many famous missions have been launched over the decades, "
              "and each one added new knowledge about our universe. "),
     "query": "The first person to walk on the Moon was"},

    {"id": "s10_statement_madrid_medium", "target_len": 800, "style": "statement",
     "answer_word": "Madrid",
     "answer_sentence": "The capital of Spain is Madrid.",
     "lead": ("Spain is a country in southwestern Europe with a rich culture and "
              "many famous cities. "),
     "query": "The capital of Spain is"},
]


def build_prompt(sample: dict, tokenizer) -> str:
    """组装 prompt: lead + answer_sentence + 填充到 target_len + query。"""
    base = (sample.get("lead", "") + sample["answer_sentence"]).strip()
    query = sample["query"]
    target = sample["target_len"]
    prompt = base
    n = 0
    # 在答案句与 query 之间填充,直到(不含 query)长度接近 target
    while True:
        cur_len = len(tokenizer(prompt + " " + query)["input_ids"])
        if cur_len >= target:
            break
        prompt += " " + FILLER[n % len(FILLER)]
        n += 1
        if n > 2000:  # 安全阀
            break
    return prompt + " " + query


def token_spans(tokenizer, input_ids: list[int]) -> list[str]:
    """逐 token 解码(含特殊 token 占位),用于定位关键位置。"""
    out = []
    for tid in input_ids:
        out.append(tokenizer.decode([tid], skip_special_tokens=False))
    return out


def find_span(tokens: list[str], needle: str, max_span: int = 80) -> list[int]:
    """找"连续 token 跨度"其拼接文本包含 needle(小写归一化)。

    处理多 token 答案(如 "42" 被拆成 "4"+"2")与整句匹配。
    返回首个匹配跨度的 token 索引列表;未找到返回 []。
    """
    norm = re.sub(r"\s+", "", needle.lower())
    texts = [re.sub(r"\s+", "", t.lower()) for t in tokens]
    best = None  # (start, end) 最小窗口
    for start in range(len(tokens)):
        acc = ""
        for end in range(start, min(len(tokens), start + max_span)):
            acc += texts[end]
            if norm in acc:
                if best is None or (end - start) < (best[1] - best[0]):
                    best = (start, end)
                break  # 该起点已含 needle, 试下一个起点找更紧的
            if len(acc) > len(norm) + 40:  # 超出太多就换起点
                break
    if best is None:
        return []
    return list(range(best[0], best[1] + 1))


@torch.no_grad()
def collect_one(model, tokenizer, sample: dict, out_dir: Path) -> dict:
    prompt = build_prompt(sample, tokenizer)
    ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    tokens = token_spans(tokenizer, ids)
    num_prompt = len(ids)

    gen_ids: list[int] = []
    cur = input_ids
    captured: list[dict] = []
    for step in range(MAX_NEW_TOKENS):
        t0 = time.time()
        out = model(input_ids=cur, output_attentions=True)
        # out.attentions: tuple[32], 每个 [1, heads, T, T]
        # 取最后 query 位置的注意力: [1, heads, T] -> [heads, T]
        step_attn = torch.stack([a[0, :, -1, :] for a in out.attentions])  # [32层, 32头, T]
        step_attn = step_attn.cpu().to(torch.float32)
        captured.append({
            "step": step,
            "seq_len_at_step": int(cur.shape[1]),
            "attn": step_attn,  # [L, H, T]
        })
        next_id = int(out.logits[0, -1].argmax().item())
        gen_ids.append(next_id)
        cur = torch.cat([cur, torch.tensor([[next_id]], dtype=torch.long, device=model.device)], dim=1)
        if step == 0:
            print(f"    step0 生成 token: {tokenizer.decode([next_id])!r} "
                  f"(期望答案 {sample['answer_word']!r}), seq_len={num_prompt}, 耗时 {time.time()-t0:.1f}s")

    # 关键位置(多 token span 匹配)
    answer_word_pos = find_span(tokens, sample["answer_word"])
    answer_sentence_pos = find_span(tokens, sample["answer_sentence"])
    gen_tokens = [tokenizer.decode([i]) for i in gen_ids]

    # 答案落在生成的哪一步(累计生成文本首次包含答案词)
    norm_ans = re.sub(r"\s+", "", sample["answer_word"].lower())
    gen_acc = ""
    answer_step_first = None
    for step, tid in enumerate(gen_ids):
        gen_acc += tokenizer.decode([tid], skip_special_tokens=False)
        if norm_ans in re.sub(r"\s+", "", gen_acc.lower()):
            answer_step_first = step
            break

    rec = {
        "id": sample["id"], "style": sample["style"], "target_len": sample["target_len"],
        "prompt_text": prompt, "answer_word": sample["answer_word"],
        "answer_sentence": sample["answer_sentence"],
        "input_ids": ids, "tokens": tokens, "num_prompt_tokens": num_prompt,
        "answer_word_positions": answer_word_pos,
        "answer_sentence_positions": answer_sentence_pos,
        "bos_position": 0,
        "last_prompt_position": num_prompt - 1,
        "generated_ids": gen_ids, "generated_tokens": gen_tokens,
        "num_generated": len(gen_ids),
        "answer_step_first": answer_step_first,  # 答案首次出现的生成步(可能>0,如 qa 式/多 token)
        "attentions": captured,  # 每步 {step, seq_len_at_step, attn[L,H,T]}
    }
    torch.save(rec, out_dir / f"{sample['id']}.pt")
    return {
        "id": sample["id"], "style": sample["style"], "target_len": sample["target_len"],
        "prompt_tokens": num_prompt, "generated": gen_tokens,
        "answer_word_positions": answer_word_pos,
        "answer_sentence_positions": answer_sentence_pos,
        "answer_step_first": answer_step_first,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default="/home/y50063564/processed_data/archive_dspark_20260904/eval/attention_raw")
    args = ap.parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    # 需要在 torch_npu 环境下: 这里探测设备
    import torch_npu  # noqa: F401
    if torch.npu.is_available():
        device = "npu:0"
    else:
        device = "cpu"
    print(f"设备: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"加载模型 {MODEL_PATH} (eager, bf16) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.to(device)
    model.eval()

    # 最小前向验证: output_attentions 是否返回
    probe = tokenizer("hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        po = model(**probe, output_attentions=True)
    if po.attentions is None:
        raise RuntimeError("output_attentions 返回 None, 无法采集注意力。")
    print(f"output_attentions OK, 层数={len(po.attentions)}, "
          f"每层形状={tuple(po.attentions[0].shape)}")

    manifest = []
    for s in SAMPLES:
        print(f"[{s['id']}] target_len={s['target_len']} ...")
        try:
            info = collect_one(model, tokenizer, s, out_dir)
            manifest.append(info)
            print(f"    完成: prompt_tokens={info['prompt_tokens']}, "
                  f"答案词位置={info['answer_word_positions']}")
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED: {e}")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"完成。原始数据在 {out_dir} (manifest.json + 每样本 .pt)")


if __name__ == "__main__":
    main()
