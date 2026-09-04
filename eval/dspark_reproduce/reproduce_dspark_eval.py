#!/usr/bin/env python3
"""用真实投机解码复现 HF DSpark 评测(不依赖 guidellm)。

对 math_reasoning 的 prompt 逐个发生成请求(completions 端点),
读取 vLLM /metrics 里 vllm:spec_decode 计数器在生成前后的增量,
计算逐位置接受率与平均接受长度,并与 HF 上报(Avg Length 5.37)对比。

用法:
    python3 reproduce_dspark_eval.py [--endpoint http://localhost:1125]
            [--max-tokens 128] [--data math_reasoning/math_reasoning.jsonl]
"""
import argparse
import json
import re
import time
import urllib.request

DEFAULT_ENDPOINT = "http://localhost:1125"
DATA = "/home/y50063564/processed_data/archive_dspark_20260904/eval/dspark_reproduce/math_reasoning/math_reasoning.jsonl"
MODEL = "/home/y50063564/Qwen3-4B"

# guidellm 文档里的采样配置(与 speculators eval 一致)
TEMP, TOP_P, TOP_K = 0.6, 0.95, 20


def fetch_metrics(endpoint: str) -> dict:
    with urllib.request.urlopen(f"{endpoint}/metrics", timeout=10) as r:
        text = r.read().decode()
    out: dict[tuple, float] = {}
    # 匹配 vllm:spec_decode 计数器(可能带 _total 后缀与 position 标签)
    pat = re.compile(
        r"vllm:spec_decode_([a-z_]+?)(?:_total)?\{([^}]*)\}\s+([0-9.eE+-]+)"
    )
    for name, labels, val in pat.findall(text):
        pos = re.search(r'position="(\d+)"', labels)
        key = (name, pos.group(1) if pos else None)
        out[key] = float(val)
    return out


def generate(endpoint: str, prompt: str, max_tokens: int, temperature: float,
             top_p: float, top_k: int, stream: bool) -> str:
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": stream,
    }
    req = urllib.request.Request(
        f"{endpoint}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        if stream:
            text = r.read().decode()
            # 从 SSE 里拼出文本(简化)
            chunks = re.findall(r'"text":"((?:[^"\\]|\\.)*)"', text)
            return "".join(chunks)
        resp = json.loads(r.read())
    return resp["choices"][0]["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--stream", action="store_true", help="用流式请求(默认非流式)")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条(0=全部)")
    args = ap.parse_args()

    prompts = [json.loads(l)["prompt"] for l in open(args.data, encoding="utf-8")]
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"样本数: {len(prompts)}, max_tokens={args.max_tokens}, "
          f"temp={TEMP}, top_p={TOP_P}, top_k={TOP_K}, stream={args.stream}")

    m0 = fetch_metrics(args.endpoint)
    t0 = time.time()
    for i, p in enumerate(prompts):
        generate(args.endpoint, p, args.max_tokens, TEMP, TOP_P, TOP_K, args.stream)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(prompts)} 完成, 已用 {time.time()-t0:.0f}s")
    m1 = fetch_metrics(args.endpoint)
    print(f"生成完成, 耗时 {time.time()-t0:.0f}s")

    def delta(name):
        return m1.get((name, None), 0.0) - m0.get((name, None), 0.0)

    drafts = delta("num_drafts")
    accepted = delta("num_accepted_tokens")
    draft_tokens = delta("num_draft_tokens")
    print(f"\ndrafts={drafts:.0f}, accepted={accepted:.0f}, draft_tokens={draft_tokens:.0f}")
    if drafts <= 0:
        print("!! 没有发生投机解码(drafts=0), 检查服务器配置。")
        return

    mean_len = 1 + accepted / drafts
    per_pos = {}
    for (name, pos), v in m1.items():
        if name == "num_accepted_tokens_per_pos" and pos is not None:
            per_pos[int(pos)] = v - m0.get((name, pos), 0.0)

    print(f"\n=== 复现结果 (math_reasoning, {len(prompts)} 请求) ===")
    print(f"平均接受长度 (含 bonus): {mean_len:.3f}")
    print("逐位置接受率:")
    pos_str = []
    for pos in sorted(per_pos):
        rate = per_pos[pos] / drafts * 100
        pos_str.append(f"Pos{pos} {rate:.1f}%")
    print("  " + " | ".join(pos_str))

    print("\n=== HF 上报 (math_reasoning) ===")
    print("Avg Length 5.37")
    print("Pos0-7: 87.7% | 75.7% | 65.1% | 56.3% | 48.2% | 40.9% | 34.6% | 28.8%")
    print(f"\n差异: avg_len {mean_len - 5.37:+.3f}")


if __name__ == "__main__":
    main()
