"""baseline(GQA) 逐层逐头注意力分布 — 与 layer0_diff_pairs_lastquery.png 相同样本/位置。

相同样本(Japan/Tokyo) + seed(0) + 末尾 query, 对 baseline(checkpoint_best) 每层
画出 32 个 head 在该 query 对 base token 的注意力曲线。
用法: ASCEND_RT_VISIBLE_DEVICES=11 python scripts/plot_baseline_attention.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "/home/y50063564/dspark_project/speculators/src")

SAMPLE = ("The capital city of Japan is Tokyo. The weather today is pleasant and "
          "the sky is clear across the whole region. Many people enjoy taking a "
          "quiet walk in the park during the afternoon. Reading books is a popular "
          "way to spend a peaceful evening at home. The local library opens early "
          "in the morning and stays open until late. The capital city of Japan is")
CHECKPOINT = "/home/y50063564/checkpoint_best"
TARGET_LAYERS = [1, 9, 17, 25, 33]
VERIFIER_LAST = 36
OUTDIR = Path("/home/y50063564/processed_data/dspark_data/attn_plots/baseline")
OUTDIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)
import torch_npu  # noqa: F401
device = "npu:0"

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("/home/y50063564/Qwen3-4B")
ids = tok(SAMPLE, return_tensors="pt")["input_ids"].to(device)
seq = ids.shape[1]
tokens = tok.convert_ids_to_tokens(ids[0].tolist())

verifier = (
    AutoModelForCausalLM.from_pretrained("/home/y50063564/Qwen3-4B", dtype=torch.bfloat16).to(device).eval()
)
with torch.no_grad():
    out = verifier(ids, output_hidden_states=True)
hs = out.hidden_states
aux = torch.cat([hs[l] for l in TARGET_LAYERS], dim=-1)
vlast = hs[VERIFIER_LAST]
del verifier, out, hs
torch.npu.empty_cache()

from safetensors.torch import load_file  # noqa: E402
from speculators.models.dspark import DSparkDraftModel  # noqa: E402
from speculators.models.dspark.config import DSparkSpeculatorConfig  # noqa: E402
config = DSparkSpeculatorConfig.from_pretrained(CHECKPOINT)
config.transformer_layer_config._attn_implementation = "eager"
model = DSparkDraftModel(config)
model.load_state_dict(load_file(f"{CHECKPOINT}/model.safetensors"), strict=False)
model.load_verifier_weights()
model.to(device).eval()
for l in model.layers:
    l.self_attn.capture_attention = True

loss_mask = torch.ones(1, seq, device=device)
doc_ids = torch.zeros(1, seq, dtype=torch.long, device=device)
max_anchors = max(1, seq // 8)
with torch.no_grad():
    model(hidden_states=aux, input_ids=ids, loss_mask=loss_mask,
          verifier_last_hidden_states=vlast, document_ids=doc_ids, max_anchors=max_anchors)
block = model.block_size
q_len = max_anchors * block
lb = (max_anchors - 1) * block
q = q_len - 1  # 末尾 query
print(f"seq={seq}, q_len={q_len}, 末尾 query q={q} (block#{q-lb}), 同 layer0_diff_pairs_lastquery.png")

kv_len = model.layers[0].self_attn._debug_attn.shape[-1]
for li, layer in enumerate(model.layers):
    attn = getattr(layer.self_attn, "_debug_attn", None)
    if attn is None:
        print(f"[layer{li}] 无权重"); continue
    a = attn[0, :, q, :].float().cpu().numpy()  # [heads, 全部 KV = base + block]
    nh = a.shape[0]
    cols = 8
    rows = (nh + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.0))
    for h in range(nh):
        ax = axes.flat[h]
        ax.plot(a[h], lw=0.8)
        ax.set_title(f"h{h}", fontsize=7)
        ax.axvline(seq - 0.5, color="gray", ls="--", lw=0.8)  # base/block 分界
        ax.set_yticks([])
        # base 区每 5 个标 token
        for i in range(0, seq, 5):
            ax.axvline(i, color="lightgray", lw=0.3)
        ax.set_xticks(list(range(0, seq, 10)) + [kv_len - 1])
        ax.set_xticklabels(
            ([tokens[i] for i in range(0, seq, 10)] + ["block"]), rotation=90, fontsize=3)
        for i, t in enumerate(tokens):
            if "Tokyo" in t:
                ax.axvline(i, color="red", ls=":", lw=0.6)
    for h in range(nh, rows * cols):
        axes.flat[h].axis("off")
    fig.suptitle(f"baseline layer{li} (GQA) 逐 head 注意力 @ q{q} | 左base(0-{seq-1}) 右block/mask({seq}-{kv_len-1}) | 红虚线=Tokyo", fontsize=9)
    fig.tight_layout()
    out = OUTDIR / f"baseline_layer{li}_perhead_q{q}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}")

# 保存原始张量
torch.save({f"layer{l}": getattr(model.layers[l].self_attn, "_debug_attn", None)
            for l in range(len(model.layers))}, OUTDIR / "baseline_attn.pt")
print("saved baseline_attn.pt")
print("done")
