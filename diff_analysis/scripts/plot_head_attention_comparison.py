"""l0_5l vs baseline: 同一样本(Japan)逐层注意力分布对比。

- 轴: x = KV position (0-70 base, 71-134 block), y = attention weight
- l0_5l layer0(diff): 每 pair 画 branch0(蓝) + branch1(橙) 两条线(同 plot_attn_layers.py)
- GQA 层: 每 head 一条线
用法: ASCEND_RT_VISIBLE_DEVICES=11 python3 plot_head_attention_comparison.py
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
MODELS = {
    "l0_5l": "/home/y50063564/processed_data/dspark_data/dspark_diff_l0_5l/checkpoints/checkpoint_best",
    "baseline": "/home/y50063564/checkpoint_best",
}
OUTDIR = Path("/home/y50063564/processed_data/archive_dspark_20260904/diff_analysis/head_attention_comparison")
TARGET_LAYERS = [1, 9, 17, 25, 33]
VERIFIER_LAST = 36

import torch_npu  # noqa: F401
device = "npu:0"

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("/home/y50063564/Qwen3-4B")
ids = tok(SAMPLE, return_tensors="pt")["input_ids"].to(device)
seq = ids.shape[1]
tokens = tok.convert_ids_to_tokens(ids[0].tolist())

verifier = (
    AutoModelForCausalLM.from_pretrained("/home/y50063564/Qwen3-4B", dtype=torch.bfloat16)
    .to(device).eval()
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

def build_model(path):
    cfg = DSparkSpeculatorConfig.from_pretrained(path)
    cfg.transformer_layer_config._attn_implementation = "eager"
    m = DSparkDraftModel(cfg)
    m.load_state_dict(load_file(f"{path}/model.safetensors"), strict=False)
    m.load_verifier_weights()
    m.to(device).eval()
    for l in m.layers:
        l.self_attn.capture_attention = True
    return m

def _style(ax, seq, kv_len, tokens):
    ax.axvline(seq - 0.5, color="k", ls="--", lw=0.8)
    for ti, t in enumerate(tokens):
        if "Tokyo" in t:
            ax.axvline(ti, color="red", ls=":", lw=0.8)
    ax.set_xticks([])

def plot_gqa(attn, model_name, layer_idx, tokens, seq, q, kv_len, out):
    """GQA 层: 每个 head 一条注意力线。"""
    a = attn[0].float().cpu().numpy()[:, q, :]  # [32, kv]
    n = a.shape[0]
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.2), sharey=True)
    for i in range(n):
        ax = axes.flat[i]
        ax.plot(a[i], lw=0.8, color="steelblue")
        _style(ax, seq, kv_len, tokens)
        if i == 0:
            ax.set_ylabel("attention\nweight", fontsize=8)
        ax.set_title(f"head {i}", fontsize=7)
    for i in range(n, rows * cols):
        axes.flat[i].axis("off")
    for i in range(max(0, n - cols), n):
        ax = axes.flat[i]
        ax.set_xticks([0, seq - 1, kv_len - 1])
        ax.set_xticklabels(["base0", f"b{seq-1}", "block"], fontsize=6)
    fig.suptitle(f"{model_name} | layer {layer_idx} | {n} GQA heads | q={q}\n"
                 f"(x: KV pos, base 0-{seq-1} | block {seq}-{kv_len-1}, red=Tokyo)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")

def plot_diff(b0, b1, model_name, layer_idx, tokens, seq, q, kv_len, out):
    """diff 层: 每个 pair 画 branch0(蓝)+branch1(橙) 两条线(相减用绿细线示意)。"""
    n = b0.shape[0]
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 2.8), sharey=True)
    for p in range(n):
        ax = axes.flat[p]
        ax.plot(b0[p, q, :].cpu().numpy(), color="tab:blue", lw=0.9, label="branch0")
        ax.plot(b1[p, q, :].cpu().numpy(), color="tab:orange", lw=0.9, label="branch1")
        _style(ax, seq, kv_len, tokens)
        if p % cols == 0:
            ax.set_ylabel("attention\nweight", fontsize=8)
        ax.set_title(f"diff pair {p}", fontsize=8)
        if p == 0:
            ax.legend(fontsize=6)
    for p in range(n, rows * cols):
        axes.flat[p].axis("off")
    for p in range(max(0, (rows - 1) * cols), n):
        ax = axes.flat[p]
        ax.set_xticks([0, seq - 1, kv_len - 1])
        ax.set_xticklabels(["base0", f"b{seq-1}", "block"], fontsize=6)
    fig.suptitle(f"{model_name} | layer {layer_idx} | {n} diff pairs (branch0-branch1) | q={q}\n"
                 f"(x: KV pos, base 0-{seq-1} | block {seq}-{kv_len-1}, red=Tokyo)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")

loss_mask = torch.ones(1, seq, device=device)
doc_ids = torch.zeros(1, seq, dtype=torch.long, device=device)
max_anchors = max(1, seq // 8)

for name, ckpt in MODELS.items():
    model = build_model(ckpt)
    torch.manual_seed(0)
    with torch.no_grad():
        model(hidden_states=aux, input_ids=ids, loss_mask=loss_mask,
              verifier_last_hidden_states=vlast, document_ids=doc_ids, max_anchors=max_anchors)
    q = max_anchors * model.block_size - 1
    kv_len = model.layers[0].self_attn._debug_attn.shape[-1]
    for li, layer in enumerate(model.layers):
        sa = layer.self_attn
        is_diff = name == "l0_5l" and li == 0
        if is_diff:
            b0 = sa._debug_branch0[0].float()
            b1 = sa._debug_branch1[0].float()
            plot_diff(b0, b1, name, li, tokens, seq, q, kv_len,
                      OUTDIR / name / f"layer{li}_diff{sa.num_heads}_q{q}.png")
        else:
            attn = getattr(sa, "_debug_attn", None)
            if attn is None:
                continue
            plot_gqa(attn, name, li, tokens, seq, q, kv_len,
                     OUTDIR / name / f"layer{li}_gqa{attn.shape[1]}_q{q}.png")
    print(f"{name} done")
    del model
    torch.npu.empty_cache()

print("all done")
