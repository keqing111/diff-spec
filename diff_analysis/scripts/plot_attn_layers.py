"""采集并绘制 l0_5l 逐层注意力分布 (卡 11)。

给定样本 -> Qwen3-4B 隐层 -> eager DSpark -> forward -> 采集每层注意力,
绘制:
  A. 每层: 最后 anchor block 的 8 个 query x KV 位置的热力图(按 head 平均 + 逐 head 网格)
  B. diff 层(layer0): 每个 pair 的 branch0 / branch1 / 相减 注意力
输出 PNG 到 $OUTDIR (默认 <repo>/scripts/attn_out/)。
用法: ASCEND_RT_VISIBLE_DEVICES=11 python scripts/plot_attn_layers.py
"""
import os
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
CHECKPOINT = "/home/y50063564/processed_data/dspark_data/dspark_diff_l0_5l/checkpoints/checkpoint_best"
TARGET_LAYERS = [1, 9, 17, 25, 33]
VERIFIER_LAST = 36
OUTDIR = Path("/home/y50063564/processed_data/dspark_data/attn_plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)
import torch_npu  # noqa: F401
device = "npu:0"

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("/home/y50063564/Qwen3-4B")
ids = tok(SAMPLE, return_tensors="pt")["input_ids"].to(device)
seq = ids.shape[1]
tokens = tok.convert_ids_to_tokens(ids[0].tolist())
print(f"seq={seq}, 无BOS, 首token={tokens[0]}")

verifier = (
    AutoModelForCausalLM.from_pretrained("/home/y50063564/Qwen3-4B", dtype=torch.bfloat16)
    .to(device)
    .eval()
)
with torch.no_grad():
    out = verifier(ids, output_hidden_states=True)
hs = out.hidden_states
aux = torch.cat([hs[l] for l in TARGET_LAYERS], dim=-1)
verifier_last = hs[VERIFIER_LAST]
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
for layer in model.layers:
    layer.self_attn.capture_attention = True

loss_mask = torch.ones(1, seq, device=device)
doc_ids = torch.zeros(1, seq, dtype=torch.long, device=device)
max_anchors = max(1, seq // 8)
with torch.no_grad():
    _, loss, _ = model(
        hidden_states=aux, input_ids=ids, loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last, document_ids=doc_ids,
        max_anchors=max_anchors,
    )
print(f"forward OK, loss={float(loss):.4f}")

block = model.block_size
q_len = max_anchors * block
kv_len = seq + q_len
lb = (max_anchors - 1) * block  # 最后 block 起始 query
qrows = list(range(lb, q_len))
print(f"block={block}, q_len={q_len}, kv_len={kv_len}, last_block_queries={qrows}")

def save_heatmap(data, path, title, xtick=None, cmap="viridis"):
    """data: 2D [rows=queries, cols=kv]; xtick: 每列的文字标签(或 None)"""
    fig, ax = plt.subplots(figsize=(max(10, data.shape[1] * 0.12), max(4, data.shape[0] * 0.7)))
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    fig.colorbar(im, ax=ax, label="attn")
    if xtick:
        ax.set_xticks(range(len(xtick)))
        ax.set_xticklabels(xtick, rotation=90, fontsize=5)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels([f"q{lb+i}" for i in range(data.shape[0])], fontsize=7)
    ax.set_xlabel("KV position (base tokens then mask)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")

base_labels = tokens + [f"M{i}" for i in range(q_len)]  # KV 标签

# ===== A. 每层最后 block 热力图 (按 head 平均) =====
for li, layer in enumerate(model.layers):
    attn = getattr(layer.self_attn, "_debug_attn", None)
    if attn is None:
        print(f"[layer {li}] 无权重"); continue
    a = attn[0].float()  # [heads, q_len, kv_len]
    avg = a[:, qrows, :].mean(dim=0).cpu().numpy()  # [8, kv_len]
    save_heatmap(avg, OUTDIR / f"layer{li}_lastblock_avgattn.png",
                 f"layer{li} ({type(layer.self_attn).__name__}) avg over {a.shape[0]} heads")
    # 逐 head 网格
    nh = a.shape[0]
    cols = 4
    rows = (nh + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 2.5))
    for h in range(nh):
        ax = axes.flat[h]
        im = ax.imshow(a[h, qrows, :].cpu().numpy(), aspect="auto", cmap="viridis")
        ax.set_title(f"h{h}")
        ax.set_yticks([])
    for h in range(nh, rows * cols):
        axes.flat[h].axis("off")
    fig.suptitle(f"layer{li} last-block per-head attention (query rows {lb}..{q_len-1})")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"layer{li}_lastblock_perhead.png", dpi=100)
    plt.close(fig)
    print(f"  saved layer{li}_lastblock_perhead.png")

# ===== B. diff 层 (layer0) 分支0 / 分支1 / 相减 =====
diff_attn = model.layers[0].self_attn
if hasattr(diff_attn, "_debug_branch0"):
    b0 = diff_attn._debug_branch0[0].float()  # [16 pairs, q_len, kv_len]
    b1 = diff_attn._debug_branch1[0].float()
    sub = diff_attn._debug_attn[0].float()
    npairs = b0.shape[0]
    # 每个 pair: 取"最后 block 平均 query"或最后 query —— 逐 query 全存
    torch.save({"branch0": b0.cpu(), "branch1": b1.cpu(), "subtracted": sub.cpu(),
                "tokens": tokens, "q_len": q_len, "kv_len": kv_len},
               OUTDIR / "layer0_diff_branches.pt")
    # 最后 query (q_len-1): 16 个 pair 的 branch0/branch1/相减 曲线
    q = q_len - 1
    fig, axes = plt.subplots(4, 4, figsize=(20, 12))
    for p in range(npairs):
        ax = axes.flat[p]
        ax.plot(b0[p, q, :].cpu().numpy(), color="tab:blue", lw=0.8, label="branch0")
        ax.plot(b1[p, q, :].cpu().numpy(), color="tab:orange", lw=0.8, label="branch1")
        ax.plot(sub[p, q, :].cpu().numpy(), color="tab:green", lw=0.8, label="diff")
        ax.set_title(f"pair{p} query{q}")
        ax.set_xticks([])
        if p % 4 == 0:
            ax.set_ylabel("attn")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"diff layer0: per-pair branch0/branch1/subtracted at last query q{q} (x=KV)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "layer0_diff_pairs_lastquery.png", dpi=100)
    plt.close(fig)
    print("  saved layer0_diff_pairs_lastquery.png")
    # branch0/branch1/相减 的"最后block平均"热力图
    for nm, tensor in [("branch0", b0), ("branch1", b1), ("subtracted", sub)]:
        avg = tensor[:, qrows, :].mean(dim=1).cpu().numpy()  # [16 pairs, kv_len]
        save_heatmap(avg, OUTDIR / f"layer0_diff_{nm}_lastblock_avg.png",
                     f"diff layer0 {nm}: avg over last-block queries, rows=pairs(16) x KV",
                     cmap="RdBu_r" if nm == "subtracted" else "viridis")

print(f"\n全部输出在: {OUTDIR}")
print("done")
