"""逐层采集 DSpark 注意力分布 (l0_5l checkpoint, 卡 11)。
给定样本 -> Qwen3-4B 生成 verifier 隐层 -> eager 构建 DSpark -> forward ->
逐层捕获注意力, 分析末尾 anchor block 的 query 在 base token 上的分布,
检查答案 token "Tokyo" 是否被显著关注。
用法: ASCEND_RT_VISIBLE_DEVICES=11 python scripts/inspect_attn_layers.py
"""
import sys

import torch

sys.path.insert(0, "/home/y50063564/dspark_project/speculators/src")

SAMPLE = ("The capital city of Japan is Tokyo. The weather today is pleasant and "
          "the sky is clear across the whole region. Many people enjoy taking a "
          "quiet walk in the park during the afternoon. Reading books is a popular "
          "way to spend a peaceful evening at home. The local library opens early "
          "in the morning and stays open until late. The capital city of Japan is")
CHECKPOINT = "/home/y50063564/processed_data/dspark_data/dspark_diff_l0_5l/checkpoints/checkpoint_best"
TARGET_LAYERS = [1, 9, 17, 25, 33]
VERIFIER_LAST = 36

torch.manual_seed(0)
import torch_npu  # noqa: F401
device = "npu:0"  # 由 ASCEND_RT_VISIBLE_DEVICES 决定实际物理卡
print(f"device: {device}")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("/home/y50063564/Qwen3-4B")
ids = tok(SAMPLE, return_tensors="pt")["input_ids"].to(device)
seq = ids.shape[1]
tokens = tok.convert_ids_to_tokens(ids[0].tolist())
tokyo_idx = [i for i, t in enumerate(tokens) if "Tokyo" in t]  # 兼容 "ĠTokyo" 等子词
print(f"样本 tokens: {seq}, 'Tokyo' 位置: {tokyo_idx}")

verifier = (
    AutoModelForCausalLM.from_pretrained("/home/y50063564/Qwen3-4B", torch_dtype=torch.bfloat16)
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
last_block_start = (max_anchors - 1) * block
print(f"q_len={q_len}, 末尾 block query=[{last_block_start},{q_len})")

for li, layer in enumerate(model.layers):
    attn = getattr(layer.self_attn, "_debug_attn", None)
    if attn is None:
        print(f"[layer {li}] 无注意力权重")
        continue
    a = attn[0]
    nh = a.shape[0]
    sel = a[:, last_block_start:q_len, :seq].abs()
    w = sel.amax(dim=0).mean(dim=0)
    top = torch.argsort(w, descending=True)[:10]
    hits = [p for p in top.tolist() if p in tokyo_idx]
    lastq = q_len - 1
    argmax_h = a[:, lastq, :seq].abs().argmax(dim=-1).tolist()
    tokyo_h = sum(1 for p in argmax_h if p in tokyo_idx)
    print(f"===== layer {li} ({type(layer.self_attn).__name__}) heads={nh} =====")
    print("  末尾block top-10: " + ", ".join(f"{p}:{tokens[p]}({w[p]:.3f})" for p in top.tolist()))
    print(f"  Tokyo命中={hits} | 最后query各head峰值位置={argmax_h} | Tokyo命中head={tokyo_h}/{nh}")

print("done")
