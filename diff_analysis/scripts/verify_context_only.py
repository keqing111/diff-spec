"""验证 diff_attention_context_only: diff 层不能关注自己的合成 block(mask 区)。
随机权重即可(验证的是 mask 行为)。用法: ASCEND_RT_VISIBLE_DEVICES=11 python scripts/verify_context_only.py
"""
import sys

import torch

sys.path.insert(0, "/home/y50063564/dspark_project/speculators/src")
torch.manual_seed(0)
import torch_npu  # noqa: F401
device = "npu:0"

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("/home/y50063564/Qwen3-4B")
text = ("The capital of France is Paris. The Eiffel Tower is a famous landmark. "
        "Many tourists visit every year. The capital of France is")
ids = tok(text, return_tensors="pt")["input_ids"][:, :120].to(device)
seq = ids.shape[1]
verifier = AutoModelForCausalLM.from_pretrained("/home/y50063564/Qwen3-4B", dtype=torch.bfloat16).to(device).eval()
with torch.no_grad():
    out = verifier(ids, output_hidden_states=True)
hs = out.hidden_states
aux = torch.cat([hs[l] for l in [1, 9, 17, 25, 33]], dim=-1)
vlast = hs[36]

from safetensors.torch import load_file  # noqa: E402
from speculators.models.dspark import DSparkDraftModel  # noqa: E402
from speculators.models.dspark.config import DSparkSpeculatorConfig  # noqa: E402

# 构建 context-only 模型(mask 行为与权重无关, 加载 checkpoint 权重仅为了前向正常)
cfg = DSparkSpeculatorConfig.from_pretrained(
    "/home/y50063564/processed_data/dspark_data/dspark_diff_l0_5l/checkpoints/checkpoint_best")
cfg.transformer_layer_config._attn_implementation = "eager"
cfg.diff_attention_context_only = True
cfg.diff_attention_layer_indices = [0]
model = DSparkDraftModel(cfg)
model.load_state_dict(load_file(
    "/home/y50063564/processed_data/dspark_data/dspark_diff_l0_5l/checkpoints/checkpoint_best/model.safetensors"), strict=False)
model.load_verifier_weights()
model.to(device).eval()
model.layers[0].self_attn.capture_attention = True

loss_mask = torch.ones(1, seq, device=device)
doc_ids = torch.zeros(1, seq, dtype=torch.long, device=device)
max_anchors = max(1, seq // 8)
with torch.no_grad():
    model(hidden_states=aux, input_ids=ids, loss_mask=loss_mask,
          verifier_last_hidden_states=vlast, document_ids=doc_ids, max_anchors=max_anchors)
b0 = model.layers[0].self_attn._debug_branch0[0].float()
b1 = model.layers[0].self_attn._debug_branch1[0].float()
mask_attn_b0 = b0[:, :, seq:].sum().item()  # 全部 query 对 mask 区注意力
mask_attn_b1 = b1[:, :, seq:].sum().item()
total_b0 = b0.sum().item()
print(f"seq={seq}, q_len={b0.shape[1]}")
print(f"branch0 对 mask 区(合成block)注意力总和 = {mask_attn_b0:.6f} (占全部 {mask_attn_b0/total_b0*100:.3f}%)")
print(f"branch1 对 mask 区注意力总和 = {mask_attn_b1:.6f}")
print("=> context_only 生效" if mask_attn_b0 < 1e-3 and mask_attn_b1 < 1e-3 else "=> context_only 未生效!")
