"""分析 diff 层退化情况 (l0_5l checkpoint):
1. 训练的 lambda 值 (lambda_full = exp(sum(q1*k1)) - exp(sum(q2*k2)) + lambda_init)
2. 最后 query 每个 pair 的 branch0/branch1 注意力落在 base vs block 区间的比例
3. 逐层对答案 token 的注意力 (看是否如 baseline 一样 layer4 才大)
用法: ASCEND_RT_VISIBLE_DEVICES=11 python scripts/analyze_diff_degen.py
"""
import sys

import torch

sys.path.insert(0, "/home/y50063564/dspark_project/speculators/src")

SAMPLE = ("The capital of France is Paris. The Eiffel Tower is a famous landmark visited by millions. "
          "Many tourists enjoy strolling along the Seine river during summer evenings. The weather is "
          "often mild and pleasant. Local cafes line the streets and serve fresh bread. Museums open "
          "early and close late. The capital of France is")
ANSWER = "Paris"
CHECKPOINT = "/home/y50063564/processed_data/dspark_data/dspark_diff_l0_5l/checkpoints/checkpoint_best"
TARGET_LAYERS = [1, 9, 17, 25, 33]
VERIFIER_LAST = 36

torch.manual_seed(0)
import torch_npu  # noqa: F401
device = "npu:0"

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("/home/y50063564/Qwen3-4B")
ids = tok(SAMPLE, return_tensors="pt")["input_ids"].to(device)
seq = ids.shape[1]
tokens = tok.convert_ids_to_tokens(ids[0].tolist())
ans_idx = [i for i, t in enumerate(tokens) if ANSWER in t]
print(f"seq={seq}, 答案 '{ANSWER}' 位置: {ans_idx}, token: {[tokens[i] for i in ans_idx]}")

verifier = (
    AutoModelForCausalLM.from_pretrained("/home/y50063564/Qwen3-4B", dtype=torch.bfloat16).to(device).eval()
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
for l in model.layers:
    l.self_attn.capture_attention = True

loss_mask = torch.ones(1, seq, device=device)
doc_ids = torch.zeros(1, seq, dtype=torch.long, device=device)
max_anchors = max(1, seq // 8)
with torch.no_grad():
    model(hidden_states=aux, input_ids=ids, loss_mask=loss_mask,
          verifier_last_hidden_states=verifier_last, document_ids=doc_ids, max_anchors=max_anchors)
block = model.block_size
q_len = max_anchors * block
lb = (max_anchors - 1) * block

# 1) lambda 值
da = model.layers[0].self_attn
l1 = torch.exp(torch.sum(da.lambda_q1 * da.lambda_k1)).item()
l2 = torch.exp(torch.sum(da.lambda_q2 * da.lambda_k2)).item()
print(f"\n=== lambda (diff layer0) ===")
print(f"  exp(sum(q1k1))={l1:.3f}  exp(sum(q2k2))={l2:.3f}  lambda_init={da.lambda_init:.3f}")
print(f"  lambda_full = {l1:.3f} - {l2:.3f} + {da.lambda_init:.3f} = {l1-l2+da.lambda_init:.3f}")

# 2) branch 落在 base vs block 的比例 (最后 query)
q = q_len - 1
b0 = da._debug_branch0[0].float()  # [16, q_len, kv]
b1 = da._debug_branch1[0].float()
sub = da._debug_attn[0].float()
# block 区间: 最后 anchor 的 mask token 位置 (KV 里 seq 之后)
kv_len = seq + q_len
block_start = seq + lb
block_end = seq + q_len
print(f"\n=== 最后 query q{q}: 每 pair 的 b0/b1 注意力在 base vs 自己block 的占比 ===")
print("pair | b0base% b0block% | b1base% b1block% | 模式")
for p in range(16):
    b0b = b0[p, q, :seq].sum().item(); b0k = b0[p, q, block_start:block_end].sum().item()
    b1b = b1[p, q, :seq].sum().item(); b1k = b1[p, q, block_start:block_end].sum().item()
    mode = ("both-base" if b0b>0.1 and b1b>0.1 else
            "b0->base,b1->block" if b0b>0.1 and b1k>0.3 else
            "b1->base,b0->block" if b1b>0.1 and b0k>0.3 else "other")
    print(f"  {p:2d} | {b0b*100:5.1f}% {b0k*100:5.1f}% | {b1b*100:5.1f}% {b1k*100:5.1f}% | {mode}")

# 3) 逐层对答案 token 的注意力 (整个最后block queries 平均)
print(f"\n=== 逐层对答案 token({ans_idx}) 的注意力 (最后block queries 平均, 归一化到 base 注意力总和) ===")
for li, layer in enumerate(model.layers):
    attn = getattr(layer.self_attn, "_debug_attn", None)
    if attn is None:
        print(f"  layer{li}: 无权重"); continue
    a = attn[0, :, lb:q_len, :seq].float()  # [heads, 8, seq]
    # 答案 token 的注意力占总 base 注意力的比例
    ans_attn = a[:, :, ans_idx].sum().item()
    total_base = a.sum().item()
    # 每 query 每 head 归一化后对答案的占比
    # 平均 head
    print(f"  layer{li}: 答案token总注意力={ans_attn:.3f}, 占base总量={ans_attn/(total_base+1e-9)*100:.1f}%")
print("\ndone")
