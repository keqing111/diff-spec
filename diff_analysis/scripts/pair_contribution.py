"""计算 diff 层(layer0)每个 pair 在选定 query 上的贡献。

贡献定义:
  raw_norm     = || (branch0 - lambda*branch1) @ V ||  (subln 前, 反映差分活动的强度)
  oproj_norm   = || o_proj[:, p*256:(p+1)*256] @ ((1-lambda_init)*subln(pair_out)) ||
                 (该 pair 经 o_proj 后对最终注意层输出的实际贡献)
用法: ASCEND_RT_VISIBLE_DEVICES=11 python scripts/pair_contribution.py
"""
import sys

import torch

sys.path.insert(0, "/home/y50063564/dspark_project/speculators/src")

SAMPLES = [
    "The capital of France is Paris. The Eiffel Tower is a famous landmark visited by millions. "
    "Many tourists enjoy strolling along the Seine river during summer evenings. The weather is "
    "often mild and pleasant. Local cafes line the streets and serve fresh bread. Museums open "
    "early and close late. The capital of France is",
    "Water boils at one hundred degrees Celsius at standard sea level pressure. Heat energy "
    "causes the liquid to evaporate quickly. Scientists use thermometers to measure temperature "
    "accurately in experiments. In daily life boiling water is common for cooking. Water boils at one hundred",
    "Elephants are the largest land animals on Earth. They live in herds and communicate with "
    "low rumbling sounds over long distances. Conservationists work hard to protect their habitats. "
    "Visitors often see them at wildlife reserves in Africa. Elephants are the largest",
]
CHECKPOINT = "/home/y50063564/processed_data/dspark_data/dspark_diff_l0_5l/checkpoints/checkpoint_best"
TARGET_LAYERS = [1, 9, 17, 25, 33]
VERIFIER_LAST = 36

torch.manual_seed(0)
import torch_npu  # noqa: F401
device = "npu:0"

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("/home/y50063564/Qwen3-4B")
verifier = (
    AutoModelForCausalLM.from_pretrained("/home/y50063564/Qwen3-4B", dtype=torch.bfloat16)
    .to(device)
    .eval()
)
from safetensors.torch import load_file  # noqa: E402
from speculators.models.dspark import DSparkDraftModel  # noqa: E402
from speculators.models.dspark.config import DSparkSpeculatorConfig  # noqa: E402

config = DSparkSpeculatorConfig.from_pretrained(CHECKPOINT)
config.transformer_layer_config._attn_implementation = "eager"
model = DSparkDraftModel(config)
model.load_state_dict(load_file(f"{CHECKPOINT}/model.safetensors"), strict=False)
model.load_verifier_weights()
model.to(device).eval()
attn_mod = model.layers[0].self_attn
attn_mod.capture_attention = True

hd2 = 2 * attn_mod.head_dim  # 每 pair 输出维度 2*head_dim
oW = attn_mod.o_proj.weight  # [2560, 4096]
lam_init = attn_mod.lambda_init
dtype = torch.float32

for si, sample in enumerate(SAMPLES):
    ids = tok(sample, return_tensors="pt")["input_ids"].to(device)
    seq = ids.shape[1]
    with torch.no_grad():
        out = verifier(ids, output_hidden_states=True)
    hs = out.hidden_states
    aux = torch.cat([hs[l] for l in TARGET_LAYERS], dim=-1)
    verifier_last = hs[VERIFIER_LAST]
    loss_mask = torch.ones(1, seq, device=device)
    doc_ids = torch.zeros(1, seq, dtype=torch.long, device=device)
    max_anchors = max(1, seq // 8)
    with torch.no_grad():
        model(hidden_states=aux, input_ids=ids, loss_mask=loss_mask,
              verifier_last_hidden_states=verifier_last, document_ids=doc_ids,
              max_anchors=max_anchors)
    raw = attn_mod._debug_pair_out_raw[0].float()  # [16, q_len, 256] pre-subln
    normed = attn_mod._debug_pair_out_normed[0].float()  # [16, q_len, 256] post-subln
    block = model.block_size
    q_len = raw.shape[1]
    lb = (max_anchors - 1) * block
    chosen = sorted({lb, lb + 1, lb + max(1, block // 2), q_len - 1})
    print(f"\n##### 样本{si+1}: seq={seq} #####")
    for q in chosen:
        # raw 活动范数
        raw_norms = raw[:, q, :].norm(dim=-1).cpu()  # [16]
        # o_proj 贡献: 对每个 pair, oW[:, p*hd2:(p+1)*hd2] @ ((1-lam_init)*normed[p,q])
        contrib = torch.zeros(16)
        for p in range(16):
            v = (1 - lam_init) * normed[p, q]  # [256]
            contrib[p] = (oW[:, p * hd2:(p + 1) * hd2] @ v).float().norm().item()
        total = contrib.sum().item()
        print(f" q{q}(block#{q-lb}):")
        print("   raw_norm[每pair]: " + " ".join(f"{raw_norms[p]:.3f}" for p in range(16)))
        print("   oproj_contrib   : " + " ".join(f"{contrib[p]:.3f}" for p in range(16)))
        top = torch.argsort(contrib, descending=True)
        pct = contrib / (total + 1e-9) * 100
        print(f"   前3贡献pair={top[:3].tolist()} 占比={pct[top[0]]:.1f}%/{pct[top[1]]:.1f}%/{pct[top[2]]:.1f}%")
print("\ndone")
