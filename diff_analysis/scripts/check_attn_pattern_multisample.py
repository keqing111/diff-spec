"""多样本检查 diff 层(layer0)注意力模式: 是否普遍存在"单分支活跃/退化为单头"。

对每个样本: 跑 forward, 对最后 anchor block 的每个 query(8个), 统计 16 个 pair
中每个 pair 的 branch0/branch1 在 base token 上的注意力总和, 判定模式:
  BOTH : b0 和 b1 都在 base 上有实质注意力(>0.1)
  only0: 只有 branch0 活跃
  only1: 只有 branch1 活跃
  ~0   : 两个分支在 base 上都几乎为 0
用法: ASCEND_RT_VISIBLE_DEVICES=11 python scripts/check_attn_pattern_multisample.py
"""
import sys

import torch

sys.path.insert(0, "/home/y50063564/dspark_project/speculators/src")

SAMPLES = [
    "The capital city of Japan is Tokyo. The weather today is pleasant and the sky is clear "
    "across the whole region. Many people enjoy taking a quiet walk in the park during the "
    "afternoon. Reading books is a popular way to spend a peaceful evening at home. The "
    "local library opens early in the morning and stays open until late. The capital city of Japan is",
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
THR = 0.1  # 判定"在 base 上活跃"的注意力总和阈值

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
diff_attn = model.layers[0].self_attn
diff_attn.capture_attention = True

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
    b0 = diff_attn._debug_branch0[0].float()  # [16, q_len, kv_len]
    b1 = diff_attn._debug_branch1[0].float()
    block = model.block_size
    q_len = b0.shape[1]
    lb = (max_anchors - 1) * block

    print(f"\n########## 样本{si+1}: seq={seq}, q_len={q_len}, 最后block query=[{lb},{q_len}) ##########")
    print("(每格: pair 在该 query 的模式, B=both活跃 b=only-branch0 1=only-branch1 0=双~0)")
    print("query\\pair |" + "".join(f"{p:>3d}" for p in range(16)))
    for qi in range(lb, q_len):
        row = []
        for p in range(16):
            s0 = b0[p, qi, :seq].sum().item()
            s1 = b1[p, qi, :seq].sum().item()
            if s0 > THR and s1 > THR:
                row.append("  B")
            elif s0 > THR:
                row.append("  b")
            elif s1 > THR:
                row.append("  1")
            else:
                row.append("  0")
        print(f"q{qi:>3d}      |" + "".join(row))

    # 汇总: 每个 query 各模式的 pair 数
    print("汇总(每个query: BOTH数/only0/only1/~0):")
    for qi in range(lb, q_len):
        cnt = {"B": 0, "b": 0, "1": 0, "0": 0}
        for p in range(16):
            s0 = b0[p, qi, :seq].sum().item()
            s1 = b1[p, qi, :seq].sum().item()
            if s0 > THR and s1 > THR:
                cnt["B"] += 1
            elif s0 > THR:
                cnt["b"] += 1
            elif s1 > THR:
                cnt["1"] += 1
            else:
                cnt["0"] += 1
        print(f"  q{qi}: BOTH={cnt['B']:2d} only-branch0={cnt['b']:2d} only-branch1={cnt['1']:2d} 双~0={cnt['0']:2d}")

print("\ndone")
