"""多样本验证: diff 层(layer0)是否普遍存在"一个分支注意力落在合成 block"的退化。

从训练数据取 N 个真实样本, 每个跑 verifier+DSpark, 对每个 (pair, query) 统计
branch0/branch1 注意力在 base(上下文) vs mask(合成 block) 区间的占比, 分类:
  both_base : 两个分支都在 base 上有实质注意力(>0.1)
  one_to_mask: 一个分支看 base, 另一个 ~不看 base(主要落 mask 区)
  both_mask : 两个分支都不看 base
汇总全样本比例。
用法: ASCEND_RT_VISIBLE_DEVICES=11 python scripts/check_branch_block_multisample.py [N]
"""
import sys

import torch

sys.path.insert(0, "/home/y50063564/dspark_project/speculators/src")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MAX_TOK = 250  # 每样本截断长度

torch.manual_seed(0)
import torch_npu  # noqa: F401
device = "npu:0"

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("/home/y50063564/Qwen3-4B")

# ---- 从训练数据解码样本 ----
import pyarrow.ipc as ipc  # noqa: E402
texts = []
for fi in range(5):
    f = f"/home/y50063564/data/open_perfectblend_qwen3_4b_700k/data-0000{fi}-of-00005.arrow"
    try:
        with ipc.open_stream(f) as r:
            for batch in r:
                ids = batch.column("input_ids").to_pylist()
                for row in ids:
                    txt = tok.decode(row, skip_special_tokens=True).strip()
                    if 80 < len(txt) < 1500:
                        texts.append(txt)
                    if len(texts) >= N:
                        break
                if len(texts) >= N:
                    break
    except Exception as e:
        print(f"skip {f}: {e}")
    if len(texts) >= N:
        break
texts = texts[:N]
print(f"取到 {len(texts)} 个样本, 平均长度 {sum(len(t) for t in texts)//max(1,len(texts))} 字符")

# ---- 加载模型 ----
verifier = (
    AutoModelForCausalLM.from_pretrained("/home/y50063564/Qwen3-4B", dtype=torch.bfloat16).to(device).eval()
)
from safetensors.torch import load_file  # noqa: E402
from speculators.models.dspark import DSparkDraftModel  # noqa: E402
from speculators.models.dspark.config import DSparkSpeculatorConfig  # noqa: E402
CHECKPOINT = "/home/y50063564/processed_data/dspark_data/dspark_diff_l0_5l/checkpoints/checkpoint_best"
config = DSparkSpeculatorConfig.from_pretrained(CHECKPOINT)
config.transformer_layer_config._attn_implementation = "eager"
model = DSparkDraftModel(config)
model.load_state_dict(load_file(f"{CHECKPOINT}/model.safetensors"), strict=False)
model.load_verifier_weights()
model.to(device).eval()
da = model.layers[0].self_attn
da.capture_attention = True

# ---- 逐样本统计 ----
stats = {"both_base": 0, "one_to_mask": 0, "both_mask": 0, "total": 0}
per_pair = {p: {"both": 0, "one_mask": 0, "both_mask": 0, "total": 0} for p in range(16)}

for si, text in enumerate(texts):
    ids = tok(text, return_tensors="pt")["input_ids"][:, :MAX_TOK].to(device)
    seq = ids.shape[1]
    if seq < 32:
        continue
    with torch.no_grad():
        out = verifier(ids, output_hidden_states=True)
    hs = out.hidden_states
    aux = torch.cat([hs[l] for l in [1, 9, 17, 25, 33]], dim=-1)
    vlast = hs[36]
    loss_mask = torch.ones(1, seq, device=device)
    doc_ids = torch.zeros(1, seq, dtype=torch.long, device=device)
    max_anchors = max(1, seq // 8)
    with torch.no_grad():
        model(hidden_states=aux, input_ids=ids, loss_mask=loss_mask,
              verifier_last_hidden_states=vlast, document_ids=doc_ids, max_anchors=max_anchors)
    b0 = da._debug_branch0[0].float()  # [16, q_len, kv]
    b1 = da._debug_branch1[0].float()
    q_len = b0.shape[1]
    # 对所有 query (anchor block 位置) 统计
    for q in range(q_len):
        for p in range(16):
            s0 = b0[p, q, :seq].sum().item()
            s1 = b1[p, q, :seq].sum().item()
            m0 = b0[p, q, seq:].sum().item()
            m1 = b1[p, q, seq:].sum().item()
            if s0 > 0.1 and s1 > 0.1:
                mode = "both_base"
            elif (s0 > 0.1 and m1 > 0.3) or (s1 > 0.1 and m0 > 0.3):
                mode = "one_to_mask"
            else:
                mode = "both_mask"
            stats[mode] += 1
            stats["total"] += 1
            per_pair[p][{"both_base": "both", "one_to_mask": "one_mask", "both_mask": "both_mask"}[mode]] += 1
            per_pair[p]["total"] += 1
    if (si + 1) % 10 == 0:
        print(f"  ... {si+1}/{len(texts)} 样本, 当前 both_base 占比={stats['both_base']/stats['total']*100:.1f}%")

print(f"\n===== {stats['total']} 个 (query,pair) 实例的分布 =====")
t = stats["total"]
print(f"both_base (两分支都在上下文): {stats['both_base']/t*100:.1f}%")
print(f"one_to_mask (一个分支~不看上下文): {stats['one_to_mask']/t*100:.1f}%")
print(f"both_mask (两分支都不看上下文): {stats['both_mask']/t*100:.1f}%")
print(f"\n===== 按 pair 统计 =====")
print("pair | both_base% | one_to_mask% | both_mask%")
for p in range(16):
    pt = per_pair[p]["total"]
    print(f"  {p:2d} | {per_pair[p]['both']/pt*100:7.1f}% | {per_pair[p]['one_mask']/pt*100:9.1f}% | {per_pair[p]['both_mask']/pt*100:7.1f}%")
print("\ndone")
