"""对多个样本、block 内几个 query 位置, 绘制 diff 层(layer0)逐 pair 的
branch0/branch1/相减 注意力曲线(风格同 layer0_diff_pairs_lastquery.png)。

用法: ASCEND_RT_VISIBLE_DEVICES=11 python scripts/plot_attn_few_tokens.py
输出: <outdir>/samp{N}_q{query}.png  (每个样本 x 每个选定 query 一张 16-pair 图)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

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
OUTDIR = Path("/home/y50063564/processed_data/dspark_data/attn_plots/few_tokens")
OUTDIR.mkdir(parents=True, exist_ok=True)

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
    tokens = tok.convert_ids_to_tokens(ids[0].tolist())
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
    sub = diff_attn._debug_attn[0].float()
    block = model.block_size
    q_len = b0.shape[1]
    lb = (max_anchors - 1) * block
    # 选定几个 query: block 首(anchor), 中段, 末段
    chosen = sorted({lb, lb + 1, lb + max(1, block // 2), q_len - 1})
    for q in chosen:
        fig, axes = plt.subplots(4, 4, figsize=(22, 14))
        for p in range(16):
            ax = axes.flat[p]
            ax.plot(b0[p, q, :seq].cpu().numpy(), color="tab:blue", lw=0.7, label="b0")
            ax.plot(b1[p, q, :seq].cpu().numpy(), color="tab:orange", lw=0.7, label="b1")
            ax.plot(sub[p, q, :seq].cpu().numpy(), color="tab:green", lw=0.7, label="diff")
            ax.axvline(seq - 1, color="gray", ls=":", lw=0.5)
            ax.set_title(f"pair{p}", fontsize=8)
            ax.set_xticks(range(0, seq, 5))
            ax.set_xticklabels([tokens[i] for i in range(0, seq, 5)], rotation=90, fontsize=4)
            ax.set_yticks([])
        axes.flat[0].legend(fontsize=6, ncol=3)
        fig.suptitle(f"样本{si+1} diff-layer0 query q{q} (block内第{q-lb}位): branch0/branch1/diff over base tokens", fontsize=10)
        fig.tight_layout()
        out = OUTDIR / f"samp{si+1}_q{q}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"saved {out}  (block内第{q-lb}位)")

print("done")
