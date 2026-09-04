"""检查 baseline 各层注意力是否分配到 block/mask 区(seq..kv_len) vs base 区(0..seq)。
对最后一个 anchor block 的所有 query + 全部 query 分别统计。"""
import torch

d = torch.load("/home/y50063564/processed_data/dspark_data/attn_plots/baseline/baseline_attn.pt",
               map_location="cpu")
kv = d["layer0"].shape[-1]
q_len = d["layer0"].shape[2]
# base 长度: 通过 token 序列已知 71; 更稳的是从 attn 反推? 用 71
seq = 71
print(f"q_len={q_len}, kv_len={kv}, base=0..{seq-1}, mask/block={seq}..{kv-1}")

for li in range(5):
    a = d[f"layer{li}"][0].float()  # [heads, q_len, kv_len]
    # 最后一个 block 的 query
    lb = q_len - 8
    blk_q = a[:, lb:q_len, :]  # [heads, 8, kv]
    # 每个 head 在最后block query 上: base vs block 注意力占比
    base_share = blk_q[:, :, :seq].sum(dim=(1, 2)) / blk_q.sum(dim=(1, 2))
    blk_share = blk_q[:, :, seq:].sum(dim=(1, 2)) / blk_q.sum(dim=(1, 2))
    print(f"layer{li} [最后block query]: base占比均值={base_share.mean().item()*100:.1f}% "
          f"(范围 {base_share.min().item()*100:.0f}-{base_share.max().item()*100:.0f}%), "
          f"block占比均值={blk_share.mean().item()*100:.1f}%")

# 全部 query 的视角(每层 base vs block)
print("\n[全部 query]")
for li in range(5):
    a = d[f"layer{li}"][0].float()
    tot = a.sum(dim=(1, 2))  # 每 head 总注意力
    base_share = a[:, :, :seq].sum(dim=(1, 2)) / tot
    blk_share = a[:, :, seq:].sum(dim=(1, 2)) / tot
    print(f"layer{li}: base均值={base_share.mean().item()*100:.1f}% block均值={blk_share.mean().item()*100:.1f}% "
          f"| 各head block占比范围 {blk_share.min().item()*100:.0f}-{blk_share.max().item()*100:.0f}%")
