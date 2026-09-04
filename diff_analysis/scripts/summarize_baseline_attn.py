"""baseline 逐层逐头注意力摘要: 末尾 query q63 上各 head 对 Tokyo 的关注。"""
import torch
from collections import Counter

d = torch.load("/home/y50063564/processed_data/dspark_data/attn_plots/baseline/baseline_attn.pt",
               map_location="cpu")
tokens = ["The", "capital", "city", "of", "Japan", "is", "Tokyo", ".", "The", "weather",
          "today", "is", "pleasant", "and", "the", "sky", "is", "clear", "across", "the",
          "whole", "region", ".", "Many", "people", "enjoy", "taking", "a", "quiet", "walk",
          "in", "the", "park", "during", "the", "afternoon", ".", "Reading", "books", "is",
          "a", "popular", "way", "to", "spend", "a", "peaceful", "evening", "at", "home",
          ".", "The", "local", "library", "opens", "early", "in", "the", "morning", "and",
          "stays", "open", "until", "late", ".", "The", "capital", "city", "of", "Japan", "is"]
tokyo = [i for i, t in enumerate(tokens) if "Tokyo" in t]
print(f"Tokyo 位置: {tokyo}")
for li in range(5):
    a = d[f"layer{li}"][0, :, 63, :71].float()  # [heads, seq]
    argmax = a.argmax(dim=-1)
    hit = int((argmax == tokyo[0]).sum())
    share = a[:, tokyo[0]] / a.sum(dim=-1)
    top = Counter(argmax.tolist()).most_common(5)
    print(f"layer{li}: top1=Tokyo 的 head={hit}/32 | 平均对Tokyo占比={share.mean().item()*100:.1f}% (最大={share.max().item()*100:.1f}%) | top1分布={top}")
