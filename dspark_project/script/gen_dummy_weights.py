"""Generate dummy verifier weights for the fake (3~5-layer) model.

Writes model.safetensors next to the config in fakemodelconfig so the
training side has target-model weights to load.
"""
import json

import torch
from safetensors.torch import save_file

cfg = json.load(open("/home/y50063564/dspark_project/fakemodelconfig/config.json"))
V, H = cfg["vocab_size"], cfg["hidden_size"]
torch.manual_seed(0)
t = {
    "model.embed_tokens.weight": torch.randn(V, H, dtype=torch.bfloat16),  # [vocab, hidden]
    "lm_head.weight": torch.randn(V, H, dtype=torch.bfloat16),
    "model.norm.weight": torch.ones(H, dtype=torch.bfloat16),
}
save_file(t, "/home/y50063564/dspark_project/fakemodelconfig/model.safetensors")
print("wrote dummy verifier weights:", {k: tuple(v.shape) for k, v in t.items()})
