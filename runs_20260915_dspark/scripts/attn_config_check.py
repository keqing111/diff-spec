#!/usr/bin/env python3
"""Offline check for the draft-attention comparison (MHA / native GQA / MLA).

Builds the exact `transformer_layer_config` that `scripts/train.py` produces for
each arm, then instantiates the attention module that `core.py` would select and
counts its parameters. No NPU / no checkpoint needed.

Run:
    python3 attn_config_check.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/home/y50063564/dspark_project/speculators")

from scripts.train import create_transformer_layer_config  # noqa: E402
from speculators.models.dflash.model_definitions import (  # noqa: E402
    Qwen3DFlashAttention,
    Qwen3DFlashMLAttention,
)

VERIFIER = "/home/y50063564/Qwen3-4B"
NUM_LAYERS = 5
FULL_ATTN = [0, 1, 2, 3, 4]

ARMS = [
    ("MHA       ", {"attention_type": "mha"}),
    ("GQA(原生) ", {"attention_type": "gqa"}),
    ("MLA r=512 ", {"attention_type": "mla"}),
]


def count(mod: torch.nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters())


def kv_key(mod: torch.nn.Module) -> str:
    """Names + shapes of the modules that produce K and V."""
    out = []
    for name, sub in mod.named_children():
        if name.startswith(("k_", "v_", "kv_")):
            out.append(f"{name}{tuple(sub.weight.shape)}")
    return "  ".join(out)


def main() -> int:
    print("=" * 96)
    print("draft attention variants -- config geometry (from create_transformer_layer_config)")
    print("=" * 96)
    header = (
        f"{'arm':11s} {'Q heads':>7s} {'KV heads':>8s} {'head_dim':>8s} "
        f"{'layer_types[0]':>15s} {'use_sliding':>11s} {'window':>7s} "
        f"{'attn_type':>9s} {'latent':>6s}"
    )
    print(header)
    print("-" * len(header))

    configs = {}
    for label, kw in ARMS:
        cfg = create_transformer_layer_config(            verifier_name_or_path=VERIFIER,
            num_layers=NUM_LAYERS,
            draft_arch="qwen3",
            hidden_act=None,
            sliding_window=2048,
            full_attention_indices=FULL_ATTN,
            attention_type=kw["attention_type"],
            mla_kv_lora_rank=512,
        )
        # train.py sets this from --draft-attn-impl; the scripts pass "sdpa", but
        # keep the standalone smoke on "eager" so attn_weights come back.
        cfg._attn_implementation = "eager"  # noqa: SLF001
        configs[label] = cfg
        print(
            f"{label:11s} {cfg.num_attention_heads:7d} {cfg.num_key_value_heads:8d} "
            f"{cfg.head_dim:8d} {cfg.layer_types[0]:>15s} "
            f"{str(cfg.use_sliding_window):>11s} {str(cfg.sliding_window):>7s} "
            f"{cfg.attention_type:>9s} {getattr(cfg, 'mla_kv_lora_rank', 0):6d}"
        )

    print()
    print("=" * 96)
    print("attention module actually selected per arm (layer 0)")
    print("=" * 96)
    params_all = {}
    for label, kw in ARMS:
        cfg = configs[label]
        cls = Qwen3DFlashMLAttention if cfg.attention_type == "mla" else Qwen3DFlashAttention
        attn = cls(cfg, layer_idx=0)
        n = count(attn)
        params_all[label] = n
        print(f"\n[{label}] class = {cls.__name__}   params = {n:,}")
        print(f"    K/V producing modules: {kv_key(attn)}")
        print(f"    q_norm present       : {hasattr(attn, 'q_norm')}")
        print(f"    k_norm present       : {hasattr(attn, 'k_norm')}")
        print(f"    sliding_window       : {attn.sliding_window}")
        print(f"    num_key_value_groups : {attn.num_key_value_groups}")

        # forward smoke: target_hidden injected as the context prefix
        bsz, q_len, hidden, ctx = 2, 8, cfg.hidden_size, 16
        hs = torch.randn(bsz, q_len, hidden)
        th = torch.randn(bsz, ctx, hidden)
        # Qwen3RotaryEmbedding emits (bsz, seq, head_dim); apply_rotary_pos_emb
        # unsqueezes dim=1 itself.
        cos = torch.randn(1, ctx + q_len, cfg.head_dim)
        sin = torch.randn(1, ctx + q_len, cfg.head_dim)
        assert cfg._attn_implementation == "eager", cfg._attn_implementation  # noqa: SLF001
        out, _ = attn(
            hidden_states=hs,
            target_hidden=th,
            position_embeddings=(cos, sin),
            attention_mask=None,
        )
        print(f"    forward out          : {tuple(out.shape)}  (expect {(bsz, q_len, hidden)})")
        assert out.shape == (bsz, q_len, hidden)

        # backward smoke
        out.sum().backward()
        grads = [p.grad is not None for p in attn.parameters()]
        print(f"    backward             : all params got grad = {all(grads)}")

    print()
    print("=" * 96)
    print("summary (per attention layer, params)")
    print("=" * 96)
    base = params_all["GQA(原生) "]
    for label, _ in ARMS:
        n = params_all[label]
        delta = n - base
        print(f"  {label}  {n:>10,}   Δ vs GQA: {delta:+,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
