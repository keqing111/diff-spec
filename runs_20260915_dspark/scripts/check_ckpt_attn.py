#!/usr/bin/env python3
"""Inspect a saved DSpark checkpoint and report what attention it actually built.

Reads `config.json` (to confirm the flags round-tripped through the speculator
config) and the attention tensors in the weights (to confirm the right module
was constructed and actually has the expected shapes).

Run:
    python3 check_ckpt_attn.py <ckpt_dir> [...]
"""

from __future__ import annotations

import glob
import json
import os
import sys

from safetensors import safe_open

# Parameter-name prefixes that tell the three variants apart.
KV_MARKERS = ("k_proj", "v_proj", "kv_down_proj", "k_up_proj", "v_up_proj")


def describe(ckpt_dir: str) -> int:
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.isfile(cfg_path):
        print(f"!! no config.json under {ckpt_dir}")
        return 1
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    tl = cfg.get("transformer_layer_config", {})

    print(f"\n=== {ckpt_dir} ===")
    print(
        "  attention_type        = {!r}   (absent => 'gqa' default)".format(
            tl.get("attention_type", "<absent>")
        )
    )
    print(f"  mla_kv_lora_rank      = {tl.get('mla_kv_lora_rank', '<absent>')}")
    print(f"  num_attention_heads   = {tl.get('num_attention_heads')}")
    print(f"  num_key_value_heads   = {tl.get('num_key_value_heads')}")
    print(f"  head_dim              = {tl.get('head_dim')}")
    print(f"  hidden_size           = {tl.get('hidden_size')}")
    print(f"  layer_types           = {tl.get('layer_types')}")
    print(f"  use_sliding_window    = {tl.get('use_sliding_window')}")
    print(f"  sliding_window        = {tl.get('sliding_window')}")

    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if not files:
        print("  !! no *.safetensors found")
        return 1

    total = 0
    layer0: dict[str, tuple[int, ...]] = {}
    for path in files:
        with safe_open(path, framework="pt") as fh:
            for key in fh.keys():
                shape = tuple(fh.get_slice(key).get_shape())
                n = 1
                for d in shape:
                    n *= d
                total += n
                if key.startswith("layers.0.") and any(m in key for m in KV_MARKERS):
                    layer0[key.split(".", 2)[-1]] = shape
                if key.startswith("layers.0.") and key.endswith(("q_norm.weight", "k_norm.weight")):
                    layer0[key.split(".", 2)[-1]] = shape

    print(f"  total params          = {total:,}")
    print("  layer0 attention K/V + qk_norm tensors:")
    for name in sorted(layer0):
        print(f"      {name:45s} {layer0[name]}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    rc = 0
    for ckpt in argv[1:]:
        rc |= describe(ckpt)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
