"""Unit tests for the DFlash Diff-Transformer (diffv1) attention."""

import math

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.attention import create_float_mask
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.model_definitions import (
    Qwen3DFlashAttention,
    Qwen3DFlashDecoderLayer,
    Qwen3DFlashDiffAttention,
    block_mask_to_float_mask,
    lambda_init_fn,
    repeat_kv,
)

# Small even head counts so the diff pairing (//2) is exact.
_HEAD_DIM = 32


def _config(**overrides) -> Qwen3Config:
    kwargs = dict(
        vocab_size=32000,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=1024,
        head_dim=_HEAD_DIM,
        layer_types=["sliding_attention"],
        sliding_window=128,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
    )
    kwargs.update(overrides)
    return Qwen3Config(**kwargs)


def _anchor_block_mask(device, q_len=8, ctx_len=16, block_size=4, sliding_window=16):
    """A realistic anchor-block mask (with sliding window) over q x (ctx+q) kv."""
    total_seq_len = ctx_len
    n_anchors = q_len // block_size
    document_ids = torch.zeros(total_seq_len, dtype=torch.long)
    anchor_positions = torch.arange(n_anchors, dtype=torch.long) * block_size
    mask_mod, mask_q_len, mask_kv_len = create_anchor_block_mask_mod(
        document_ids=document_ids,
        total_seq_len=total_seq_len,
        anchor_positions=anchor_positions,
        block_size=block_size,
        sliding_window=sliding_window,
    )
    assert mask_q_len == q_len
    assert mask_kv_len == ctx_len + q_len
    return mask_mod, mask_q_len, mask_kv_len


def _run(attn, attention_mask=None, seed=0):
    torch.manual_seed(seed)
    bsz, q_len, ctx_len = 1, 8, 16
    hidden = torch.randn(bsz, q_len, attn.config.hidden_size)
    target = torch.randn(bsz, ctx_len, attn.config.hidden_size)
    seq_len = ctx_len + q_len
    cos = torch.cos(
        torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) * 0.01
    ).repeat(1, 1, attn.head_dim)
    sin = torch.sin(
        torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) * 0.01
    ).repeat(1, 1, attn.head_dim)
    return attn(
        hidden_states=hidden,
        target_hidden=target,
        position_embeddings=(cos, sin),
        attention_mask=attention_mask,
    )


@pytest.fixture
def attn():
    # Last layer of a 5-layer stack, like the running DSPark config.
    return Qwen3DFlashDiffAttention(
        _config(
            num_hidden_layers=5,
            layer_types=["sliding_attention"] * 5,
        ),
        layer_idx=4,
    )


def test_output_shape(attn):
    """Forward returns (bsz, q_len, hidden_size) and no attention weights."""
    out, weights = _run(attn)
    assert weights is None
    assert tuple(out.shape) == (1, 8, 256)
    assert torch.isfinite(out).all()


def test_lambda_parameters_registered(attn):
    """Lambda vectors and the per-head subln are registered, trainable params."""
    state = attn.state_dict()
    for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
        assert name in state, f"{name} missing from state_dict"
        assert tuple(state[name].shape) == (attn.head_dim,)
        assert getattr(attn, name).requires_grad
    assert "subln.weight" in state
    assert tuple(state["subln.weight"].shape) == (2 * attn.head_dim,)


def test_lambda_init_formula():
    """lambda_init(depth) = 0.8 - 0.6 * exp(-0.3 * depth), frozen scalar."""
    assert lambda_init_fn(0) == pytest.approx(0.8 - 0.6)
    assert lambda_init_fn(4) == pytest.approx(0.8 - 0.6 * math.exp(-1.2))


def test_lambda_is_trainable_and_affects_output(attn):
    """Changing a trainable lambda vector changes the attention output."""
    out1, _ = _run(attn, seed=0)
    with torch.no_grad():
        attn.lambda_q1.add_(0.5)
    out2, _ = _run(attn, seed=0)
    assert not torch.allclose(out1, out2)


def test_block_mask_to_float_mask_matches_create_float_mask():
    """Vectorized BlockMask conversion equals the eager 0/-inf float mask."""
    device = torch.device("cpu")
    mask_mod, q_len, kv_len = _anchor_block_mask(device)
    block_mask = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device
    )
    vec = block_mask_to_float_mask(block_mask, device=device, dtype=torch.float32)
    ref = create_float_mask(
        mask_mod, Q_LEN=q_len, KV_LEN=kv_len, device=device, dtype=torch.float32
    )
    assert vec.shape == ref.shape == (1, 1, q_len, kv_len)
    assert torch.equal(vec, ref)


def test_all_mask_paths_agree():
    """Diff attention produces identical output for BlockMask, bool and float mask."""
    device = torch.device("cpu")
    attn = Qwen3DFlashDiffAttention(_config(), layer_idx=0)
    mask_mod, q_len, kv_len = _anchor_block_mask(device)

    block_mask = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device
    )
    bool_mask = create_mask(
        mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device
    )
    float_mask = create_float_mask(
        mask_mod, Q_LEN=q_len, KV_LEN=kv_len, device=device, dtype=torch.float32
    )

    out_block, _ = _run(attn, attention_mask=block_mask)
    out_bool, _ = _run(attn, attention_mask=bool_mask)
    out_float, _ = _run(attn, attention_mask=float_mask)
    assert torch.allclose(out_block, out_bool, atol=1e-6)
    assert torch.allclose(out_block, out_float, atol=1e-6)


def test_k_expansion_gives_distinct_k_per_branch():
    """同一 pair 的两个分支必须用不同的 K(K1/K2), 而不是共享同一个 k。

    repeat_kv 按 i//n_rep 扁平重复会把 pair 的两个分支映射到同一个 k;
    修复后的展开保留 (kv_head, branch) 结构, 使 k[2j]->K1_{j//n_rep},
    k[2j+1]->K2_{j//n_rep}。
    """
    bsz, kv_len, head_dim = 1, 4, 8
    num_heads, num_kv_heads, n_rep = 16, 4, 4
    # orig k: 2*num_kv_heads 个子头, 布局 [K1_0,K2_0,K1_1,K2_1,...]
    k = (
        torch.arange(2 * num_kv_heads, dtype=torch.float32)
        .view(1, 2 * num_kv_heads, 1, 1)
        .expand(bsz, 2 * num_kv_heads, kv_len, head_dim)
        .contiguous()
    )

    old = repeat_kv(k, n_rep)  # (bsz, 2*num_heads, kv_len, head_dim)
    new = (
        k.reshape(bsz, num_kv_heads, 2, kv_len, head_dim)
        .repeat_interleave(n_rep, dim=1)
        .reshape(bsz, 2 * num_heads, kv_len, head_dim)
    )

    for j in range(num_heads):
        # 旧: pair j 的两个分支共享同一个 k(重复值)
        assert torch.equal(old[0, 2 * j], old[0, 2 * j + 1])
        # 新: pair j 的两个分支用不同的 K1/K2
        assert not torch.equal(new[0, 2 * j], new[0, 2 * j + 1])
    # 新: 同一 kv group 内的相邻 pair 共享同一个 kv head 的 K1/K2
    for j in range(num_heads - 1):
        if (j + 1) // n_rep == j // n_rep:  # 同一 kv group
            assert torch.equal(new[0, 2 * j], new[0, 2 * (j + 1)])
            assert torch.equal(new[0, 2 * j + 1], new[0, 2 * (j + 1) + 1])
    # 不同 kv head 之间应当不同 (pair 3 -> kv_head0, pair 4 -> kv_head1)
    assert not torch.equal(new[0, 2 * (n_rep - 1)], new[0, 2 * n_rep])


def test_decoder_layer_attention_class_dispatch():
    """Decoder layer uses the requested attention class; default stays GQA."""
    config = _config(num_hidden_layers=1)
    default_layer = Qwen3DFlashDecoderLayer(config, layer_idx=0)
    assert isinstance(default_layer.self_attn, Qwen3DFlashAttention)
    assert not isinstance(default_layer.self_attn, Qwen3DFlashDiffAttention)

    diff_layer = Qwen3DFlashDecoderLayer(
        config, layer_idx=0, attention_class=Qwen3DFlashDiffAttention
    )
    assert isinstance(diff_layer.self_attn, Qwen3DFlashDiffAttention)
