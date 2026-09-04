import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
    eager_attention_forward,
)
from typing_extensions import Unpack

if TYPE_CHECKING:
    from collections.abc import Callable


# Local copy of rotate_half to avoid dependency on internal transformers functions
def _rotate_half(x):
    """Rotates half the hidden dims of the input (local implementation)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids=None,  # noqa: ARG001
    unsqueeze_dim=1,
):
    """Apply rotary position embeddings (local implementation)."""

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (_rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Implements the custom attention which injects the target models
    # hidden states into the kv cache.
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,  # type: ignore[operator]
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads  # type: ignore[operator]
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_attention_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.k_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_key_value_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.v_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_key_value_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,  # type: ignore[operator]
            config.hidden_size,  # type: ignore[arg-type]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.sliding_window = (
            config.sliding_window
            if hasattr(config, "layer_types")
            and config.layer_types is not None
            and config.layer_types[layer_idx] == "sliding_attention"  # type: ignore[index]
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Instead of computing the k and v matricies from the hidden states,
        # the target_hidden is injected into the kv cache, (shape is context
        # length + block size)
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        # This is the main difference from the usual attention mechanism.
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        # note the length becomes context length + block size
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if (
            self.config._attn_implementation is not None  # noqa: SLF001
            and self.config._attn_implementation != "eager"  # noqa: SLF001
        ):
            attn_fn = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation  # noqa: SLF001
            ]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        if getattr(self, "capture_attention", False) and attn_weights is not None:
            # 调试用: 捕获 GQA 层注意力权重(默认关闭, 仅 eager 等会返回权重时生效)
            self._debug_attn = attn_weights.detach()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def lambda_init_fn(depth: int) -> float:
    """Frozen depth-dependent lambda bias from the Diff-Transformer paper."""
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match the query heads (GQA).

    Equivalent to ``torch.repeat_interleave(x, dim=1, repeats=n_rep)`` but keeps
    each sub-head's grouped ordering so two paired sub-heads stay in the same KV
    group (required for the diff-attention subtraction).
    """
    bsz, num_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bsz, num_kv_heads, n_rep, slen, head_dim)
        .reshape(bsz, num_kv_heads * n_rep, slen, head_dim)
    )


def block_mask_to_float_mask(
    block_mask: BlockMask,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a flex BlockMask to a dense float attention mask.

    Vectorized: the ``mask_mod`` is evaluated once over the full ``(Q, KV)`` grid
    (broadcastable index tensors) instead of a per-row Python loop, so it is
    cheap enough to run per-forward and stays ``torch.compile`` compatible.
    """
    bsz, num_heads, q_len, kv_len = block_mask.shape
    q_idx = torch.arange(q_len, device=device, dtype=torch.long)[:, None]
    kv_idx = torch.arange(kv_len, device=device, dtype=torch.long)[None, :]
    mask = block_mask.mask_mod(
        torch.zeros(1, device=device, dtype=torch.long),
        torch.zeros(1, device=device, dtype=torch.long),
        q_idx,
        kv_idx,
    )  # [q_len, kv_len] bool
    dense = torch.zeros((bsz, num_heads, q_len, kv_len), device=device, dtype=dtype)
    dense.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    return dense


class Qwen3DFlashDiffAttention(nn.Module):
    """Diff-Transformer (diffv1) attention adapted to DFlash's KV injection.

    Mirrors ``MultiheadDiffAttn`` from the Diff-Transformer reference repo: the
    baseline heads are paired two-by-two and the two attention distributions over
    the same keys are subtracted (``attn[:, :, 0] - lambda * attn[:, :, 1]``).
    ``lambda`` is a scalar produced by a formula over four trainable
    ``(head_dim,)`` vectors plus a frozen depth-dependent bias ``lambda_init``:

        lambda = exp(sum(lambda_q1 * lambda_k1))
               - exp(sum(lambda_q2 * lambda_k2))
               + lambda_init(depth)

    The subtracted output is normalized per head over its ``2 * head_dim``
    (the paper's per-head GroupNorm, implemented here as RMSNorm, matching the
    reference repo's ``subln``) and scaled by ``(1 - lambda_init)``.

    Unlike the reference, K/V are injected from the verifier ``target_hidden``
    and concatenated with the draft tokens' K/V, exactly like
    ``Qwen3DFlashAttention`` (DSPark/DFlash design). Projection shapes are
    identical to the baseline GQA layer: heads are halved but V doubles, so
    q/k/v/o output widths are unchanged.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,  # type: ignore[operator]
        )
        # Baseline heads are paired two-by-two; KV heads are halved the same way.
        self.num_heads = config.num_attention_heads // 2  # type: ignore[operator]
        self.num_kv_heads = config.num_key_value_heads // 2  # type: ignore[operator]
        self.n_rep = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        # Projection shapes are identical to the baseline GQA attention.
        self.q_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_attention_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.k_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            2 * self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.v_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            self.num_kv_heads * 2 * self.head_dim,
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,  # type: ignore[operator]
            config.hidden_size,  # type: ignore[arg-type]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # type: ignore[arg-type]

        # Lambda: four trainable (head_dim,) vectors reparameterized through exp,
        # plus the frozen depth-dependent lambda_init.
        self.lambda_init = lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1)
        )

        # Per-head normalization over each paired head's 2*head_dim output.
        self.subln = Qwen3RMSNorm(2 * self.head_dim, eps=1e-5)

        self.sliding_window = (
            config.sliding_window
            if hasattr(config, "layer_types")
            and config.layer_types is not None
            and config.layer_types[layer_idx] == "sliding_attention"  # type: ignore[index]
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | BlockMask | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        # [bsz, 2*num_heads, q_len, head_dim]

        # K/V are injected from the verifier hidden states (DSPark/DFlash design):
        # the length becomes context length + block size.
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        # [bsz, 2*num_kv_heads, ctx_len+q_len, head_dim]
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, 2 * self.head_dim
        )
        v = v.transpose(1, 2)
        # [bsz, num_kv_heads, ctx_len+q_len, 2*head_dim]

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # 同一 pair 的两个分支必须用不同的 K(K1/K2): 展开时保留 (kv_head, branch)
        # 结构, 使 k[2j] -> K1_{j//n_rep}, k[2j+1] -> K2_{j//n_rep};
        # 不能像 repeat_kv 那样按 i//n_rep 扁平重复(会让两个分支共享同一个 k)。
        k = (
            k.reshape(bsz, self.num_kv_heads, 2, ctx_len + q_len, self.head_dim)
            .repeat_interleave(self.n_rep, dim=1)
            .reshape(bsz, 2 * self.num_heads, ctx_len + q_len, self.head_dim)
        )
        # V 保持共享: 每个 kv head 的 2*head_dim 同时给 pair 的两个分支。
        v = repeat_kv(v, self.n_rep)  # [bsz, num_heads, kv_len, 2*head_dim]
        q = q * self.scaling

        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        # [bsz, 2*num_heads, q_len, kv_len]

        if attention_mask is not None:
            if isinstance(attention_mask, BlockMask):
                # Flex `create_block_mask` path (simple_flex_attention impl).
                attention_mask = block_mask_to_float_mask(
                    attention_mask,
                    device=attn_weights.device,
                    dtype=attn_weights.dtype,
                )
            elif attention_mask.dtype == torch.bool:
                # Dense boolean mask (flex `create_mask`, used by the sdpa impl):
                # True = attend, False = masked. Convert to 0/-inf like the eager
                # float mask so the addition below is correct.
                attention_mask = torch.zeros(
                    attention_mask.shape,
                    device=attn_weights.device,
                    dtype=attn_weights.dtype,
                ).masked_fill_(
                    ~attention_mask.to(attn_weights.device), float("-inf")
                )
            attn_weights = torch.nan_to_num(attn_weights)
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).type_as(attn_weights)

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn_pair = attn_weights.view(bsz, self.num_heads, 2, q_len, -1)
        if getattr(self, "capture_attention", False):
            # 调试用: 捕获两个分支及相减后的注意力权重(默认关闭)
            self._debug_branch0 = attn_pair[:, :, 0].detach()
            self._debug_branch1 = attn_pair[:, :, 1].detach()
        attn_weights = attn_pair[:, :, 0] - lambda_full * attn_pair[:, :, 1]
        # [bsz, num_heads, q_len, kv_len]
        if getattr(self, "capture_attention", False):
            self._debug_attn = attn_weights.detach()

        attn = torch.matmul(attn_weights, v)
        # [bsz, num_heads, q_len, 2*head_dim]
        if getattr(self, "capture_attention", False):
            self._debug_pair_out_raw = attn.detach()  # subln 前每 pair 输出
        attn = self.subln(attn)
        if getattr(self, "capture_attention", False):
            self._debug_pair_out_normed = attn.detach()  # subln 后
        attn = attn * (1 - self.lambda_init)

        attn = attn.transpose(1, 2).reshape(bsz, q_len, -1)
        # [bsz, q_len, num_heads * 2 * head_dim] == baseline Q width
        attn_output = self.o_proj(attn)
        return attn_output, None


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        attention_class: type[nn.Module] = Qwen3DFlashAttention,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = attention_class(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,  # type: ignore[arg-type]
        )

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        # necessary, but kept here for BC
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        # The main difference between this method and the qwen 3 layer it is
        # built from is that it
        # passes the extra hidden states to the self attention from the verifier model.
        # Note that target_hidden is not modified here.
        assert hidden_states is not None  # noqa: S101
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states  # type: ignore[operator]
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states  # type: ignore[operator,return-value]
