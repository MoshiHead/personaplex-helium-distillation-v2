# SPDX-License-Identifier: MIT
"""GQA (grouped-query attention) streaming transformer for the distilled student.

The teacher (`moshi.modules.transformer.StreamingTransformer`) only implements
plain multi-head attention: one shared head count for Q, K and V, with a single
fused `in_proj_weight`. There is no GQA anywhere in the teacher codebase.

This module is a parallel implementation, used only by the student's temporal
transformer, that supports `num_kv_heads < num_heads`. The teacher's
`StreamingTransformer` and `StreamingMultiheadAttention` are not modified or
imported into the forward path here (only a few stateless helpers are reused).
It mirrors the teacher's streaming/KV-cache/RoPE/CUDA-graph-ability contract so
it can be dropped into `LMGen` exactly like the teacher's transformer.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import math
import typing as tp

import torch
import torch.nn as nn
from torch.nn import functional as F

from moshi.modules.streaming import StreamingModule
from moshi.modules.transformer import (
    KVCacheResult,
    RingKVCache,
    LayerScale,
    create_norm_fn,
)
from moshi.utils.compile import no_compile
from moshi.modules.gating import make_gating


def _apply_rope_single(x: torch.Tensor, offset: torch.Tensor, max_period: float) -> torch.Tensor:
    """RoPE for a single tensor `x`: [B, H, T, D] (always `time_before_heads=False`
    layout here). A deliberate reimplementation of the rotation math in
    `moshi.modules.rope.apply_rope`, NOT a call into it -- that function is
    decorated `@torch_compile_lazy`, giving it one process-wide compiled-function
    cache shared with the teacher's `StreamingMultiheadAttention`. Two problems
    with routing through it here: (1) it takes a `(q, k)` pair, and this module's
    Q and K have different head counts (GQA), so calling it as `apply_rope(q, q,
    ...)` / `apply_rope(k, k, ...)` aliases its two arguments -- a pattern the
    teacher's code never produces, since its own q and k are always distinct
    tensors; (2) the student's shapes differ from the teacher's, so a shape-
    triggered recompile of that SHARED compiled function can occur on a call
    made from inside `LMGen`'s `CUDAGraphed` capture window (`torch.cuda.graph()`
    requires no new CUDA work -- allocation, compilation, kernel autotuning --
    during capture). Both produced a real `torch.cuda.graphs.py` capture failure
    ("operation failed due to a previous error") on an actual GPU. Plain, eager,
    uncompiled ops here sidestep all of that; RoPE is cheap next to the rest of
    attention, so there's no real performance cost to not compiling it.
    """
    B, H, T, D = x.shape
    assert D % 2 == 0
    ds = torch.arange(D // 2, device=x.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = (offset.float() + torch.arange(T, device=x.device, dtype=torch.float32)).view(1, 1, -1, 1)
    xr = x[..., 0::2].float()
    xi = x[..., 1::2].float()
    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)
    xor = xr * rotr - xi * roti
    xoi = xr * roti + xi * rotr
    return torch.stack([xor, xoi], dim=-1).reshape(B, H, T, D).to(x.dtype)


@dataclass
class _GQAMHAState:
    kv_cache: RingKVCache
    offset: torch.Tensor
    offset_cpu: int

    def reset(self):
        self.kv_cache.reset()
        self.offset.zero_()
        self.offset_cpu = 0


class GQAStreamingMultiheadAttention(StreamingModule[_GQAMHAState]):
    """Causal streaming attention with `num_kv_heads <= num_heads` (GQA/MQA).

    Unlike the teacher's `StreamingMultiheadAttention`, this uses separate
    `q_proj` / `k_proj` / `v_proj` linears (instead of one fused `in_proj_weight`)
    since Q and KV live at different widths. `num_heads` must be divisible by
    `num_kv_heads`.

    Args:
        embed_dim (int): model dimension (d_model).
        num_heads (int): number of query heads.
        num_kv_heads (int): number of key/value heads. `num_heads % num_kv_heads == 0`.
        causal (bool): causal mask applied automatically.
        context (int, optional): number of past time steps visible to attention.
        max_period (float, optional): RoPE base period; None disables RoPE.
        device, dtype: passed to factory kwargs.
    """

    _fsdp_final = True

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        causal: bool = True,
        context: tp.Optional[int] = None,
        max_period: tp.Optional[float] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (embed_dim, num_heads)
        assert num_heads % num_kv_heads == 0, (num_heads, num_kv_heads)
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.causal = causal
        self.context = context
        self.max_period = max_period

        kv_dim = num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=False, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **factory_kwargs)

    def _init_streaming_state(self, batch_size: int) -> _GQAMHAState:
        if self.context is None:
            raise RuntimeError(
                "Cannot create a streaming KVCache without a context to estimate capacity."
            )
        device = self.q_proj.weight.device
        dtype = self.k_proj.weight.dtype
        kv_cache = RingKVCache(
            batch_size, self.num_kv_heads, self.head_dim, self.context, device, dtype
        )
        return _GQAMHAState(
            kv_cache,
            offset=torch.zeros(1, device=device, dtype=torch.long),
            offset_cpu=0,
        )

    def _complete_kv(self, k, v) -> KVCacheResult:
        state = self._streaming_state
        if state is None:
            return KVCacheResult.from_kv(k, v)
        else:
            return state.kv_cache.complete(k, v)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        assert query is key and key is value, "self-attention only"
        state = self._streaming_state
        B, T, _ = query.shape

        if state is None:
            offset = torch.zeros(1, device=query.device, dtype=torch.long)
        else:
            assert self.causal, "Streaming only available for causal"
            offset = state.offset

        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.max_period is not None:
            # See _apply_rope_single's docstring: deliberately not moshi.modules.rope.apply_rope
            # (shared, torch.compile-cached, and shaped for a paired q/k of equal head count).
            q = _apply_rope_single(q, offset, self.max_period)
            k = _apply_rope_single(k, offset, self.max_period)

        k, v, pos_k = self._complete_kv(k, v)
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        if self.causal:
            pos_k = pos_k.view(1, -1)
            pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(-1, 1)
            delta = pos_q - pos_k
            attn_bias = (pos_k >= 0) & (delta >= 0)
            if self.context is not None:
                attn_bias = attn_bias & (delta < self.context)
        else:
            attn_bias = None

        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(B, T, self.embed_dim)
        x = self.out_proj(x)

        if state is not None:
            state.offset.add_(T)
            state.offset_cpu += T
        return x


@dataclass
class _GQALayerState:
    offset_cpu: int

    def reset(self):
        self.offset_cpu = 0


class GQAStreamingTransformerLayer(StreamingModule[_GQALayerState]):
    """Pre-norm transformer layer with GQA attention and a gated FFN.

    Structurally mirrors `moshi.modules.transformer.StreamingTransformerLayer`
    (same norm placement, same gated-FFN helper), swapping in
    `GQAStreamingMultiheadAttention` for the teacher's plain MHA.
    """

    _fsdp_final = True

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dim_feedforward: int,
        causal: bool = True,
        context: tp.Optional[int] = None,
        max_period: tp.Optional[float] = None,
        norm: str = "rms_norm_f32",
        layer_scale: tp.Optional[float] = None,
        gating: str = "silu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = GQAStreamingMultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            causal=causal,
            context=context,
            max_period=max_period,
            **factory_kwargs,
        )
        self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.gating = make_gating(gating, d_model, dim_feedforward, **factory_kwargs)

        if layer_scale is None:
            self.layer_scale_1: nn.Module = nn.Identity()
            self.layer_scale_2: nn.Module = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale, **factory_kwargs)
            self.layer_scale_2 = LayerScale(d_model, layer_scale, **factory_kwargs)

    def _init_streaming_state(self, batch_size: int) -> _GQALayerState:
        return _GQALayerState(offset_cpu=0)

    def _sa_block(self, x: torch.Tensor) -> torch.Tensor:
        x_orig = x
        x = self.norm1(x)
        update = self.self_attn(x, x, x)
        return x_orig + self.layer_scale_1(update)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x_orig = x
        x = self.norm2(x)
        update = self.gating(x)
        return x_orig + self.layer_scale_2(update)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with ExitStack() as stack:
            if x.device.type != "cuda":
                stack.enter_context(no_compile())
            x = self._sa_block(x)
            x = self._ff_block(x)
            state = self._streaming_state
            if state:
                state.offset_cpu += x.shape[1]
            return x


@dataclass
class _GQATransformerState:
    offset: torch.Tensor

    def reset(self):
        self.offset.zero_()


class GQAStreamingTransformer(StreamingModule[_GQATransformerState]):
    """Stack of `GQAStreamingTransformerLayer`, with RoPE shared across layers.

    Drop-in replacement for `moshi.modules.transformer.StreamingTransformer`
    from the point of view of `LMModel.forward_codes` / `LMGen`: same
    `forward(x) -> x` contract, same `.streaming()` / `.reset_streaming()` API.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        num_layers: int,
        dim_feedforward: int,
        causal: bool = True,
        context: tp.Optional[int] = None,
        max_period: float = 10_000,
        positional_scale: float = 1.0,
        norm: str = "rms_norm_f32",
        layer_scale: tp.Optional[float] = None,
        gating: str = "silu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.max_period = max_period
        self.positional_scale = positional_scale

        self.layers = nn.ModuleList(
            [
                GQAStreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    dim_feedforward=dim_feedforward,
                    causal=causal,
                    context=context,
                    max_period=max_period,
                    norm=norm,
                    layer_scale=layer_scale,
                    gating=gating,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def _init_streaming_state(self, batch_size: int) -> _GQATransformerState:
        device = next(self.parameters()).device
        return _GQATransformerState(offset=torch.zeros(1, device=device, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        state = self._streaming_state
        if state is not None:
            state.offset.add_(x.shape[1])
        return x
