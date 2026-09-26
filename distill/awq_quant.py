# SPDX-License-Identifier: MIT
"""Minimal AWQ-style INT4 weight-only quantizer, scoped to the student's
TEMPORAL TRANSFORMER linears only (q_proj/k_proj/v_proj/out_proj, gating
linear_in/linear_out). Bridge, depth transformer, output heads, embeddings, and
Mimi stay FP16 -- see PersonaPlex_Distill_RunPod.ipynb export step and
moshi/moshi/offline.py's `--model student` / benchmark-mode changes.

Why hand-rolled rather than `autoawq`: that library calibrates standard HF
`*ForCausalLM` module trees by name (`self_attn.q_proj`, `mlp.gate_proj`, ...);
this repo's attention/gating modules have different names and forward
signatures (see distill/gqa_attention.py, moshi/modules/gating.py), so
`autoawq` cannot be pointed at this model without first rewriting it to look
like a HF model -- more invasive than implementing the actual algorithm.

This implements the core AWQ idea (Lin et al., 2023): search for a per-input-
channel scale that minimizes the round-to-nearest INT4 quantization error of
the *salient* channels (those with large activation magnitude), rather than
plain RTN. It does NOT implement AWQ's full grid search over multiple
candidate scale exponents with a real GEMM kernel -- weights are quantized
in place and simulated via fake-quant (dequantize immediately after
quantizing) for accuracy testing; `pack_int4` additionally provides a real
bit-packed INT4 representation (2 values per byte) plus per-group scales for
storage/export, dequantized back to the working dtype at load time (weight-
only, activations stay FP16/BF16 -- "W4A16").
"""

from dataclasses import dataclass
import typing as tp

import torch
import torch.nn as nn

TEMPORAL_LINEAR_NAMES = ("q_proj", "k_proj", "v_proj", "out_proj", "linear_in", "linear_out")


def _is_temporal_linear(name: str) -> bool:
    return name.startswith("transformer.") and any(name.endswith(f".{n}.weight") for n in TEMPORAL_LINEAR_NAMES)


@dataclass
class QuantizedLinear:
    """Bit-packed INT4 weight-only representation of one `nn.Linear.weight`."""
    packed: torch.Tensor  # uint8, [out, ceil(in/2)]
    scales: torch.Tensor  # [out, num_groups]
    zeros: torch.Tensor  # [out, num_groups] (int4 zero-point, stored as int8)
    group_size: int
    in_features: int
    out_features: int

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        out, packed_in = self.packed.shape
        high = (self.packed >> 4).to(torch.int16)
        low = (self.packed & 0x0F).to(torch.int16)
        unpacked = torch.stack([low, high], dim=-1).reshape(out, packed_in * 2)[:, : self.in_features]
        num_groups = self.scales.shape[1]
        unpacked = unpacked.view(out, num_groups, -1)
        zeros = self.zeros.view(out, num_groups, 1).to(torch.int16)
        scales = self.scales.view(out, num_groups, 1)
        dequant = (unpacked - zeros).to(scales.dtype) * scales
        return dequant.reshape(out, self.in_features).to(dtype)


def compute_awq_scales(weight: torch.Tensor, activation_rms: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Per-input-channel scale `s = activation_rms^alpha`, the AWQ heuristic: scale
    UP the channels the activations lean on most heavily before quantizing (so
    they get more effective precision), and divide the scale back out of the
    weight so the linear's output is unchanged before quantization error is
    introduced. `weight`: [out, in], `activation_rms`: [in].
    """
    s = activation_rms.clamp_min(1e-5).pow(alpha)
    s = s / s.mean()
    return s


def quantize_int4_grouped(
    weight: torch.Tensor, scale_per_channel: torch.Tensor, group_size: int = 128,
) -> QuantizedLinear:
    """Per-(output-row, input-group) affine INT4 quantization, after applying the
    AWQ activation-aware channel scale to the weight's input dimension.
    """
    out_features, in_features = weight.shape
    assert in_features % group_size == 0 or group_size >= in_features, (
        f"in_features={in_features} must be a multiple of group_size={group_size}, "
        "or group_size >= in_features (single group)."
    )
    group_size = min(group_size, in_features)
    scaled = weight.float() * scale_per_channel.float().unsqueeze(0)  # [out, in]

    num_groups = in_features // group_size
    grouped = scaled.view(out_features, num_groups, group_size)
    g_min = grouped.min(dim=-1, keepdim=True).values
    g_max = grouped.max(dim=-1, keepdim=True).values
    scales = ((g_max - g_min) / 15.0).clamp_min(1e-8)
    zeros = torch.round(-g_min / scales).clamp(0, 15)

    q = torch.round(grouped / scales + zeros).clamp(0, 15).to(torch.uint8)
    q = q.view(out_features, in_features)

    if in_features % 2 == 1:
        q = torch.nn.functional.pad(q, (0, 1))
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)

    return QuantizedLinear(
        packed=packed, scales=scales.squeeze(-1), zeros=zeros.squeeze(-1).to(torch.int8),
        group_size=group_size, in_features=in_features, out_features=out_features,
    )


@torch.no_grad()
def collect_activation_rms(module: nn.Module, calibration_inputs: tp.Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Runs `module` (the student's `.transformer`) over calibration batches,
    capturing per-input-channel activation RMS for every temporal linear, keyed
    by the linear's qualified name (matching `TEMPORAL_LINEAR_NAMES`).

    `linear_in`/`linear_out` (inside `ActivationGating`, moshi/modules/gating.py)
    need special handling: `gating_forward_kernel` calls `F.linear` directly on
    their `.weight` tensors rather than invoking them as `nn.Module`s, so a
    forward hook registered directly on those submodules never fires. Instead
    this hooks the parent `ActivationGating` module (`gating_forward_kernel` IS
    reached via a real `self.gating(x)` call in `StreamingTransformerLayer`)
    and reconstructs both linears' activations from its input -- `linear_in`'s
    input is exactly the gate's input; `linear_out`'s input is the intermediate
    gated-hidden value, recomputed via `_gating_hidden_activation` (same helper
    distill/init_from_teacher.py uses for the identical reason).
    """
    from .init_from_teacher import _gating_hidden_activation

    rms: dict[str, torch.Tensor] = {}

    def update(name: str, x: torch.Tensor):
        flat = x.detach().float().reshape(-1, x.shape[-1])
        batch_rms = flat.pow(2).mean(dim=0).sqrt()
        rms[name] = 0.5 * rms[name] + 0.5 * batch_rms if name in rms else batch_rms

    def make_linear_hook(name):
        def hook(mod, inputs, output):
            update(name, inputs[0])
        return hook

    def make_gating_hook(prefix):
        def hook(mod, inputs, output):
            x = inputs[0]
            update(f"{prefix}.linear_in", x)
            update(f"{prefix}.linear_out", _gating_hidden_activation(mod, x))
        return hook

    handles = []
    for name, sub in module.named_modules():
        leaf = name.split(".")[-1]
        if isinstance(sub, nn.Linear) and leaf in ("q_proj", "k_proj", "v_proj", "out_proj"):
            handles.append(sub.register_forward_hook(make_linear_hook(name)))
        elif leaf == "gating":
            handles.append(sub.register_forward_hook(make_gating_hook(name)))
    try:
        for x in calibration_inputs:
            module(x)
    finally:
        for h in handles:
            h.remove()
    return rms


def quantize_student_temporal_transformer(
    student, calibration_inputs: tp.Sequence[torch.Tensor], group_size: int = 128, alpha: float = 0.5,
) -> dict[str, QuantizedLinear]:
    """Returns {qualified_name: QuantizedLinear} for every temporal-transformer
    linear. Does NOT modify `student` in place -- see `export.py` for how this
    is packaged into an export artifact alongside the FP16 bridge/heads/etc.
    """
    activation_rms = collect_activation_rms(student.transformer, calibration_inputs)
    quantized = {}
    for name, sub in student.transformer.named_modules():
        if isinstance(sub, nn.Linear) and name.split(".")[-1] in TEMPORAL_LINEAR_NAMES:
            full_name = f"transformer.{name}.weight"
            if name not in activation_rms:
                raise RuntimeError(
                    f"No calibration activations captured for {full_name}; "
                    "calibration_inputs did not exercise this layer."
                )
            scale = compute_awq_scales(sub.weight, activation_rms[name], alpha=alpha)
            quantized[full_name] = quantize_int4_grouped(sub.weight, scale, group_size=group_size)
            quantized[full_name].awq_scale = scale  # stashed to undo the scale on dequantize
    return quantized


def dequantize_for_inference(quantized: dict[str, tp.Any], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Reconstructs FP weight tensors from a `quantize_student_temporal_transformer`
    result, undoing the AWQ channel scale. Used both by a correctness test
    (quantized vs. original weight error) and by an inference path that prefers
    dequantize-then-matmul over a fused INT4 GEMM kernel (no custom kernel is
    implemented here -- see module docstring)."""
    out = {}
    for name, q in quantized.items():
        w = q.dequantize(dtype)
        out[name] = w / q.awq_scale.to(dtype).unsqueeze(0)
    return out
