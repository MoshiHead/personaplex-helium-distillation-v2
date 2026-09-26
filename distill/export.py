# SPDX-License-Identifier: MIT
"""Export a trained student to inference-ready checkpoints:
  1. BF16 checkpoint: the trainable delta only (transformer, embeddings,
     bridge) -- frozen depformer/heads/Mimi are never duplicated, they're
     always loaded from the teacher checkpoint at inference time
     (moshi/models/loaders.get_student_lm).
  2. AWQ INT4 checkpoint: same, but the temporal transformer's Linear weights
     (q_proj/k_proj/v_proj/out_proj/linear_in/linear_out) are additionally
     quantized to INT4 (W4A16) via distill/awq_quant.py. Everything else in
     the exported file (bridge, embeddings) stays BF16 -- per the quantization
     policy, INT4 error is only acceptable in the temporal transformer, never
     in the bridge/heads/Mimi (those aren't even part of this export; heads
     and Mimi are always the teacher's, loaded separately).

Note on the plan's "FP16" wording: this repo uses BF16 uniformly for the LM
(see moshi/models/loaders.get_moshi_lm's default dtype) -- there is no FP16
path anywhere in the existing code to match, so exports here use BF16 for
every non-INT4 tensor rather than introducing a third dtype.
"""

from pathlib import Path
import json
import typing as tp

import torch
from safetensors.torch import save_file, load_file

from .awq_quant import quantize_student_temporal_transformer, TEMPORAL_LINEAR_NAMES, QuantizedLinear
from .checkpoint import trainable_state_dict
from .student_model import StudentLMModel


def export_bf16(student: StudentLMModel, output_path: str, selected_teacher_layers: list[int]):
    state_dict = {k: v.to(torch.bfloat16) for k, v in trainable_state_dict(student).items()}
    metadata = {
        "student_config_name": student.student_config.name,
        "selected_teacher_layers": json.dumps(selected_teacher_layers),
        "quantization": "none",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, output_path, metadata=metadata)


def export_awq_int4(
    student: StudentLMModel,
    calibration_inputs: tp.Sequence[torch.Tensor],
    output_path: str,
    selected_teacher_layers: list[int],
    group_size: int = 128,
):
    quantized = quantize_student_temporal_transformer(student, calibration_inputs, group_size=group_size)

    state_dict: dict[str, torch.Tensor] = {}
    for name, q in quantized.items():
        prefix = name.removesuffix(".weight")
        state_dict[f"{prefix}.qweight"] = q.packed
        state_dict[f"{prefix}.scales"] = q.scales.to(torch.bfloat16)
        state_dict[f"{prefix}.zeros"] = q.zeros
        state_dict[f"{prefix}.awq_scale"] = q.awq_scale.to(torch.bfloat16)

    non_quantized_prefixes = tuple(f".{n}.weight" for n in TEMPORAL_LINEAR_NAMES)
    for k, v in trainable_state_dict(student).items():
        if k.startswith("transformer.") and k.endswith(non_quantized_prefixes):
            continue  # replaced by the quantized tensors above
        state_dict[k] = v.to(torch.bfloat16)

    metadata = {
        "student_config_name": student.student_config.name,
        "selected_teacher_layers": json.dumps(selected_teacher_layers),
        "quantization": "awq_int4_temporal_linears_only",
        "group_size": str(group_size),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, output_path, metadata=metadata)


def load_export_metadata(path: str) -> dict:
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    return header.get("__metadata__", {})


def load_export(path: str, student: StudentLMModel):
    """Loads a checkpoint produced by `export_bf16` or `export_awq_int4` into an
    already-built `StudentLMModel` (i.e. after `build_student_lm`, so the frozen
    teacher submodules are already in place). For an AWQ export, the temporal
    transformer's Linear weights are dequantized to the student's working dtype
    at load time (no custom INT4 GEMM kernel -- see awq_quant.py docstring): the
    on-disk file is 4-bit, the in-memory weights used for matmul are not.
    """
    metadata = load_export_metadata(path)
    tensors = load_file(path, device="cpu")
    working_dtype = student.text_linear.weight.dtype

    if metadata.get("quantization", "none") == "none":
        state_dict = {k: v.to(working_dtype) for k, v in tensors.items()}
        student.load_state_dict(state_dict, strict=False, assign=False)
        return

    assert metadata["quantization"] == "awq_int4_temporal_linears_only", metadata["quantization"]
    group_size = int(metadata["group_size"])

    quantized_prefixes = set()
    for key in tensors:
        if key.endswith(".qweight"):
            quantized_prefixes.add(key.removesuffix(".qweight"))

    plain_state_dict = {
        k: v.to(working_dtype) for k, v in tensors.items()
        if not any(k.startswith(p + ".") for p in quantized_prefixes)
    }
    student.load_state_dict(plain_state_dict, strict=False, assign=False)

    for prefix in quantized_prefixes:
        packed = tensors[f"{prefix}.qweight"]
        scales = tensors[f"{prefix}.scales"].float()
        zeros = tensors[f"{prefix}.zeros"]
        awq_scale = tensors[f"{prefix}.awq_scale"].float()
        # prefix looks like "transformer.layers.<i>.self_attn.q_proj"
        module = student
        for part in prefix.split("."):
            module = getattr(module, part) if not part.isdigit() else module[int(part)]
        out_features, in_features = module.weight.shape
        q = QuantizedLinear(packed=packed, scales=scales, zeros=zeros, group_size=group_size,
                             in_features=in_features, out_features=out_features)
        weight = q.dequantize(working_dtype) / awq_scale.to(working_dtype).unsqueeze(0)
        module.weight.data.copy_(weight)
