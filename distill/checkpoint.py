# SPDX-License-Identifier: MIT
"""Resumable checkpointing for student training.

The saved checkpoint holds ONLY the trained delta: the student's transformer,
embeddings, and bridge (plus the L_hidden projection heads, optimizer, and
schedule/RNG state). It never re-saves the frozen depformer/heads/Mimi weights
that are reused verbatim from the teacher checkpoint at load time -- "don't
duplicate them" (see PersonaPlex_Distill_RunPod.ipynb export step and
moshi/moshi/models/loaders.get_student_lm).
"""

from dataclasses import dataclass
from pathlib import Path
import random
import typing as tp

import numpy as np
import torch

from .student_model import StudentLMModel, FROZEN_PREFIXES


def trainable_state_dict(model: StudentLMModel) -> dict[str, torch.Tensor]:
    full = model.state_dict()
    return {k: v for k, v in full.items() if not k.startswith(FROZEN_PREFIXES)}


def load_trainable_state_dict(model: StudentLMModel, state_dict: dict[str, torch.Tensor]):
    result = model.load_state_dict(state_dict, strict=False, assign=False)
    unexpected_non_frozen = [k for k in result.unexpected_keys if not k.startswith(FROZEN_PREFIXES)]
    assert not unexpected_non_frozen, unexpected_non_frozen
    missing_non_frozen = [k for k in result.missing_keys if not k.startswith(FROZEN_PREFIXES)]
    assert not missing_non_frozen, f"Checkpoint is missing trainable keys: {missing_non_frozen}"


@dataclass
class TrainingCheckpoint:
    step: int
    student_config_name: str
    selected_teacher_layers: list[int]


def save(
    path: str,
    step: int,
    student: StudentLMModel,
    hidden_projections: torch.nn.ModuleList,
    optimizer: torch.optim.Optimizer,
    schedule_state: dict,
    running_norm_states: dict,
    selected_teacher_layers: list[int],
    extra: tp.Optional[dict] = None,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "student_config_name": student.student_config.name,
        "student_state_dict": trainable_state_dict(student),
        "hidden_projections_state_dict": hidden_projections.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "schedule_state": schedule_state,
        "running_norm_states": running_norm_states,
        "selected_teacher_layers": selected_teacher_layers,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "extra": extra or {},
    }
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    Path(tmp_path).replace(path)


def load(
    path: str,
    student: StudentLMModel,
    hidden_projections: torch.nn.ModuleList,
    optimizer: tp.Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> dict:
    # weights_only=False: the payload holds numpy/Python RNG state alongside the tensors, which
    # torch>=2.6's default weights_only=True refuses to unpickle. These are our own checkpoints.
    payload = torch.load(path, map_location=map_location, weights_only=False)
    assert payload["student_config_name"] == student.student_config.name, (
        f"Checkpoint was trained with config {payload['student_config_name']!r}, "
        f"but model was built with {student.student_config.name!r}."
    )
    load_trainable_state_dict(student, payload["student_state_dict"])
    hidden_projections.load_state_dict(payload["hidden_projections_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    rng = payload["rng_state"]
    torch.set_rng_state(rng["torch"])
    if rng["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])

    return payload
