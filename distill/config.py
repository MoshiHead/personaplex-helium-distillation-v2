# SPDX-License-Identifier: MIT
"""Config loading for student architectures (`distill/configs/*.yaml`)."""

from dataclasses import dataclass, field
from pathlib import Path
import typing as tp

import yaml

_CONFIG_DIR = Path(__file__).parent / "configs"


@dataclass
class InitConfig:
    num_calibration_batches: int = 32
    keep_first: int = 3
    keep_last: int = 3


@dataclass
class TeacherReference:
    dim: int = 4096
    num_heads: int = 32
    num_layers: int = 32
    context: int = 3000
    n_q: int = 16
    dep_q: int = 16
    card: int = 2048
    text_card: int = 32000
    depformer_dim: int = 1024
    depformer_num_heads: int = 16
    depformer_num_layers: int = 6
    depformer_dim_feedforward: int = 4224
    max_period: float = 10000.0


@dataclass
class BridgeConfig:
    in_dim: int = 2048
    out_dim: int = 4096


@dataclass
class StudentConfig:
    name: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rope_theta: float
    max_frames: int
    norm: str
    gating: str
    layer_scale: tp.Optional[float]
    causal: bool
    init: InitConfig
    teacher_reference: TeacherReference
    bridge: BridgeConfig

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0
        head_dim = self.hidden_size // self.num_attention_heads
        assert head_dim == 128, (
            f"head_dim must stay 128 to match the teacher's RoPE/head geometry, got {head_dim}"
        )
        assert self.bridge.in_dim == self.hidden_size
        assert self.bridge.out_dim == self.teacher_reference.dim

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def load_student_config(name_or_path: str) -> StudentConfig:
    """Load a student config by name (e.g. "student_ppx_s") or by path to a yaml file."""
    path = Path(name_or_path)
    if not path.exists():
        path = _CONFIG_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        path = _CONFIG_DIR / name_or_path
    if not path.exists():
        raise FileNotFoundError(f"No student config found for '{name_or_path}' (looked in {_CONFIG_DIR})")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return StudentConfig(
        name=raw["name"],
        num_hidden_layers=raw["num_hidden_layers"],
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        rope_theta=raw["rope_theta"],
        max_frames=raw["max_frames"],
        norm=raw["norm"],
        gating=raw["gating"],
        layer_scale=raw.get("layer_scale"),
        causal=raw.get("causal", True),
        init=InitConfig(**raw.get("init", {})),
        teacher_reference=TeacherReference(**raw.get("teacher_reference", {})),
        bridge=BridgeConfig(**raw["bridge"]),
    )
