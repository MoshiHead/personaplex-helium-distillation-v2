# SPDX-License-Identifier: MIT
"""The distilled student LM: small GQA temporal transformer + Bridge, with the
depth transformer, output heads, embeddings-for-the-frozen-path and Mimi all
reused verbatim (frozen) from the teacher checkpoint.

`StudentLMModel` subclasses `moshi.models.lm.LMModel` so that `LMGen` (the
streaming generation driver used by both `server.py` and `offline.py`) can
drive it exactly like the teacher: `forward_codes`, `forward_depformer`,
`forward_depformer_training`, `.depformer`, `.dep_q`, `.delays`, etc. are all
inherited unchanged. Only `forward_embeddings` is overridden, to route the
student's transformer output through the Bridge before it reaches the frozen
`out_norm` / `text_linear` / depformer chain.
"""

from pathlib import Path
import typing as tp

import torch
from safetensors.torch import load_file

from moshi.models.lm import LMModel, ScaledEmbedding
from moshi.models.loaders import _lm_kwargs as _TEACHER_LM_KWARGS

from .config import StudentConfig, load_student_config
from .bridge import Bridge
from .gqa_attention import GQAStreamingTransformer

# Module-name prefixes that are frozen and reused verbatim from the teacher
# checkpoint. Everything else (transformer.*, emb.*, text_emb.*, bridge.*) is
# new and trainable.
FROZEN_PREFIXES = (
    "out_norm.",
    "text_linear.",
    "depformer_in.",
    "depformer_emb.",
    "depformer_text_emb.",
    "depformer.",
    "linears.",
)


def teacher_lm_kwargs() -> dict:
    """The exact kwargs `moshi.models.loaders.get_moshi_lm` uses to build the teacher,
    with the same `dep_q=16` patch applied. Used to build a teacher-shaped meta
    skeleton whose frozen submodules line up 1:1 with the teacher checkpoint."""
    kwargs = dict(_TEACHER_LM_KWARGS)
    kwargs["dep_q"] = 16
    return kwargs


class StudentLMModel(LMModel):
    def __init__(
        self,
        student_config: StudentConfig,
        device: tp.Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        teacher_kwargs: tp.Optional[dict] = None,
    ):
        # Build a teacher-shaped skeleton on the meta device: near-zero cost, and it
        # guarantees the frozen submodules (depformer*, linears, text_linear, out_norm)
        # come out with exactly the shapes the teacher checkpoint expects.
        # `teacher_kwargs` defaults to the real 7B teacher's construction kwargs;
        # it's overridable so tests/test_init_sanity.py can exercise the full
        # init/wiring pipeline against a tiny synthetic teacher instead of
        # requiring a multi-GB download.
        super().__init__(device="meta", dtype=dtype, **(teacher_kwargs or teacher_lm_kwargs()))
        # `LMModel.__init__` builds `text_linear`, `out_norm`, `depformer_in`, and `linears`
        # via plain `torch.nn.Linear(...)` / `create_norm_fn(...)` calls with NO device/dtype
        # kwargs (see moshi/models/lm.py) -- they always land on default CPU/float32 regardless
        # of the `device="meta", dtype=dtype` passed above. `get_moshi_lm` (loaders.py) papers
        # over this with a blanket `model.to(device=device, dtype=dtype)` after construction;
        # without the same fix here these four submodule groups would be real (non-meta, wrong
        # dtype) tensors, defeating the "near-zero cost" meta build AND making
        # `load_frozen_from_teacher` below cast teacher weights to float32 (via
        # `self.text_linear.weight.dtype`) instead of the model's real working dtype.
        self.to(device="meta", dtype=dtype)

        cfg = student_config
        self.student_config = cfg

        # Replace the teacher-scale trainable path with the student-scale one.
        del self.transformer
        self.transformer = GQAStreamingTransformer(
            d_model=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            num_layers=cfg.num_hidden_layers,
            dim_feedforward=cfg.intermediate_size,
            causal=cfg.causal,
            context=cfg.max_frames,
            max_period=cfg.rope_theta,
            layer_scale=cfg.layer_scale,
            norm=cfg.norm,
            gating=cfg.gating,
            device=device,
            dtype=dtype,
        )

        del self.emb
        self.emb = torch.nn.ModuleList(
            [
                ScaledEmbedding(self.card + 1, cfg.hidden_size, zero_idx=self.zero_token_id,
                                 device=device, dtype=dtype)
                for _ in range(self.n_q)
            ]
        )
        del self.text_emb
        self.text_emb = ScaledEmbedding(
            self.text_card + 1, cfg.hidden_size, zero_idx=self.zero_token_id,
            device=device, dtype=dtype,
        )

        self.bridge = Bridge(cfg.bridge.in_dim, cfg.bridge.out_dim, device=device, dtype=dtype)
        self._student_device = torch.device(device)

    def load_frozen_from_teacher(self, teacher_state_dict: dict[str, torch.Tensor]):
        """Fill in the frozen submodules (depformer*, linears, text_linear, out_norm)
        from a teacher `LMModel` state dict, and freeze them. `transformer.*`,
        `emb.*`, `text_emb.*` keys in `teacher_state_dict` are ignored (unexpected) by
        design -- those modules were already replaced with student-scale ones above.
        """
        state_dict = {
            k: v.to(device=self._student_device, dtype=self.text_linear.weight.dtype)
            for k, v in teacher_state_dict.items()
            if k.startswith(FROZEN_PREFIXES)
        }
        result = self.load_state_dict(state_dict, strict=False, assign=True)
        missing_frozen = [k for k in result.missing_keys if k.startswith(FROZEN_PREFIXES)]
        assert not missing_frozen, f"Teacher checkpoint is missing frozen keys: {missing_frozen}"
        assert not result.unexpected_keys, (
            f"Unexpected keys after filtering to frozen prefixes: {result.unexpected_keys}"
        )
        for meta_name, module in self.named_modules():
            for p in module.parameters(recurse=False):
                assert p.device.type != "meta", (
                    f"{meta_name} still has meta-device parameters after loading frozen "
                    "teacher weights -- this module was expected to be either replaced "
                    "with a student-scale module or filled in from the teacher checkpoint."
                )
        self.freeze_teacher_components()

    def freeze_teacher_components(self):
        for name, p in self.named_parameters():
            if name.startswith(FROZEN_PREFIXES):
                p.requires_grad = False

    def unfreeze_depth_transformer(self):
        """Phase 4 ("Polish") only: unfreeze the depth transformer at 0.1x LR."""
        for p in self.depformer.parameters():
            p.requires_grad = True

    def trainable_parameters(self) -> tp.Iterator[torch.nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def forward_embeddings(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformer_out = self.transformer(input)
        transformer_out = self.bridge(transformer_out)
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        text_logits = self.text_linear(transformer_out)
        text_logits = text_logits[:, None]
        return transformer_out, text_logits

    def raw_temporal_output(self, sequence: torch.Tensor) -> torch.Tensor:
        """Student transformer output before the Bridge (used by L_hidden, which
        distills at the temporal-transformer level, not the bridged/teacher level)."""
        return self.transformer(self.embed_codes(sequence))

    def bridged_conditioning(self, sequence: torch.Tensor) -> torch.Tensor:
        """Bridge output (pre teacher out_norm) -- used as the L_bridge prediction,
        compared against the teacher's raw (pre out_norm) transformer output."""
        return self.bridge(self.raw_temporal_output(sequence))


def build_student_lm(
    student_config: tp.Union[str, StudentConfig],
    teacher_checkpoint: tp.Union[str, Path],
    device: tp.Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> StudentLMModel:
    """Build a `StudentLMModel` with freshly-initialized (untrained) student-scale
    modules and the frozen teacher submodules loaded in. This is the *structural*
    build only -- see `distill/init_from_teacher.py` for the actual data-driven
    initialization (layer selection, SVD width reduction, GQA KV pooling, bridge
    least-squares fit) that should be run before this model produces anything
    resembling coherent speech.
    """
    if isinstance(student_config, str):
        student_config = load_student_config(student_config)

    model = StudentLMModel(student_config, device=device, dtype=dtype)

    teacher_checkpoint = str(teacher_checkpoint)
    if teacher_checkpoint.endswith(".safetensors"):
        teacher_state_dict = load_file(teacher_checkpoint, device="cpu")
    else:
        teacher_state_dict = torch.load(teacher_checkpoint, map_location="cpu")

    model.load_frozen_from_teacher(teacher_state_dict)
    model.eval()
    return model
