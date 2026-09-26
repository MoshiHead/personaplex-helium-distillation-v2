# SPDX-License-Identifier: MIT
"""The Bridge: the only new module standing between the student temporal
transformer and the frozen teacher depth transformer.

Bridge(x) stands in for the teacher's raw temporal-transformer output (before
`out_norm`), so everything downstream -- the teacher's `out_norm`, `text_linear`,
`depformer_in` (16 per-codebook projections), `depformer`, and `linears` -- is
reused completely unmodified. See distill/init_from_teacher.py for why the
bridge maps to the teacher's 4096-dim space rather than to `depformer_dim`
(1024): the latter would require bypassing/discarding the teacher's 16 frozen
per-codebook `depformer_in` projections.
"""

import torch
import torch.nn as nn

from moshi.modules.transformer import create_norm_fn


class Bridge(nn.Module):
    """Linear(in_dim -> out_dim) + RMSNorm, new and learned."""

    def __init__(self, in_dim: int, out_dim: int, device=None, dtype=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=False, device=device, dtype=dtype)
        # `create_norm_fn`'s "rms_norm_f32" branch pops any `dtype` kwarg and hardcodes
        # `torch.float` internally (by design, for numerical stability -- see
        # moshi/modules/transformer.py's create_norm_fn/RMSNorm, same as the teacher's own
        # norm1/norm2/out_norm), so `dtype` is not passed here. `device` is NOT similarly
        # defaulted, though: omitting it (as an earlier version of this file did) leaves
        # `self.norm.alpha` on the CPU while `self.linear` (and everything else in the
        # model) is on CUDA. That mismatch is invisible in eager execution (`alpha.to(var)`
        # inside RMSNorm's forward just does a slower cross-device cast) but is a hard
        # failure ("operation not permitted when stream is capturing") the moment this
        # runs inside LMGen's CUDAGraphed capture -- a cross-device transfer cannot be
        # captured into a CUDA graph, unlike the same-device dtype-only casts (bf16<->f32)
        # that RMSNorm does everywhere else in this model.
        self.norm = create_norm_fn("rms_norm_f32", out_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))
