# SPDX-License-Identifier: MIT
"""Loss terms for temporal-transformer distillation.

    L = alpha*CE + beta*KL + gamma*L_bridge + delta*L_hidden + epsilon*L_speaker

Per-codebook weights and temperatures, transition upweighting, and running-mean
normalization are all implemented here; `distill/schedule.py` owns how
alpha..epsilon and the depth-transformer freeze state vary over training phases.

Codebook indexing convention (matches `LMModel`: index 0 is text, 1..dep_q are
audio codebooks in delay order): `CODEBOOK_KL_WEIGHTS[0]` is the text weight,
`CODEBOOK_KL_WEIGHTS[1:]` the first 8 audio codebooks (cb1..cb8). The teacher's
`dep_q=16` audio codebooks are two interleaved 8-codebook streams (self/other
party, see moshi/models/lm.py); the same 8 weights are applied to both halves
since they represent the same underlying acoustic hierarchy for each party.
"""

from dataclasses import dataclass, field
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_KL_TEMPERATURE = 2.0
DEFAULT_AUDIO_KL_TEMPERATURE = 1.0  # configurable: audio distributions are already high-entropy

# [text, cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8]
CODEBOOK_KL_WEIGHTS = [1.0, 1.0, 0.8, 0.8, 0.7, 0.5, 0.4, 0.3, 0.25]


def _codebook_weights(num_audio_codebooks: int, device, dtype) -> torch.Tensor:
    """Expand the 8 acoustic-hierarchy weights to however many audio codebooks the
    depformer actually has (16, two 8-codebook streams), by tiling."""
    base = torch.tensor(CODEBOOK_KL_WEIGHTS[1:], device=device, dtype=dtype)
    reps = (num_audio_codebooks + len(base) - 1) // len(base)
    return base.repeat(reps)[:num_audio_codebooks]


def kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    weight: tp.Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """KL(teacher || student) at the given temperature, with the standard tau^2
    gradient-scale correction, reduced over the vocabulary and averaged over the
    (masked) batch/time/codebook positions. `weight` can be a per-codebook tensor
    broadcastable against `student_logits`'s codebook dimension.

    Shapes: student_logits/teacher_logits [B, K, T, card] (or [B, 1, T, card] for
    text), mask [B, K, T] (True where the position is valid).

    `_undelay_sequence` (moshi/models/lm.py) fills positions past the sequence's
    delay boundary with NaN and marks them invalid in `mask`; those NaNs are
    scrubbed here so they can't survive a `nan * 0.0 == nan` accident in the
    masked reduction below.
    """
    student_logits = torch.nan_to_num(student_logits, nan=0.0)
    teacher_logits = torch.nan_to_num(teacher_logits, nan=0.0)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    kl = kl * (temperature ** 2)
    kl = kl * weight
    mask = mask.to(kl.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (kl * mask).sum() / denom


def ce_loss(student_logits: torch.Tensor, targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Hard-label cross entropy, `student_logits` [B, K, T, card], `targets` [B, K, T]."""
    logits_flat = student_logits.reshape(-1, student_logits.shape[-1])
    targets_flat = targets.reshape(-1)
    return F.cross_entropy(logits_flat.float(), targets_flat, ignore_index=ignore_index)


@dataclass
class DistillationLossOutput:
    total: torch.Tensor
    ce: torch.Tensor
    kl_text: torch.Tensor
    kl_audio: torch.Tensor
    bridge: torch.Tensor
    hidden: torch.Tensor
    speaker: torch.Tensor
    raw: dict[str, float] = field(default_factory=dict)


class RunningNorm:
    """Normalizes a scalar loss term by a running mean of its own magnitude, so
    the phase weights (alpha..epsilon) control relative emphasis rather than
    fighting each term's raw scale. `momentum` close to 1 => slow-moving average.
    """

    def __init__(self, momentum: float = 0.99, eps: float = 1e-6):
        self.momentum = momentum
        self.eps = eps
        self._mean: tp.Optional[float] = None

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        v = float(value.detach().abs().item())
        # A single non-finite `value` (e.g. from a degenerate init or a bad batch) must not
        # permanently corrupt `self._mean` to NaN: the EMA update `momentum*mean + (1-momentum)*nan`
        # is NaN forever after, silently normalizing every FUTURE (otherwise-healthy) call by NaN
        # too -- unlike a single bad optimizer step (which a finite-loss check can just skip), this
        # state has no way to self-correct once corrupted. Skip the update and keep the last-known-
        # good mean instead; the caller is still responsible for not stepping the optimizer on a
        # non-finite total loss (see distill/train.py's finite check).
        if v == v and v not in (float("inf"), float("-inf")):  # finite check without importing math
            if self._mean is None:
                self._mean = v
            else:
                self._mean = self.momentum * self._mean + (1 - self.momentum) * v
        denom = (self._mean if self._mean is not None else 1.0) + self.eps
        return value / denom

    def state_dict(self) -> dict:
        return {"mean": self._mean, "momentum": self.momentum}

    def load_state_dict(self, state: dict):
        self._mean = state["mean"]
        self.momentum = state["momentum"]


def bridge_loss(student_bridge_out: torch.Tensor, teacher_raw_hidden: torch.Tensor) -> torch.Tensor:
    """MSE + cosine between the student's Bridge output and the teacher's raw
    (pre out_norm) temporal-transformer hidden state -- both dense, positionally
    aligned, teacher_dim-wide vectors."""
    mse = F.mse_loss(student_bridge_out.float(), teacher_raw_hidden.float())
    cos = 1.0 - F.cosine_similarity(student_bridge_out.float(), teacher_raw_hidden.float(), dim=-1).mean()
    return mse + cos


class HiddenProjection(nn.Module):
    """Per-mapped-layer student->teacher projection for L_hidden. Orthogonal init,
    meant to be excluded from weight decay, and discarded at export (it never
    participates in inference, only in the training-time loss).
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, student_hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(student_hidden)


def hidden_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    projection: HiddenProjection,
) -> torch.Tensor:
    """L2-normalize both sides after projecting student->teacher space, then
    MSE + cosine. Distills attention-block *outputs* (the residual stream at a
    mapped layer boundary), not attention matrices.
    """
    # `projection` is kept in fp32 (see distill/train.py); cast the (bf16) model
    # hidden state up front rather than relying on autocast.
    projected = projection(student_hidden.float())
    projected = F.normalize(projected, dim=-1)
    teacher = F.normalize(teacher_hidden.float(), dim=-1)
    mse = F.mse_loss(projected, teacher)
    cos = 1.0 - F.cosine_similarity(projected, teacher, dim=-1).mean()
    return mse + cos


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """1D max-pool dilation of a boolean/float mask along the time dimension.
    `mask`: [..., T]."""
    if radius <= 0:
        return mask
    m = mask.float()
    shape = m.shape
    m = m.reshape(-1, 1, shape[-1])
    kernel = 2 * radius + 1
    dilated = F.max_pool1d(m, kernel_size=kernel, stride=1, padding=radius)
    return dilated.reshape(shape)


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax(logits) along the last dim, normalized to [0, 1] by
    the maximum possible entropy (log of the vocab size)."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(logits.shape[-1]), device=logits.device))
    return entropy / max_entropy


def frame_weights(
    speaker_transition_mask: torch.Tensor,
    teacher_cb1_logits: torch.Tensor,
    radius: int = 8,
    transition_boost: float = 3.0,
    entropy_boost: float = 0.5,
) -> torch.Tensor:
    """w_frame = (1 + transition_boost * dilate(speaker_transition_mask, radius))
                 * (1 + entropy_boost * normalized_teacher_entropy(cb1))
    `speaker_transition_mask`: [B, T] boolean/float, 1 at frames flagged as a
    speaker turn-taking transition. `teacher_cb1_logits`: [B, T, card] teacher
    logits for the first acoustic codebook (the one most tied to turn-taking).
    """
    dilated = dilate_mask(speaker_transition_mask.float(), radius)
    entropy = normalized_entropy(teacher_cb1_logits)
    return (1.0 + transition_boost * dilated) * (1.0 + entropy_boost * entropy)


class WavLMSpeakerSimilarity:
    """WavLM-TDNN speaker cosine similarity, teacher vs. student audio decoded
    through the SAME frozen Mimi. Implemented as a validation metric (phase 1-3):
    call `.similarity(teacher_wav, student_wav)` -> per-utterance cosine in
    [-1, 1]. The differentiable (straight-through Gumbel) variant belongs to
    phase 4 only, and is NOT implemented here -- phase 4 should subclass or wrap
    this with a straight-through sampler around the frozen Mimi/depformer chain
    before backpropagating through it.

    Lazily loads `microsoft/wavlm-base-plus-sv` (WavLM + TDNN/x-vector head) via
    `transformers` on first use, since it's a heavy optional dependency not
    needed outside of evaluation.
    """

    MODEL_NAME = "microsoft/wavlm-base-plus-sv"
    SAMPLE_RATE = 16000

    def __init__(self, device: tp.Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)
        self._model = None
        self._feature_extractor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor

        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.MODEL_NAME)
        self._model = WavLMForXVector.from_pretrained(self.MODEL_NAME).to(self.device).eval()

    @torch.no_grad()
    def embed(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """`wav`: [B, T] or [T] mono float audio at `sample_rate`. Returns [B, D] embeddings."""
        self._load()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if sample_rate != self.SAMPLE_RATE:
            import torchaudio

            wav = torchaudio.functional.resample(wav, sample_rate, self.SAMPLE_RATE)
        inputs = self._feature_extractor(
            list(wav.cpu().numpy()), sampling_rate=self.SAMPLE_RATE, return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self._model(**inputs)
        return out.embeddings

    def similarity(self, teacher_wav: torch.Tensor, student_wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        teacher_emb = self.embed(teacher_wav, sample_rate)
        student_emb = self.embed(student_wav, sample_rate)
        return F.cosine_similarity(teacher_emb, student_emb, dim=-1)
