# SPDX-License-Identifier: MIT
"""Data-driven initialization of a `StudentLMModel` from a loaded teacher `LMModel`.

This is the part of the pipeline the plan calls out as mattering most: a
randomly-initialized 20-layer/2048-dim/GQA-4:1 student, dropped in front of a
frozen depth transformer it was never trained with, will not produce anything
resembling the teacher's discrete token distribution. Every step here exists to
give the student a head start that is at least structurally aligned with the
teacher before any gradient step is taken.

Because the teacher has no GQA (`num_key_value_heads == num_heads == 32`, plain
MHA -- see distill/gqa_attention.py's docstring), "mean-pool teacher KV heads
into groups" is implemented as: pool groups of *all* 32 teacher heads' K/V
projection rows into `num_key_value_heads` bands (no heads dropped), while
query heads are reduced by strided selection (matching plan's the intent that a
GQA-4:1 student should still see a broad, representative slice of the
teacher's 32 attention heads).

Steps, run in order by `initialize_student`:
  1. `compute_layer_scores`      -- score[l] = 1 - cos(h[l], h[l+1]) per teacher layer.
  2. `select_layers`             -- top-`num_layers` by score, always keeping the
                                     first/last `keep_first`/`keep_last`, order preserved.
  3. `compute_residual_projection` -- one global activation-aware SVD basis
                                     (4096 -> hidden_size) shared by every selected layer,
                                     the embeddings, and the bridge.
  4. per selected layer: `init_attention_layer` (head selection + KV pooling + residual
     projection) and `init_ffn_layer` (own per-layer SVD of the FFN's gated-hidden space).
  5. `init_embeddings`           -- project teacher embedding tables through the same
                                     residual projection.
  6. `init_bridge_least_squares` -- closed-form regression of the Bridge so the frozen
                                     depth transformer gets a roughly-correct input at step 0.
  7. `sanity_check_student`      -- coarse, automatable degeneracy checks. This is a
                                     heuristic gate, not a quality guarantee: LISTEN to the
                                     output before starting a real training run.
"""

from dataclasses import dataclass, field
import logging
import typing as tp

import torch
import torch.nn.functional as F

from moshi.models.lm import LMModel, AUDIO_TOKENS_PER_STREAM, SILENCE_TOKENS
from moshi.models.compression import MimiModel
from moshi.modules.transformer import StreamingTransformerLayer

from .student_model import StudentLMModel
from .bridge import Bridge

logger = logging.getLogger(__name__)

Batch = torch.Tensor  # [B, K, T] token sequences, teacher `LMModel.forward_train` layout


# --------------------------------------------------------------------------
# 1-2. Layer importance probing and selection
# --------------------------------------------------------------------------

@torch.no_grad()
def compute_layer_scores(teacher: LMModel, batches: tp.Sequence[Batch]) -> torch.Tensor:
    """score[l] = 1 - cos(h[l], h[l+1]) for l in [0, num_teacher_layers - 1], averaged
    over all calibration batches, batch and time. `h[0]` is the embedding output (the
    input to layer 0), `h[num_layers]` is the output of the last layer (pre out_norm).
    A high score means layer l changes its input a lot (a "load-bearing" layer); a low
    score means the layer is close to a no-op for this data distribution.
    """
    teacher.eval()
    layers = list(teacher.transformer.layers)
    num_layers = len(layers)
    captured: list[torch.Tensor] = []

    def make_hook(store):
        def hook(module, inputs, output):
            store.append(inputs[0].detach())
            store.append(output.detach())
        return hook

    handles = []
    for layer in layers:
        handles.append(layer.register_forward_hook(make_hook(captured)))

    score_sum = torch.zeros(num_layers, dtype=torch.float64)
    score_count = 0
    try:
        for batch in batches:
            captured.clear()
            codes = batch.to(teacher.device)
            teacher.forward_train(codes)
            # captured holds [in_0, out_0, in_1, out_1, ..., in_{L-1}, out_{L-1}]
            hiddens = [captured[0]] + [captured[2 * i + 1] for i in range(num_layers)]
            for l in range(num_layers):
                h_l = hiddens[l].float()
                h_l1 = hiddens[l + 1].float()
                cos = F.cosine_similarity(h_l, h_l1, dim=-1)
                score_sum[l] += (1.0 - cos).mean().double().item()
            score_count += 1
    finally:
        for h in handles:
            h.remove()

    assert score_count > 0, "No calibration batches provided."
    return (score_sum / score_count).float()


def select_layers(scores: torch.Tensor, num_layers: int, keep_first: int, keep_last: int) -> list[int]:
    total = scores.shape[0]
    assert num_layers <= total
    forced = set(range(keep_first)) | set(range(total - keep_last, total))
    assert len(forced) <= num_layers, (
        f"keep_first={keep_first} + keep_last={keep_last} exceeds num_layers={num_layers}"
    )
    remaining_slots = num_layers - len(forced)
    candidates = [l for l in range(total) if l not in forced]
    candidates.sort(key=lambda l: scores[l].item(), reverse=True)
    chosen = forced | set(candidates[:remaining_slots])
    return sorted(chosen)


# --------------------------------------------------------------------------
# 3. Activation-aware SVD projection (shared residual-stream basis)
# --------------------------------------------------------------------------

@dataclass
class Projection:
    """`down`: (teacher_dim, student_dim) encodes a teacher activation vector `t`
    into student space via `s = t @ down`. `up`: (student_dim, teacher_dim)
    reconstructs `t ~= s @ up`. `up @ down == I_{student_dim}` exactly (down's
    columns are an orthonormal SVD basis); `down @ up` is the (rank student_dim)
    projector onto the retained subspace, not identity -- that's the truncation.

    For adapting a teacher WEIGHT matrix `W` (F.linear convention, `y = x @ W^T`)
    rather than a plain activation vector, the two cases are NOT symmetric:
      - `W` reads from a reduced space (its `in` dim shrinks, e.g. attention
        in_proj, FFN linear_in wrt the residual): `W' = W @ up.T`.
      - `W` writes to a reduced space (its `out` dim shrinks, e.g. attention
        out_proj, FFN linear_out wrt the residual): `W' = down.T @ W`.
    Both are derived from `t ~= s @ up` substituted into `y = t @ W^T` /
    `y_reduced = y @ down` respectively -- see the two derivations inline in
    `init_attention_layer` and `init_ffn_layer` if this needs re-deriving.
    """
    down: torch.Tensor
    up: torch.Tensor
    rms: torch.Tensor


@torch.no_grad()
def _collect_activations(module_input_hook_target: torch.nn.Module,
                          run_fn: tp.Callable[[Batch], None],
                          batches: tp.Sequence[Batch]) -> torch.Tensor:
    """Run `run_fn(batch)` for each batch, capturing the input activations seen by
    `module_input_hook_target`, concatenated over batch/time into a single
    `[N, dim]` matrix."""
    captured: list[torch.Tensor] = []

    def hook(module, inputs, output):
        captured.append(inputs[0].detach().float().reshape(-1, inputs[0].shape[-1]))

    handle = module_input_hook_target.register_forward_hook(hook)
    try:
        for batch in batches:
            run_fn(batch)
    finally:
        handle.remove()
    return torch.cat(captured, dim=0)


def _gating_hidden_activation(gate: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Recomputes `ActivationGating`'s intermediate gated-hidden value (the
    input `linear_out` would see) from the gating module's own INPUT `x`.

    `gating_forward_kernel` (moshi/modules/gating.py) calls `F.linear` directly
    on `linear_in.weight` / `linear_out.weight` rather than invoking
    `linear_in`/`linear_out` as `nn.Module`s -- so a forward hook on
    `gate.linear_out` never fires (its `.forward()` is simply never called).
    Hooking the whole `gate` module's input (which DOES fire, since
    `StreamingTransformerLayer._ff_block` calls `self.gating(x)`) and
    replicating the first half of the kernel here is the only way to observe
    this intermediate value without modifying moshi/modules/gating.py.

    Casts `x` to `gate.linear_in.weight`'s dtype before the matmul regardless
    of what dtype it arrives in -- callers that capture activations via a
    forward hook commonly upcast to float32 for numerically stable statistics
    (e.g. `collect_ffn_hidden_activations` below), while the model itself runs
    in bf16, and `F.linear` requires both operands to match.
    """
    proj = F.linear(x.to(gate.linear_in.weight.dtype), gate.linear_in.weight)
    B, T, _ = proj.shape
    proj = proj.view(B, T, 2, -1)
    return (gate.activation(proj[..., 0, :]) * proj[..., 1, :]).float()


@torch.no_grad()
def collect_ffn_hidden_activations(
    gate: torch.nn.Module, run_fn: tp.Callable[[Batch], None], batches: tp.Sequence[Batch],
) -> torch.Tensor:
    """Like `_collect_activations`, but for the FFN's gated-hidden space
    specifically -- see `_gating_hidden_activation` for why this can't just be
    `_collect_activations(gate.linear_out, ...)`."""
    captured: list[torch.Tensor] = []

    def hook(module, inputs, output):
        captured.append(inputs[0].detach().float())

    handle = gate.register_forward_hook(hook)
    try:
        for batch in batches:
            run_fn(batch)
    finally:
        handle.remove()
    hidden = torch.cat([_gating_hidden_activation(gate, x) for x in captured], dim=0)
    return hidden.reshape(-1, hidden.shape[-1])


@torch.no_grad()
def compute_activation_projection(activations: torch.Tensor, student_dim: int) -> Projection:
    """Activation-aware SVD width reduction: scale by per-channel activation RMS
    before the SVD (so high-energy channels dominate which directions are kept),
    take the top `student_dim` eigenvectors of the SCALED covariance -- but,
    unlike a textbook ASVD writeup, do NOT fold the RMS scale back into
    `down`/`up` afterward. `down`/`up` are the raw orthonormal eigenvectors, so
    `down` stays (semi-)orthogonal (`down.T @ down == I` always; `down @ down.T
    == I` too at full rank, i.e. student_dim == teacher_dim).

    This matters because the residual stream in `transformer.py` passes through
    RMSNorm, whose normalization divides by a SCALAR (the per-row rms) rather
    than a per-channel one. That commutes with `down` cleanly only when `down`
    is norm-preserving, i.e. genuinely (semi-)orthogonal: `||x @ down||^2 = x @
    down @ down.T @ x^T`, which equals `||x||^2` exactly at full rank and is a
    legitimate (expected, trainable-away) truncation loss otherwise. Folding
    the RMS scale into `down`/`up` (as an early version of this function did)
    makes `down @ down.T = diag(rms^2)` instead of (approximately) `I` --
    RMSNorm no longer commutes with the projection even at full rank, which
    showed up as a large, spurious "reproduction error" in
    tests/test_init_sanity.py's equal-width losslessness check that had
    nothing to do with the actual attention/FFN weight-transform math. The
    activation RMS still does real work here: it decides which eigenvectors of
    the covariance are worth keeping (high-activation-energy channels dominate
    the ranking), it's just not baked into the basis vectors themselves.
    """
    teacher_dim = activations.shape[-1]
    assert student_dim <= teacher_dim
    rms = activations.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)  # [teacher_dim]
    scaled = activations * rms  # activation-aware weighting, not a whitening normalize
    # Eigendecomposition of the (teacher_dim x teacher_dim) covariance is far cheaper
    # than an SVD of the (N x teacher_dim) activation matrix for large N.
    cov = (scaled.T @ scaled) / scaled.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending order
    order = torch.argsort(eigvals, descending=True)
    v_k = eigvecs[:, order[:student_dim]]  # [teacher_dim, student_dim], orthonormal columns

    down = v_k        # [teacher_dim, student_dim]
    up = v_k.T         # [student_dim, teacher_dim]
    return Projection(down=down, up=up, rms=rms)


# --------------------------------------------------------------------------
# 4a. Attention head selection / GQA KV pooling
# --------------------------------------------------------------------------

def select_query_heads(num_teacher_heads: int, num_student_heads: int) -> list[int]:
    """Evenly-spaced strided selection across all teacher heads, so the student's
    (fewer) query heads still cover the full range of what the teacher learned,
    rather than e.g. always taking the first N. Integer floor spacing guarantees
    exactly `num_student_heads` distinct indices (unlike rounding a linspace,
    which can collide)."""
    assert num_student_heads <= num_teacher_heads
    return [(i * num_teacher_heads) // num_student_heads for i in range(num_student_heads)]


def kv_head_bands(num_teacher_heads: int, num_student_kv_heads: int) -> list[list[int]]:
    """Contiguous bands over ALL teacher heads (none dropped), one band per pooled
    student KV head. `num_teacher_heads` must be divisible by `num_student_kv_heads`.
    """
    assert num_teacher_heads % num_student_kv_heads == 0
    band_size = num_teacher_heads // num_student_kv_heads
    return [list(range(g * band_size, (g + 1) * band_size)) for g in range(num_student_kv_heads)]


def _is_identity_projection(proj: Projection, atol: float = 1e-5) -> bool:
    """True if `proj.down` is (numerically) the identity matrix -- i.e. no actual
    width reduction or basis rotation is happening (e.g. tests/test_bit_exactness.py's
    identity-clone student, or any config where student_dim == teacher_dim and no
    calibration-driven rotation was computed). Used to decide whether a teacher's
    RMSNorm scale (`alpha`) can be copied to the student directly (exact per-channel
    correspondence) instead of reset to a neutral ones-vector.
    """
    if proj.down.shape[0] != proj.down.shape[1]:
        return False
    eye = torch.eye(proj.down.shape[0], device=proj.down.device, dtype=proj.down.dtype)
    return torch.allclose(proj.down, eye, atol=atol)


def _init_norm_alpha(teacher_norm: torch.nn.Module, student_norm: torch.nn.Module, proj: Projection):
    """Sets `student_norm.alpha` from `teacher_norm.alpha` when the residual basis is
    unchanged (identity projection), or resets it to ones otherwise.

    RMSNorm normalizes by a SCALAR (`torch.mean(x**2)`, not per-channel) before
    applying `alpha` elementwise. That means `alpha` does not transfer through an
    arbitrary change of basis: if `down` genuinely rotates or truncates the residual
    space, `student_norm(x @ down)` computed with the teacher's `alpha` values (which
    were learned for teacher-basis channels) would apply the WRONG scale to WRONG
    channels post-rotation, in general. Resetting to ones there is the safe, neutral
    choice -- P1 "Align" training calibrates it quickly.

    But when `down` is the identity (no rotation at all -- see `_is_identity_projection`),
    each student channel IS the corresponding teacher channel, so the teacher's `alpha`
    is exactly correct and resetting it to ones would be a real, unnecessary numerical
    regression: this was caught by tests/test_bit_exactness.py's identity-clone check
    (a real 7B teacher's alpha values are trained, non-uniform -- forcing them to ones
    produced a large, spurious "reproduction failure" that had nothing to do with the
    actual attention/FFN weight-transform math).
    """
    if _is_identity_projection(proj):
        student_norm.alpha.data.copy_(teacher_norm.alpha.data.to(student_norm.alpha.dtype))
    else:
        torch.nn.init.ones_(student_norm.alpha)


@torch.no_grad()
def init_attention_layer(
    teacher_layer: StreamingTransformerLayer,
    student_layer,  # GQAStreamingTransformerLayer
    q_head_idx: list[int],
    kv_bands: list[list[int]],
    proj: Projection,
    teacher_head_dim: int = 128,
):
    attn_t = teacher_layer.self_attn
    attn_s = student_layer.self_attn
    num_teacher_heads = attn_t.num_heads
    d_in = attn_t.in_proj_weight.shape[1]
    assert attn_t.in_proj_weight.shape[0] == 3 * num_teacher_heads * teacher_head_dim

    w_qkv = attn_t.in_proj_weight.view(3, num_teacher_heads, teacher_head_dim, d_in)
    w_q, w_k, w_v = w_qkv[0], w_qkv[1], w_qkv[2]  # each [num_teacher_heads, head_dim, d_in]

    down = proj.down.to(w_q.dtype)
    up_t = proj.up.to(w_q.dtype).T  # [teacher_dim, student_dim]

    # Query/Key/Value all READ from the residual (it's their `in` dimension), so per
    # the Projection docstring this is the `W @ up.T` case, not `W @ down`.
    w_q_sel = w_q[q_head_idx].reshape(len(q_head_idx) * teacher_head_dim, d_in)
    attn_s.q_proj.weight.copy_((w_q_sel @ up_t).to(attn_s.q_proj.weight.dtype))

    # KV: mean-pool each band over ALL teacher heads (no heads dropped), then reduce input dim.
    def pooled(w):
        bands = [w[band].mean(dim=0) for band in kv_bands]  # each [head_dim, d_in]
        return torch.cat(bands, dim=0)  # [num_kv_heads * head_dim, d_in]

    w_k_pooled = pooled(w_k)
    w_v_pooled = pooled(w_v)
    attn_s.k_proj.weight.copy_((w_k_pooled @ up_t).to(attn_s.k_proj.weight.dtype))
    attn_s.v_proj.weight.copy_((w_v_pooled @ up_t).to(attn_s.v_proj.weight.dtype))

    # Output projection WRITES to the residual (it's its `out` dimension): the
    # `down.T @ W` case. Select the same query heads on the input (concat) side first.
    w_out_t = attn_t.out_proj.weight  # [teacher_dim, num_teacher_heads*head_dim]
    w_out_t = w_out_t.view(d_in, num_teacher_heads, teacher_head_dim)
    w_out_sel = w_out_t[:, q_head_idx, :].reshape(d_in, len(q_head_idx) * teacher_head_dim)
    down_T = down.T  # [student_dim, teacher_dim]
    w_out_student = down_T @ w_out_sel  # [student_dim, len(q_head_idx)*head_dim]
    attn_s.out_proj.weight.copy_(w_out_student.to(attn_s.out_proj.weight.dtype))

    _init_norm_alpha(teacher_layer.norm1, student_layer.norm1, proj)


# --------------------------------------------------------------------------
# 4b. FFN (gated) width reduction: residual dim via `proj` (a rotation, valid
#     since residual reads/writes are linear); hidden dim via CHANNEL
#     SELECTION, not an SVD rotation -- see init_ffn_layer docstring for why.
# --------------------------------------------------------------------------

def select_ffn_channels(hidden_activations: torch.Tensor, student_hidden: int) -> torch.Tensor:
    """Which of the teacher's `teacher_hidden` gated-FFN channels to keep, by
    activation RMS (highest-energy channels first, then sorted back into their
    original order so `linear_in`'s two halves and `linear_out`'s columns stay
    consistently indexed)."""
    rms = hidden_activations.pow(2).mean(dim=0).sqrt()
    idx = torch.argsort(rms, descending=True)[:student_hidden]
    return torch.sort(idx).values


@torch.no_grad()
def init_ffn_layer(
    teacher_layer: StreamingTransformerLayer,
    student_layer,  # GQAStreamingTransformerLayer
    proj: Projection,
    hidden_activations: torch.Tensor,
):
    """`hidden_activations` are the FFN's gated-hidden values (post SiLU-gate *
    value, see `_gating_hidden_activation`) -- i.e. `linear_out`'s true input.

    Unlike the residual stream, the FFN hidden dimension sits behind an
    ELEMENTWISE nonlinearity (`activation(gate) * value`, per channel). An SVD
    rotation of that space (as an earlier version of this function did, mapping
    it through a `compute_activation_projection`-style basis, symmetric to how
    the residual is handled) is mathematically wrong: a rotation mixes
    channels together BEFORE the nonlinearity is (conceptually) re-applied,
    but the actual elementwise nonlinearity in `gating_forward_kernel` only
    ever sees the ORIGINAL, unrotated channel basis -- `activation(x @ R) !=
    activation(x) @ R` for a rotation `R` in general. This showed up as a
    large (~0.6 max-abs-logit), layer-compounding numerical error in
    tests/test_init_sanity.py that was NOT present in the attention path
    (which stays entirely linear, aside from softmax's normalization, and
    doesn't touch this hidden space at all).

    The correct reduction here is the same idea already used for attention
    heads: SELECT a subset of the teacher's hidden channels (by activation
    RMS) rather than rotating into a new basis. Selection preserves the
    elementwise correspondence exactly -- each kept channel is still
    `activation(gate_c) * value_c` for the SAME original channel `c`, just
    fewer of them.
    """
    gate_t = teacher_layer.gating
    gate_s = student_layer.gating
    student_hidden = gate_s.linear_out.weight.shape[1]
    teacher_hidden = gate_t.linear_out.weight.shape[1]

    channel_idx = select_ffn_channels(hidden_activations, student_hidden)
    down_r = proj.down.to(gate_t.linear_in.weight.dtype)
    up_r = proj.up.to(gate_t.linear_in.weight.dtype)

    # linear_in: [2*teacher_hidden, teacher_dim] -> [2*student_hidden, student_dim].
    # READS the residual (`up_r.T`, Case A); selects the same `student_hidden` output
    # channels in BOTH halves (gate / value, see moshi/modules/gating.py), since hidden
    # channel c = activation(gate[c]) * value[c] depends only on row c of each half.
    w_in = gate_t.linear_in.weight.view(2, teacher_hidden, -1)[:, channel_idx, :]
    w_in_reduced = torch.stack([w_in[i] @ up_r.T for i in range(2)], dim=0)
    gate_s.linear_in.weight.copy_(w_in_reduced.reshape(2 * student_hidden, -1).to(gate_s.linear_in.weight.dtype))

    # linear_out: [teacher_dim, teacher_hidden] -> [student_dim, student_hidden].
    # Selects the same channels on its input (columns), WRITES the residual (`down_r.T`, Case B).
    w_out_sel = gate_t.linear_out.weight[:, channel_idx]
    w_out_reduced = down_r.T @ w_out_sel
    gate_s.linear_out.weight.copy_(w_out_reduced.to(gate_s.linear_out.weight.dtype))

    _init_norm_alpha(teacher_layer.norm2, student_layer.norm2, proj)


# --------------------------------------------------------------------------
# 5. Embeddings
# --------------------------------------------------------------------------

@torch.no_grad()
def init_embeddings(teacher: LMModel, student: StudentLMModel, proj: Projection):
    down = proj.down.to(teacher.text_emb.weight.dtype)
    student.text_emb.weight.copy_((teacher.text_emb.weight @ down).to(student.text_emb.weight.dtype))
    for cb in range(student.n_q):
        student.emb[cb].weight.copy_(
            (teacher.emb[cb].weight @ down).to(student.emb[cb].weight.dtype)
        )


# --------------------------------------------------------------------------
# 6. Bridge: closed-form least-squares fit
# --------------------------------------------------------------------------

@torch.no_grad()
def init_bridge_least_squares(student: StudentLMModel, teacher: LMModel, batches: tp.Sequence[Batch],
                               ridge: float = 1e-3):
    """Regress the Bridge so that `bridge(student_raw_hidden) ~= teacher_raw_hidden`
    (both pre-`out_norm`), giving the frozen depth transformer a roughly correct
    input from step 0 rather than from random init.

    Uses RIDGE (L2-regularized) least squares rather than a raw `torch.linalg.lstsq`
    solve. The system has `student_dim` unknowns per output column (e.g. 2048 for
    student_ppx_s); a realistic calibration set (a handful of batches, particularly
    during a quick/smoke-test run) can easily provide fewer than `student_dim`
    effective rows, or a near-rank-deficient covariance even with more. A raw lstsq
    solve in that regime is not guaranteed to be well-behaved -- this surfaced as an
    actually non-finite ("inf" residual) solution in practice, which then poisoned
    every OTHER parameter's gradient a few training steps later via
    `clip_grad_norm_` (a single non-finite gradient makes the whole-model clip
    coefficient non-finite, corrupting the entire model on that optimizer step).
    Ridge regression is unconditionally well-posed for `ridge > 0`: adding
    `ridge * I` to `X^T X` makes it positive-definite even when `X` itself is
    rank-deficient, at the cost of a small bias toward a smaller-norm solution.
    """
    student.eval()
    teacher.eval()
    xs, ys = [], []
    for batch in batches:
        codes = batch.to(student.device)
        teacher_hidden = teacher.transformer(teacher.embed_codes(codes[:, :, :-1]))
        student_hidden = student.transformer(student.embed_codes(codes[:, :, :-1]))
        xs.append(student_hidden.float().reshape(-1, student_hidden.shape[-1]))
        ys.append(teacher_hidden.float().reshape(-1, teacher_hidden.shape[-1]))
    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)

    student_dim = X.shape[-1]
    xtx = X.T @ X
    xty = X.T @ Y
    reg = ridge * torch.diagonal(xtx).mean().clamp_min(1e-6)
    xtx_reg = xtx + reg * torch.eye(student_dim, device=X.device, dtype=X.dtype)
    solution = torch.linalg.solve(xtx_reg, xty)  # [student_dim, teacher_dim]

    if not torch.isfinite(solution).all():
        logger.warning(
            "Bridge least-squares fit produced non-finite values even with ridge "
            "regularization -- falling back to a zero-initialized bridge linear weight "
            "(P1 'Align' training will need to learn it from scratch). This usually means "
            "the calibration data itself contains NaN/Inf (check your dataset), not just "
            "an ill-conditioned fit."
        )
        solution = torch.zeros_like(solution)

    student.bridge.linear.weight.copy_(solution.T.to(student.bridge.linear.weight.dtype))

    residual_rms = Y.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
    student.bridge.norm.alpha.data.copy_(
        residual_rms.view(1, 1, -1).to(student.bridge.norm.alpha.dtype)
    )
    return {"bridge_lstsq_residual": (X @ solution - Y).pow(2).mean().sqrt().item()}


# --------------------------------------------------------------------------
# 7. Zero-shot sanity check
# --------------------------------------------------------------------------

@dataclass
class SanityReport:
    passed: bool
    silence_frame_fraction: float
    unique_token_fraction: float
    nan_or_inf: bool
    notes: list[str] = field(default_factory=list)


@torch.no_grad()
def sanity_check_student(
    student: StudentLMModel,
    mimi: MimiModel,
    voice_prompt_path: str,
    persona_text_tokens: tp.Optional[list[int]],
    num_frames: int = 200,
    max_silence_fraction: float = 0.9,
    min_unique_token_fraction: float = 0.02,
) -> SanityReport:
    """Coarse, automatable degeneracy checks -- NOT a quality guarantee. Catches
    the failure mode the plan warns about ("if it outputs noise, stop and report")
    where the student collapses to constant silence/garbage tokens; it does not
    replace listening to the output. Thresholds here are heuristic guards against
    total collapse, not benchmarked quality targets.

    Mirrors the exact prompt-loading flow used by `moshi.offline.run_inference`
    (voice prompt -> silence -> text prompt -> silence, via `LMGen.step_system_prompts`),
    then lets the student free-run for `num_frames` with a placeholder ("sine") input
    on the other-party channel, same convention used during prompt loading.

    `voice_prompt_path` must be raw audio (.wav), not a `.pt` embedding cache -- those
    are tied to whichever model produced them (typically the teacher) and are not valid
    for the student; see the assertion below for why.
    """
    # LMGen asserts the model is not in training mode; nn.Module defaults to training=True,
    # so don't depend on the caller having called .eval() first (build_student_lm does, but a
    # freshly-constructed StudentLMModel passed in directly would otherwise hit that assertion).
    was_training = student.training
    student.eval()
    try:
        return _sanity_check_student_impl(
            student, mimi, voice_prompt_path, persona_text_tokens, num_frames,
            max_silence_fraction, min_unique_token_fraction,
        )
    finally:
        student.train(was_training)


@torch.no_grad()
def _sanity_check_student_impl(
    student: StudentLMModel,
    mimi: MimiModel,
    voice_prompt_path: str,
    persona_text_tokens: tp.Optional[list[int]],
    num_frames: int,
    max_silence_fraction: float,
    min_unique_token_fraction: float,
) -> SanityReport:
    from moshi.models.lm import LMGen

    # `.pt` voice-prompt files (e.g. the shipped voices.tgz's NATF2.pt) are PRE-COMPUTED
    # embeddings -- `LMGen.load_voice_prompt_embeddings` skips audio encoding and Mimi
    # entirely, replaying `state["embeddings"]` straight into `step_embeddings`. Those
    # tensors were produced by whichever model originally called
    # `save_voice_prompt_embeddings=True` (the teacher, at its 4096-dim embed_codes output)
    # -- they are not just "a voice", they're a specific model's embedding space, and feeding
    # them into the student's differently-shaped transformer fails with a shape mismatch deep
    # inside the first RMSNorm rather than at this obviously-wrong call site. Always use raw
    # audio (.wav) for the student, so it computes its own embeddings via its own embed_codes.
    assert not voice_prompt_path.endswith(".pt"), (
        f"{voice_prompt_path} is a pre-computed embedding cache (produced by whichever model "
        "called save_voice_prompt_embeddings=True, typically the teacher) -- it cannot be "
        "reused for the student, which has a different embed_codes output width. Pass a raw "
        "audio (.wav) voice prompt instead, so the student computes its own embeddings."
    )

    notes: list[str] = []
    lm_gen = LMGen(student, device=student.device, use_sampling=True,
                   audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
                   sample_rate=mimi.sample_rate, frame_rate=mimi.frame_rate)
    with lm_gen.streaming(1), mimi.streaming(1):
        mimi.reset_streaming()
        lm_gen.reset_streaming()
        lm_gen.load_voice_prompt(voice_prompt_path)
        lm_gen.text_prompt_tokens = persona_text_tokens
        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()

        generated_codes = []
        for _ in range(num_frames):
            tokens = lm_gen.step(input_tokens=lm_gen._encode_sine_frame())
            if tokens is not None:
                generated_codes.append(tokens[:, 1:1 + AUDIO_TOKENS_PER_STREAM])

    if not generated_codes:
        return SanityReport(False, 1.0, 0.0, False, ["No frames were generated at all."])

    codes = torch.cat(generated_codes, dim=-1)  # [1, 8, T]
    silence = torch.as_tensor(SILENCE_TOKENS, device=codes.device).view(1, 8, 1)
    silence_fraction = (codes == silence).all(dim=1).float().mean().item()
    unique_fraction = codes.unique().numel() / codes.numel()

    audio = mimi.decode(codes)
    nan_or_inf = bool(torch.isnan(audio).any() or torch.isinf(audio).any())

    if silence_fraction > max_silence_fraction:
        notes.append(f"{silence_fraction:.1%} of frames are exact silence tokens (collapse suspected).")
    if unique_fraction < min_unique_token_fraction:
        notes.append(f"Only {unique_fraction:.1%} of emitted tokens are unique (collapse suspected).")
    if nan_or_inf:
        notes.append("Decoded audio contains NaN/Inf.")

    passed = not notes
    if not passed:
        logger.warning("Student sanity check FAILED: %s. Listen to the output before training.", notes)
    return SanityReport(passed, silence_fraction, unique_fraction, nan_or_inf, notes)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

@torch.no_grad()
def initialize_student(
    student: StudentLMModel,
    teacher: LMModel,
    calibration_batches: tp.Sequence[Batch],
    keep_first: int = 3,
    keep_last: int = 3,
) -> dict:
    """Run the full initialization pipeline (steps 1-6 above) in place on `student`.
    Returns a dict of diagnostics (layer scores, chosen layers, bridge residual)
    for logging -- run `sanity_check_student` separately afterwards.
    """
    teacher.eval()
    student.eval()
    num_student_layers = len(student.transformer.layers)

    scores = compute_layer_scores(teacher, calibration_batches)
    selected = select_layers(scores, num_student_layers, keep_first, keep_last)
    logger.info("Selected teacher layers: %s (scores=%s)", selected, [round(scores[l].item(), 4) for l in selected])

    def run_teacher(batch: Batch):
        teacher.forward_train(batch.to(teacher.device))

    residual_acts = _collect_activations(teacher.out_norm, run_teacher, calibration_batches)
    proj = compute_activation_projection(residual_acts, student.student_config.hidden_size)

    num_teacher_heads = teacher.transformer.layers[0].self_attn.num_heads
    q_head_idx = select_query_heads(num_teacher_heads, student.student_config.num_attention_heads)
    kv_bands = kv_head_bands(num_teacher_heads, student.student_config.num_key_value_heads)

    # Re-running the full teacher forward pass per selected layer is `num_layers`x
    # more calibration compute than strictly necessary (one pass with all hooks
    # attached at once would do), but this is a one-time init cost over
    # `num_calibration_batches` batches, and the simplicity is worth it here.
    for student_idx, teacher_idx in enumerate(selected):
        teacher_layer = teacher.transformer.layers[teacher_idx]
        student_layer = student.transformer.layers[student_idx]
        init_attention_layer(teacher_layer, student_layer, q_head_idx, kv_bands, proj)

        hidden_acts = collect_ffn_hidden_activations(teacher_layer.gating, run_teacher, calibration_batches)
        init_ffn_layer(teacher_layer, student_layer, proj, hidden_acts)

    init_embeddings(teacher, student, proj)
    bridge_diag = init_bridge_least_squares(student, teacher, calibration_batches)

    return {
        "selected_layers": selected,
        "layer_scores": scores.tolist(),
        **bridge_diag,
    }
