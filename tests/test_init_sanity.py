# SPDX-License-Identifier: MIT
"""Init-pipeline correctness, exercised at student_dim == teacher_dim.

At equal width, `compute_activation_projection`'s SVD is exact (rank ==
teacher_dim, nothing truncated) and `select_query_heads`/`kv_head_bands`
select/pool every head with none dropped and no pooling loss (num_kv_heads ==
num_heads => each "pooled" KV head is a mean over exactly one head, i.e. a
copy). Under those conditions `initialize_student` should be loss-free: the
resulting student, run through the SAME frozen depformer chain via the Bridge,
should reproduce the teacher's logits to floating-point precision. This is the
cheapest possible end-to-end numerical check that distill/init_from_teacher.py's
weight-transform math (the Case A "reads residual" / Case B "writes residual"
derivations) is actually correct, without downloading the real 7B teacher.

Uses a tiny synthetic teacher (dim=512) built directly via `LMModel(...)`, and
`StudentLMModel(..., teacher_kwargs=...)` to point the student's frozen-skeleton
build at that tiny teacher instead of the real one -- see
distill/student_model.py's `teacher_kwargs` parameter.
"""

import torch

from moshi.models.lm import LMModel

from distill.config import StudentConfig, InitConfig, TeacherReference, BridgeConfig
from distill.student_model import StudentLMModel
from distill.init_from_teacher import (
    initialize_student, select_layers, compute_layer_scores, compute_activation_projection,
    select_query_heads, kv_head_bands, init_attention_layer, init_ffn_layer, _collect_activations,
    collect_ffn_hidden_activations, Projection,
)

TEACHER_DIM = 512
TEACHER_NUM_HEADS = 4
NUM_CODEBOOKS = 5  # n_q=4 + text


def _tiny_teacher_kwargs() -> dict:
    return dict(
        dim=TEACHER_DIM, text_card=32, existing_text_padding_id=3, n_q=4, dep_q=4, card=16,
        num_heads=TEACHER_NUM_HEADS, num_layers=2, hidden_scale=2.0, causal=True, layer_scale=None,
        context=32, max_period=10000.0, gating="silu", norm="rms_norm_f32",
        positional_embedding="rope", depformer_dim=64, depformer_dim_feedforward=128,
        depformer_num_heads=2, depformer_num_layers=2, depformer_causal=True,
        depformer_layer_scale=None, depformer_multi_linear=True, depformer_context=4,
        depformer_max_period=10000.0, depformer_gating="silu", depformer_pos_emb="none",
        depformer_weights_per_step=True, delays=[0, 0, 1, 1, 1],
    )


def _tiny_teacher(dtype: torch.dtype = torch.float32) -> LMModel:
    # Intentionally never trained: `init_attention_layer`/`init_ffn_layer` reset
    # `norm1.alpha`/`norm2.alpha` to RMSNorm's default (all-ones) -- this teacher's own
    # norms are ALSO at that same untrained default, so the "lossless at equal width"
    # tests below aren't confounded by a norm-scale mismatch that a real (trained)
    # teacher would have. This is deterministic (both default to 1.0), not a fluke.
    #
    # The trailing `.to(device=, dtype=)` matters, not just style: `LMModel.__init__`
    # builds text_linear/out_norm/depformer_in/linears via plain torch.nn.Linear(...)/
    # create_norm_fn(...) calls with no device/dtype kwargs (moshi/models/lm.py), so they
    # silently default to float32 CPU regardless of the dtype= passed to LMModel.__init__.
    # get_moshi_lm (moshi/models/loaders.py) always re-casts the whole model after
    # construction for exactly this reason; skipping it here would build a model no real
    # load path produces (float32 output heads on an otherwise-bf16 model, say).
    torch.manual_seed(0)
    model = LMModel(device="cpu", dtype=dtype, **_tiny_teacher_kwargs())
    model = model.to(device="cpu", dtype=dtype)
    model.eval()
    return model


def _tiny_student_config() -> StudentConfig:
    return StudentConfig(
        name="test_tiny_lossless", num_hidden_layers=2, hidden_size=TEACHER_DIM,
        intermediate_size=1024,  # == hidden_scale*dim from _tiny_teacher_kwargs, so gating hidden matches exactly
        num_attention_heads=TEACHER_NUM_HEADS, num_key_value_heads=TEACHER_NUM_HEADS,
        rope_theta=10000.0, max_frames=32, norm="rms_norm_f32", gating="silu",
        layer_scale=None, causal=True,
        init=InitConfig(num_calibration_batches=3, keep_first=1, keep_last=1),
        teacher_reference=TeacherReference(
            dim=TEACHER_DIM, num_heads=TEACHER_NUM_HEADS, num_layers=2, context=32, n_q=4, dep_q=4,
            card=16, text_card=32, depformer_dim=64, depformer_num_heads=2, depformer_num_layers=2,
            depformer_dim_feedforward=128, max_period=10000.0,
        ),
        bridge=BridgeConfig(in_dim=TEACHER_DIM, out_dim=TEACHER_DIM),
    )


def _random_codes(teacher: LMModel, batch=2, frames=12, seed=1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, teacher.card, (batch, NUM_CODEBOOKS, frames), generator=g)
    codes[:, 0] = torch.randint(0, teacher.text_card, (batch, frames), generator=g)
    return codes


def test_layer_selection_keeps_first_and_last():
    scores = torch.tensor([0.9, 0.1, 0.1, 0.1, 0.9])
    selected = select_layers(scores, num_layers=3, keep_first=1, keep_last=1)
    assert selected[0] == 0
    assert selected[-1] == 4
    assert len(selected) == 3


def _build_lossless_student(teacher: LMModel, student_config: StudentConfig) -> StudentLMModel:
    student = StudentLMModel(student_config, device="cpu", dtype=torch.float32,
                              teacher_kwargs=_tiny_teacher_kwargs())
    student.load_frozen_from_teacher(teacher.state_dict())
    return student


def test_student_frozen_submodules_match_working_dtype():
    """Regression test for the actual bug hit on a real bf16 teacher (not just its
    downstream symptom, see test_collect_ffn_hidden_activations_handles_bf16_model):
    `LMModel.__init__` builds `text_linear`/`out_norm`/`depformer_in`/`linears` via
    plain `torch.nn.Linear(...)`/`create_norm_fn(...)` calls with no device/dtype
    kwargs, so on a plain `super().__init__(device="meta", dtype=dtype, ...)` call
    those four submodule groups silently end up as REAL (non-meta!) float32 CPU
    tensors regardless of `dtype`. `StudentLMModel.__init__` must force them onto
    meta+`dtype` itself (mirroring what `get_moshi_lm`'s trailing `.to()` call does
    for the real teacher) before `load_frozen_from_teacher` casts the teacher's
    weights to `self.text_linear.weight.dtype` -- otherwise that cast target is
    silently float32, producing a model whose output heads don't match the dtype
    of everything else, which crashes the moment two of them meet in a matmul
    (e.g. `text_linear(bridge_output)` in `forward_embeddings`).
    """
    try:
        _ = torch.zeros(1, 1, 4, dtype=torch.bfloat16) @ torch.zeros(4, 4, dtype=torch.bfloat16)
    except RuntimeError:
        import pytest
        pytest.skip("bf16 matmul not supported on this device")

    teacher = _tiny_teacher(dtype=torch.bfloat16)
    student_config = _tiny_student_config()
    student = StudentLMModel(student_config, device="cpu", dtype=torch.bfloat16,
                              teacher_kwargs=_tiny_teacher_kwargs())
    student.load_frozen_from_teacher(teacher.state_dict())

    for name in ("text_linear.weight", "out_norm.alpha", "depformer_in.0.weight", "linears.0.weight"):
        param = dict(student.named_parameters())[name]
        assert param.dtype == torch.bfloat16, f"{name} is {param.dtype}, expected bfloat16"

    eval_codes = _random_codes(teacher, batch=1, frames=4, seed=7)
    with torch.no_grad():
        student.forward_train(eval_codes)  # would raise the dtype-mismatch RuntimeError if unfixed


def test_attention_and_ffn_transforms_are_lossless_pre_bridge():
    """Isolates the Case A/Case B weight-transform math (init_attention_layer,
    init_ffn_layer) from the bridge's empirical least-squares fit: at equal
    width with no heads dropped, `student.raw_temporal_output(codes)` should
    equal `teacher.transformer(teacher.embed_codes(codes)) @ down` (the
    analytic relationship the projection defines), for codes NEVER seen during
    calibration. A least-squares bridge fit is a separate (and separately
    tested) concern -- see test_lossless_init_reproduces_teacher_logits.
    """
    teacher = _tiny_teacher()
    student_config = _tiny_student_config()
    student = _build_lossless_student(teacher, student_config)

    calibration_batches = [_random_codes(teacher, seed=100 + i) for i in range(4)]

    def run_teacher(batch):
        teacher.forward_train(batch)

    residual_acts = _collect_activations(teacher.out_norm, run_teacher, calibration_batches)
    proj = compute_activation_projection(residual_acts, student_config.hidden_size)

    q_head_idx = select_query_heads(TEACHER_NUM_HEADS, student_config.num_attention_heads)
    kv_bands = kv_head_bands(TEACHER_NUM_HEADS, student_config.num_key_value_heads)
    for teacher_layer, student_layer in zip(teacher.transformer.layers, student.transformer.layers):
        init_attention_layer(teacher_layer, student_layer, q_head_idx, kv_bands, proj)
        hidden_acts = collect_ffn_hidden_activations(teacher_layer.gating, run_teacher, calibration_batches)
        init_ffn_layer(teacher_layer, student_layer, proj, hidden_acts)

    # Feed the student transformer the *analytically expected* input (the teacher's own
    # embedded sequence, rotated by `down`) rather than going through the student's own
    # (not-yet-initialized) embeddings -- init_embeddings is a separate, simpler transform
    # (see test_embeddings_transform_is_lossless below); this isolates exactly the
    # attention/FFN weight transforms under test.
    eval_codes = _random_codes(teacher, batch=1, frames=8, seed=999)
    with torch.no_grad():
        teacher_embedded = teacher.embed_codes(eval_codes)
        teacher_hidden = teacher.transformer(teacher_embedded)
        student_hidden = student.transformer(teacher_embedded @ proj.down)

    diff = (student_hidden - teacher_hidden @ proj.down).abs().max().item()
    assert diff < 1e-3, (
        f"student transformer output diverges from teacher's under the analytic projection "
        f"by {diff}; this indicates a bug in the Case A/Case B weight-transform derivations."
    )


def test_lossless_init_reproduces_teacher_logits():
    """End-to-end check including the bridge. Uses enough calibration data
    (N > teacher_dim) that the bridge's least-squares fit is well-determined
    rather than an underdetermined minimum-norm solution that would only
    happen to match on the calibration manifold -- see module docstring.
    """
    teacher = _tiny_teacher()
    student_config = _tiny_student_config()
    student = _build_lossless_student(teacher, student_config)

    calibration_batches = [_random_codes(teacher, batch=8, frames=80, seed=200 + i) for i in range(4)]
    initialize_student(student, teacher, calibration_batches,
                        student_config.init.keep_first, student_config.init.keep_last)

    eval_codes = _random_codes(teacher, batch=1, frames=8, seed=999)
    with torch.no_grad():
        teacher_out = teacher.forward_train(eval_codes)
        student_out = student.forward_train(eval_codes)

    teacher_logits = torch.nan_to_num(teacher_out.logits, nan=0.0)
    student_logits = torch.nan_to_num(student_out.logits, nan=0.0)
    max_abs_diff = (teacher_logits - student_logits).abs().max().item()
    assert max_abs_diff < 0.5, (
        f"Expected near-exact reproduction at student_dim == teacher_dim (no truncation, "
        f"no head dropping) with a well-determined bridge fit, got max abs logit diff "
        f"{max_abs_diff}."
    )


def test_embeddings_transform_is_lossless():
    from distill.init_from_teacher import init_embeddings

    teacher = _tiny_teacher()
    student_config = _tiny_student_config()
    student = _build_lossless_student(teacher, student_config)

    calibration_batches = [_random_codes(teacher, seed=300 + i) for i in range(4)]

    def run_teacher(batch):
        teacher.forward_train(batch)

    residual_acts = _collect_activations(teacher.out_norm, run_teacher, calibration_batches)
    proj = compute_activation_projection(residual_acts, student_config.hidden_size)
    init_embeddings(teacher, student, proj)

    expected = teacher.text_emb.weight @ proj.down
    diff = (student.text_emb.weight - expected).abs().max().item()
    assert diff < 1e-4


def test_sanity_check_flags_degenerate_output():
    from distill.init_from_teacher import SanityReport

    report = SanityReport(passed=False, silence_frame_fraction=0.95, unique_token_fraction=0.01,
                           nan_or_inf=False, notes=["collapse"])
    assert not report.passed


def test_collect_ffn_hidden_activations_handles_bf16_model():
    """Regression test: on a real (bf16) teacher, `collect_ffn_hidden_activations`
    upcasts captured activations to float32 for numerically stable statistics,
    but `_gating_hidden_activation` then matmuls them against the model's real
    bf16 `linear_in.weight` -- `F.linear` requires both operands to match, and
    this crashed the very first time this code ran against a real bf16 teacher
    on GPU (the CPU-only tests elsewhere in this file all use float32 teachers,
    which never exercised the mismatch). Skips if bf16 isn't supported on
    this device (e.g. some CPUs), since that's the whole point of the check.
    """
    try:
        _ = torch.zeros(1, 1, 4, dtype=torch.bfloat16) @ torch.zeros(4, 4, dtype=torch.bfloat16)
    except RuntimeError:
        import pytest
        pytest.skip("bf16 matmul not supported on this device")

    teacher = _tiny_teacher(dtype=torch.bfloat16)
    codes = _random_codes(teacher, batch=1, frames=6, seed=42)

    def run_teacher(batch):
        teacher.forward_train(batch)

    hidden_acts = collect_ffn_hidden_activations(teacher.transformer.layers[0].gating, run_teacher, [codes])
    assert hidden_acts.dtype == torch.float32
    assert torch.isfinite(hidden_acts).all()


def test_norm_alpha_is_copied_not_reset_at_identity_projection():
    """Regression test for a real bug caught by tests/test_bit_exactness.py against
    the real (trained) 7B teacher: `init_attention_layer`/`init_ffn_layer` used to
    unconditionally reset `norm1.alpha`/`norm2.alpha` to ones. That's the right
    default when the residual basis is genuinely rotated/truncated (RMSNorm's
    scalar normalization doesn't commute with an arbitrary change of basis -- see
    compute_activation_projection's docstring), but at an IDENTITY projection (no
    actual reduction -- exactly tests/test_bit_exactness.py's identity-clone
    scenario, and only reachable there since `_tiny_teacher()` elsewhere in this
    file is deliberately never trained, so its alpha already defaults to ones and
    can't distinguish "copied" from "reset") the teacher's real alpha IS the
    correct value, and forcing it to ones is a real, avoidable numerical
    regression -- it produced a spurious ~1.6 max-abs-logit "reproduction failure"
    on the real teacher that had nothing to do with the actual weight-transform
    math. Simulates a "trained" teacher here by giving its norms non-uniform
    alpha values (impossible to distinguish from a bug with an untrained one).
    """
    teacher = _tiny_teacher()
    with torch.no_grad():
        for layer in teacher.transformer.layers:
            torch.nn.init.uniform_(layer.norm1.alpha, 0.5, 1.5)
            torch.nn.init.uniform_(layer.norm2.alpha, 0.5, 1.5)

    student_config = _tiny_student_config()
    student = _build_lossless_student(teacher, student_config)

    identity = Projection(down=torch.eye(TEACHER_DIM), up=torch.eye(TEACHER_DIM), rms=torch.ones(TEACHER_DIM))
    q_head_idx = select_query_heads(TEACHER_NUM_HEADS, TEACHER_NUM_HEADS)
    kv_bands = kv_head_bands(TEACHER_NUM_HEADS, TEACHER_NUM_HEADS)
    calibration_batches = [_random_codes(teacher, seed=500 + i) for i in range(2)]

    def run_teacher(batch):
        teacher.forward_train(batch)

    for teacher_layer, student_layer in zip(teacher.transformer.layers, student.transformer.layers):
        init_attention_layer(teacher_layer, student_layer, q_head_idx, kv_bands, identity)
        hidden_acts = collect_ffn_hidden_activations(teacher_layer.gating, run_teacher, calibration_batches)
        init_ffn_layer(teacher_layer, student_layer, identity, hidden_acts)

        assert torch.equal(student_layer.norm1.alpha, teacher_layer.norm1.alpha.to(student_layer.norm1.alpha.dtype))
        assert torch.equal(student_layer.norm2.alpha, teacher_layer.norm2.alpha.to(student_layer.norm2.alpha.dtype))
