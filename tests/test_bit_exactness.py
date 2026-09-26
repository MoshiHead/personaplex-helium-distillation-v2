# SPDX-License-Identifier: MIT
"""Frozen-component bit-exactness test: teacher temporal -> frozen depth ->
Mimi must reproduce the teacher's own audio when the "student" is, structurally,
an exact 1:1 clone of the teacher (hidden_size == teacher dim, num_attention_heads
== num_key_value_heads == teacher num_heads, i.e. no GQA reduction, no width
reduction, and an identity Bridge). Any drift here means the frozen wiring
itself (Bridge target space, out_norm reuse, depformer_in reuse, Mimi encode/
decode) is broken -- independent of anything the SVD/GQA-pooling init math or
actual distillation training does (see tests/test_init_sanity.py for those).

This is the SAME hard gate PersonaPlex_Distill_RunPod.ipynb's Section 3 runs at
full scale against the real downloaded 7B teacher; it needs real weights on
disk and is skipped otherwise (this sandboxed dev environment has neither the
weights nor the VRAM/time budget to run a 7B model, so this test is meant for
CI or the RunPod notebook, not for routine local runs).

Set PERSONAPLEX_TEACHER_WEIGHT / PERSONAPLEX_MIMI_WEIGHT / PERSONAPLEX_TOKENIZER
to real local paths (e.g. the files `PersonaPlex_Distill_RunPod.ipynb` downloads
in its Section 2) to run this for real.
"""

import os

import pytest
import torch

TEACHER_WEIGHT = os.environ.get("PERSONAPLEX_TEACHER_WEIGHT")
MIMI_WEIGHT = os.environ.get("PERSONAPLEX_MIMI_WEIGHT")
TOKENIZER = os.environ.get("PERSONAPLEX_TOKENIZER")

pytestmark = pytest.mark.skipif(
    not (TEACHER_WEIGHT and MIMI_WEIGHT and os.path.exists(TEACHER_WEIGHT) and os.path.exists(MIMI_WEIGHT)),
    reason="Requires real PersonaPlex teacher + Mimi weights on disk (set "
           "PERSONAPLEX_TEACHER_WEIGHT / PERSONAPLEX_MIMI_WEIGHT). This is the hard gate the "
           "notebook's Section 3 runs at full scale; it is not expected to run in a plain dev checkout.",
)


def _identity_clone_student_config():
    from distill.config import StudentConfig, InitConfig, TeacherReference, BridgeConfig

    return StudentConfig(
        name="identity_clone", num_hidden_layers=32, hidden_size=4096, intermediate_size=16896,
        num_attention_heads=32, num_key_value_heads=32,  # no GQA reduction: degenerates to teacher's MHA
        rope_theta=10000.0, max_frames=3000, norm="rms_norm_f32", gating="silu",
        layer_scale=None, causal=True,
        init=InitConfig(num_calibration_batches=1, keep_first=0, keep_last=0),
        teacher_reference=TeacherReference(),  # defaults already match the real teacher
        bridge=BridgeConfig(in_dim=4096, out_dim=4096),
    )


@torch.no_grad()
def test_frozen_component_chain_reproduces_teacher_audio():
    from moshi.models import loaders
    from moshi.models.lm import LMGen
    from distill.student_model import StudentLMModel
    from distill.init_from_teacher import (
        select_query_heads, kv_head_bands, init_attention_layer, init_ffn_layer, init_embeddings,
        collect_ffn_hidden_activations, Projection,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = loaders.get_moshi_lm(TEACHER_WEIGHT, device=device)
    teacher.eval()
    mimi = loaders.get_mimi(MIMI_WEIGHT, device)

    cfg = _identity_clone_student_config()
    student = StudentLMModel(cfg, device=device, dtype=teacher.emb[0].weight.dtype)
    student.load_frozen_from_teacher(teacher.state_dict())

    dim = cfg.hidden_size
    identity = Projection(down=torch.eye(dim, device=device, dtype=torch.float32),
                           up=torch.eye(dim, device=device, dtype=torch.float32),
                           rms=torch.ones(dim, device=device))
    q_head_idx = select_query_heads(32, 32)
    kv_bands = kv_head_bands(32, 32)

    # Minimal valid calibration batch, just to drive one teacher forward pass for
    # init_ffn_layer's channel-importance ranking. Since student_hidden == teacher_hidden
    # here (16896, no reduction), select_ffn_channels selects every channel regardless of
    # activation values (see its docstring) -- the actual content of this batch doesn't
    # matter, only that it's shaped like real token codes.
    calib_codes = torch.randint(0, teacher.card, (1, teacher.num_codebooks, 8), device=device)
    calib_codes[:, 0] = torch.randint(0, teacher.text_card, (1, 8), device=device)

    def run_teacher(batch):
        teacher.forward_train(batch)

    for teacher_layer, student_layer in zip(teacher.transformer.layers, student.transformer.layers):
        init_attention_layer(teacher_layer, student_layer, q_head_idx, kv_bands, identity)
        hidden_acts = collect_ffn_hidden_activations(teacher_layer.gating, run_teacher, [calib_codes])
        init_ffn_layer(teacher_layer, student_layer, identity, hidden_acts)
    init_embeddings(teacher, student, identity)
    student.bridge.linear.weight.copy_(torch.eye(dim, device=device, dtype=student.bridge.linear.weight.dtype))
    torch.nn.init.ones_(student.bridge.norm.alpha)
    student.eval()  # nn.Module defaults to training mode; LMGen asserts against that

    import sentencepiece
    from moshi.utils.compile import no_cuda_graph
    text_tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER)

    def run(lm):
        lm_gen = LMGen(lm, device=device, use_sampling=False, sample_rate=mimi.sample_rate,
                        frame_rate=mimi.frame_rate)
        lm_gen.text_prompt_tokens = text_tokenizer.encode("<system> You are a helpful assistant. <system>")
        with lm_gen.streaming(1), mimi.streaming(1):
            mimi.reset_streaming()
            lm_gen.reset_streaming()
            lm_gen._step_text_prompt()
            frames = []
            for _ in range(40):
                tokens = lm_gen.step(input_tokens=lm_gen._encode_sine_frame())
                if tokens is not None:
                    frames.append(mimi.decode(tokens[:, 1:9]))
        return torch.cat(frames, dim=-1)

    # CUDA graph capture is a performance optimization LMGen applies transparently
    # (moshi/utils/compile.py's CUDAGraphed); this test cares only about numerical
    # correctness. Running the teacher's LMGen and the student's LMGen sequentially in
    # the SAME process -- needed here to compare their outputs directly -- is not a
    # pattern any normal training/inference path exercises (training scores the teacher
    # via plain forward_train, never LMGen; moshi.offline's teacher-vs-student benchmark
    # runs each in its own subprocess). It surfaced a real CUDA stream-capture failure
    # ("operation failed due to a previous error during capture", raised from
    # torch.cuda.graphs.CUDAGraph.capture_end) that is about capture-protocol state, not
    # a kernel bug -- CUDA_LAUNCH_BLOCKING does not help pinpoint it, since it is not an
    # ordinary kernel launch error. Disabling CUDA graphs for this comparison sidesteps
    # the whole class of problem and is free: this test runs 40 frames total.
    with no_cuda_graph():
        teacher_audio = run(teacher)
        student_audio = run(student)

    assert teacher_audio.shape == student_audio.shape
    max_abs_diff = (teacher_audio - student_audio).abs().max().item()
    assert max_abs_diff < 1e-3, (
        f"Frozen-chain reproduction failed: max abs sample diff {max_abs_diff}. The student "
        "was built as a structural 1:1 clone of the teacher (no width/head reduction, identity "
        "bridge) -- any drift here points at the frozen wiring (Bridge target space, out_norm "
        "reuse, depformer_in reuse), not at the distillation math."
    )
