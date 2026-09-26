# SPDX-License-Identifier: MIT
"""Main distillation training loop: P1 (Align) -> P2 (Behavior) -> P3 (On-policy)
-> P4 (Polish), per distill/schedule.py. Single-GPU, resumable from checkpoint.

Not implemented here on purpose (out of scope for a single-GPU training script,
and not requested): multi-GPU/FSDP, mixed-precision loss scaling beyond the
repo's existing bf16-everywhere convention, and experiment tracking beyond
plain stdout + a CSV log -- wire in wandb/tensorboard around `log_step` if needed.
"""

import argparse
import csv
import logging
import time
from pathlib import Path
import typing as tp

import torch
import torch.nn as nn

from moshi.models import loaders
from moshi.models.lm import LMModel, LMGen, AUDIO_TOKENS_PER_STREAM, _delay_sequence, _undelay_sequence

from . import checkpoint as ckpt
from .config import load_student_config
from .data.dataset import TeacherTokenDataset, collate_chunks
from .init_from_teacher import initialize_student, sanity_check_student
from .losses import (
    RunningNorm, HiddenProjection, bridge_loss, ce_loss, hidden_loss,
    kl_distillation_loss, frame_weights, _codebook_weights, TEXT_KL_TEMPERATURE,
)
from .schedule import TrainingSchedule, PhaseSpec
from .student_model import StudentLMModel, build_student_lm

logger = logging.getLogger(__name__)


class ForwardCapture(tp.NamedTuple):
    logits: torch.Tensor
    logits_mask: torch.Tensor
    text_logits: torch.Tensor
    text_logits_mask: torch.Tensor
    raw_hidden: torch.Tensor
    layer_hidden: dict[int, torch.Tensor]


def run_forward_train_with_hooks(model: LMModel, codes: torch.Tensor, layer_indices: list[int]) -> ForwardCapture:
    """Re-implements `LMModel.forward_train`'s body (delay/undelay glue) with
    forward hooks attached so the (pre-out_norm, pre-bridge-for-teacher /
    post-bridge... no: pre-out_norm covers both, see forward_embeddings)
    intermediate hidden states are captured for L_bridge/L_hidden. Not a
    modification of `moshi/models/lm.py` -- a parallel path for training only.
    """
    B, K, T = codes.shape
    initial = model._get_initial_token().expand(B, -1, -1)
    delayed_codes = _delay_sequence(model.delays, codes, initial)
    delayed_codes = torch.cat([initial, delayed_codes], dim=2)

    captured_layers: dict[int, torch.Tensor] = {}
    raw_hidden_box: dict[str, torch.Tensor] = {}
    handles = []

    def make_layer_hook(i):
        def hook(module, inputs, output):
            captured_layers[i] = output
        return hook

    for idx in layer_indices:
        handles.append(model.transformer.layers[idx].register_forward_hook(make_layer_hook(idx)))

    def out_norm_hook(module, inputs, output):
        raw_hidden_box["raw"] = inputs[0]

    handles.append(model.out_norm.register_forward_hook(out_norm_hook))

    try:
        transformer_out, text_logits = model.forward_embeddings(model.embed_codes(delayed_codes[:, :, :-1]))
        logits = model.forward_depformer_training(delayed_codes[:, :, 1:], transformer_out)
    finally:
        for h in handles:
            h.remove()

    logits, logits_mask = _undelay_sequence(
        model.delays[model.audio_offset:model.audio_offset + model.dep_q], logits, fill_value=float("nan"))
    logits_mask &= (codes[:, model.audio_offset:model.audio_offset + model.dep_q] != model.zero_token_id)
    text_logits, text_logits_mask = _undelay_sequence(model.delays[:1], text_logits, fill_value=float("nan"))
    text_logits_mask &= (codes[:, :1] != model.zero_token_id)

    return ForwardCapture(logits, logits_mask, text_logits, text_logits_mask,
                           raw_hidden_box["raw"], captured_layers)


def _ce_targets(codes: torch.Tensor, model: LMModel,
                 text_mask: torch.Tensor, audio_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds CE targets with ignore_index=-100 both at the true start-of-sequence
    marker tokens AND wherever `_undelay_sequence` marked the position invalid
    (NaN-padded tail from the delay shift) -- these are disjoint position sets
    (start vs. tail) and both must be excluded, or CE trains against fabricated
    zero-logits at the delay tail.
    """
    text_target = codes[:, :1].clone()
    text_target[text_target == model.text_initial_token_id] = -100
    text_target[~text_mask] = -100
    audio_target = codes[:, model.audio_offset:model.audio_offset + model.dep_q].clone()
    audio_target[audio_target == model.initial_token_id] = -100
    audio_target[~audio_mask] = -100
    return text_target, audio_target


@torch.no_grad()
def generate_student_rollout(student: StudentLMModel, codes_prompt: torch.Tensor,
                              rollout_frames: int) -> torch.Tensor:
    """On-policy rollout (phases P3/P4): the student generates its own trajectory
    autoregressively (sampling, not teacher-forced) starting from a real prompt
    prefix. The returned `codes` are then teacher-forced through BOTH models in
    the caller so the teacher scores each student-visited state.

    Codebook layout (see AUDIO_TOKENS_PER_STREAM / prepare_step_input in
    moshi/models/lm.py): 0 = text, [1, 1+AUDIO_TOKENS_PER_STREAM) = "moshi"
    (self/agent) stream, [1+AUDIO_TOKENS_PER_STREAM, ...) = "input" (other
    party) stream. During the prompt prefix both streams are teacher-forced from
    real data; during the rollout only the self/text channels are freely
    sampled, while the other-party channel continues with a neutral "sine"
    placeholder (same convention `LMGen` uses while loading voice/text prompts).
    """
    device = student.device
    # LMGen asserts the model is not in training mode. compute_losses puts the model back into
    # train() right after this returns (for the actual gradient step), so don't just assume
    # whatever mode the caller left it in -- force eval here and restore it afterward.
    was_training = student.training
    student.eval()
    try:
        lm_gen = LMGen(student, device=device, use_sampling=True)
        split = student.audio_offset + AUDIO_TOKENS_PER_STREAM
        B = codes_prompt.shape[0]
        with lm_gen.streaming(B):
            prefix_len = codes_prompt.shape[-1]
            rollout_out = []
            for c in range(prefix_len):
                input_tokens = codes_prompt[:, split:, c:c + 1]
                moshi_tokens = codes_prompt[:, student.audio_offset:split, c:c + 1]
                text_token = codes_prompt[:, 0, c]
                tokens = lm_gen.step(input_tokens=input_tokens, moshi_tokens=moshi_tokens, text_token=text_token)
                if tokens is not None:
                    rollout_out.append(tokens[:, :, 0])

            sine_frame = lm_gen._encode_sine_frame().expand(B, -1, -1)
            for _ in range(rollout_frames):
                tokens = lm_gen.step(input_tokens=sine_frame)
                if tokens is not None:
                    rollout_out.append(tokens[:, :, 0])
        return torch.stack(rollout_out, dim=-1)
    finally:
        student.train(was_training)


def compute_losses(
    student: StudentLMModel,
    teacher: LMModel,
    hidden_projections: nn.ModuleList,
    selected_teacher_layers: list[int],
    codes: torch.Tensor,
    transition_mask: torch.Tensor,
    phase: PhaseSpec,
    running_norms: dict[str, RunningNorm],
    audio_kl_temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        teacher_capture = run_forward_train_with_hooks(teacher, codes, selected_teacher_layers)
    student_capture = run_forward_train_with_hooks(student, codes, list(range(len(hidden_projections))))

    codebook_w = _codebook_weights(student.dep_q, codes.device, torch.float32)
    # cb1 (index 0 of the dep_q audio codebooks) teacher logits, with the delay-tail
    # NaN padding (see kl_distillation_loss docstring) scrubbed before computing entropy.
    cb1_logits = torch.nan_to_num(teacher_capture.logits[:, 0].float(), nan=0.0)
    frame_w = frame_weights(transition_mask, cb1_logits)
    frame_w = frame_w.unsqueeze(1)  # [B, 1, T] broadcast over codebooks

    kl_audio = kl_distillation_loss(
        student_capture.logits, teacher_capture.logits, student_capture.logits_mask,
        temperature=audio_kl_temperature, weight=codebook_w.view(1, -1, 1) * frame_w,
    )
    kl_text = kl_distillation_loss(
        student_capture.text_logits, teacher_capture.text_logits, student_capture.text_logits_mask,
        temperature=TEXT_KL_TEMPERATURE, weight=1.0,
    )

    text_target, audio_target = _ce_targets(
        codes, student, student_capture.text_logits_mask, student_capture.logits_mask)
    loss_ce = ce_loss(student_capture.text_logits, text_target, ignore_index=-100) + \
        ce_loss(student_capture.logits, audio_target, ignore_index=-100)

    loss_bridge = bridge_loss(student_capture.raw_hidden, teacher_capture.raw_hidden)

    hidden_terms = []
    for student_idx in range(len(hidden_projections)):
        hidden_terms.append(hidden_loss(
            student_capture.layer_hidden[student_idx],
            teacher_capture.layer_hidden[selected_teacher_layers[student_idx]],
            hidden_projections[student_idx],
        ))
    loss_hidden = torch.stack(hidden_terms).mean() if hidden_terms else torch.zeros((), device=codes.device)

    loss_speaker = torch.zeros((), device=codes.device)  # phase 1-3: speaker sim is a validation metric only

    terms = {
        "ce": running_norms["ce"](loss_ce),
        "kl": running_norms["kl"](kl_audio + kl_text),
        "bridge": running_norms["bridge"](loss_bridge),
        "hidden": running_norms["hidden"](loss_hidden),
        "speaker": loss_speaker,
    }
    total = (
        phase.alpha * terms["ce"] + phase.beta * terms["kl"] + phase.gamma * terms["bridge"] +
        phase.delta * terms["hidden"] + phase.epsilon * terms["speaker"]
    )
    raw = {
        "ce_raw": loss_ce.item(), "kl_audio_raw": kl_audio.item(), "kl_text_raw": kl_text.item(),
        "bridge_raw": loss_bridge.item(), "hidden_raw": loss_hidden.item(), "total": total.item(),
    }
    return total, raw


def train(
    student_config_name: str,
    teacher_checkpoint: str,
    data_dir: str,
    output_dir: str,
    total_steps: int,
    batch_size: int,
    chunk_frames: int,
    lr: float,
    audio_kl_temperature: float,
    device: str,
    resume: bool,
    save_every: int,
    smoke_test: bool,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16

    logger.info("Loading teacher (frozen, resident) from %s", teacher_checkpoint)
    teacher = loaders.get_moshi_lm(teacher_checkpoint, device=device, dtype=dtype)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student_config = load_student_config(student_config_name)
    student = build_student_lm(student_config, teacher_checkpoint, device=device, dtype=dtype)

    dataset = TeacherTokenDataset(data_dir)
    logger.info("Loaded %d samples (%.1f frames total) from %s",
                len(dataset), dataset.total_frames(), data_dir)

    schedule = TrainingSchedule(total_steps)
    checkpoint_path = output_dir / "student_checkpoint.pt"

    selected_teacher_layers: list[int]
    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # see checkpoint.load
        selected_teacher_layers = payload["selected_teacher_layers"]
        start_step = payload["step"]
    else:
        # Cycle through DIFFERENT samples per calibration batch -- `dataset[i] for i in
        # range(min(len(dataset), 4))` inside a loop over `range(calib_n)` (an earlier version
        # of this code) always selects the SAME first-4 samples regardless of which of the
        # `calib_n` "different" batches is being built, badly limiting calibration diversity
        # (init_bridge_least_squares's ridge regularization keeps a repetitive calibration set
        # from producing a non-finite fit, but a diverse one is still a materially better fit).
        calib_batch_size = min(len(dataset), 4)
        calib_n = min(student_config.init.num_calibration_batches, max(1, len(dataset) // batch_size))
        calibration_batches = [
            collate_chunks(
                [dataset[(b * calib_batch_size + i) % len(dataset)] for i in range(calib_batch_size)],
                chunk_frames,
            )["codes"].to(device)
            for b in range(calib_n)
        ]
        diag = initialize_student(student, teacher, calibration_batches,
                                   student_config.init.keep_first, student_config.init.keep_last)
        selected_teacher_layers = diag["selected_layers"]
        logger.info("Student initialized from teacher. Diagnostics: %s",
                    {k: v for k, v in diag.items() if k != "layer_scores"})
        start_step = 0

    hidden_projections = nn.ModuleList([
        HiddenProjection(student_config.hidden_size, student_config.teacher_reference.dim)
        for _ in selected_teacher_layers
    ]).to(device=device, dtype=torch.float32)

    main_params = list(student.trainable_parameters()) + list(hidden_projections.parameters())
    depth_params = list(student.depformer.parameters())
    optimizer = torch.optim.AdamW([
        {"params": main_params, "lr": lr},
        {"params": depth_params, "lr": lr * 0.1},
    ])

    running_norms = {name: RunningNorm() for name in ("ce", "kl", "bridge", "hidden")}

    if resume and checkpoint_path.exists():
        full_payload = ckpt.load(str(checkpoint_path), student, hidden_projections, optimizer)
        for name, norm in running_norms.items():
            state = full_payload["running_norm_states"].get(name)
            if state:
                norm.load_state_dict(state)
        schedule.load_state_dict(full_payload["schedule_state"])
        logger.info("Resumed from %s at step %d", checkpoint_path, start_step)

    csv_path = output_dir / "train_log.csv"
    csv_is_new = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if csv_is_new:
        csv_writer.writerow(["step", "phase", "total", "ce_raw", "kl_audio_raw", "kl_text_raw",
                              "bridge_raw", "hidden_raw", "seconds_per_step"])

    idx_cycle = list(range(len(dataset)))
    cursor = 0
    end_step = 5 if smoke_test else total_steps

    for step in range(start_step, end_step):
        t0 = time.time()
        phase, changed = schedule.step(step)
        if changed and phase.depth_transformer_unfrozen:
            student.unfreeze_depth_transformer()
            logger.info("Step %d: entering %s, depth transformer unfrozen at %.2fx LR",
                        step, phase.name, phase.depth_transformer_lr_mult)
        if changed:
            logger.info("Step %d: entering phase %s", step, phase.name)

        batch_indices = [idx_cycle[(cursor + i) % len(idx_cycle)] for i in range(batch_size)]
        cursor += batch_size
        samples = [dataset[i] for i in batch_indices]
        batch = collate_chunks(samples, chunk_frames)
        codes = batch["codes"].to(device)
        transition_mask = batch["transition_mask"].to(device)

        if phase.on_policy:
            prompt_len = max(1, chunk_frames // 4)
            codes = generate_student_rollout(student, codes[:, :, :prompt_len],
                                              rollout_frames=chunk_frames - prompt_len)
            # LMGen.step() returns None for the first max_delay+1 calls (it buffers for the
            # codebook delay pattern before yielding a token), so the rollout's returned
            # `codes` is shorter than `chunk_frames` by a few frames -- re-slice
            # transition_mask (built from the original, full-length batch) to match, or the
            # frame_weights broadcast inside compute_losses fails with a shape mismatch.
            transition_mask = transition_mask[:, :codes.shape[-1]]

        student.train()
        total_loss, raw = compute_losses(
            student, teacher, hidden_projections, selected_teacher_layers,
            codes, transition_mask, phase, running_norms, audio_kl_temperature,
        )
        student.eval()

        if not torch.isfinite(total_loss):
            # A single non-finite loss corrupts the ENTIRE model if allowed through:
            # clip_grad_norm_'s total-norm computation becomes non-finite the moment any one
            # parameter's gradient is, and the resulting non-finite clip coefficient scales
            # (and so corrupts) every other parameter's gradient on that step. Skip the step
            # entirely rather than let one bad batch/initialization propagate forward.
            logger.warning("step=%d phase=%s: non-finite total_loss (%s) -- skipping optimizer "
                            "step. raw=%s", step, phase.name, total_loss.item(), raw)
            continue

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(main_params + depth_params, max_norm=1.0)
        optimizer.step()

        dt = time.time() - t0
        csv_writer.writerow([step, phase.name, raw["total"], raw["ce_raw"], raw["kl_audio_raw"],
                              raw["kl_text_raw"], raw["bridge_raw"], raw["hidden_raw"], dt])
        csv_file.flush()
        if step % 10 == 0 or smoke_test:
            logger.info("step=%d phase=%s total=%.4f ce=%.4f kl=%.4f+%.4f bridge=%.4f hidden=%.4f (%.2fs)",
                        step, phase.name, raw["total"], raw["ce_raw"], raw["kl_audio_raw"], raw["kl_text_raw"],
                        raw["bridge_raw"], raw["hidden_raw"], dt)

        if (step + 1) % save_every == 0 or step + 1 == end_step:
            ckpt.save(
                str(checkpoint_path), step + 1, student, hidden_projections, optimizer,
                schedule.state_dict(), {k: v.state_dict() for k, v in running_norms.items()},
                selected_teacher_layers,
            )
            logger.info("Saved checkpoint at step %d to %s", step + 1, checkpoint_path)

    csv_file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-config", required=True, help="e.g. student_ppx_s")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-frames", type=int, default=250, help="~20s at 12.5Hz")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--audio-kl-temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--smoke-test", action="store_true", help="Run 5 steps end-to-end and exit.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train(
        student_config_name=args.student_config,
        teacher_checkpoint=args.teacher_checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        chunk_frames=args.chunk_frames,
        lr=args.lr,
        audio_kl_temperature=args.audio_kl_temperature,
        device=args.device,
        resume=args.resume,
        save_every=args.save_every,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
