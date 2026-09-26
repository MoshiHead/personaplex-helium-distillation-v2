# SPDX-License-Identifier: MIT
"""Distillation trainer (v2): P1 Align -> P2 Behavior -> P3 On-policy -> P4 Polish (distill/schedule.py).

Single GPU:  python -m distill.train --student-config student_ppx_s --teacher-checkpoint model.safetensors \
                 --data-dir <dataset>/train --val-dir <dataset>/val --output-dir <run> --chunk-frames 1000 ...
Multi GPU:   torchrun --nproc_per_node=<N> -m distill.train <same args>          (DDP, one process per GPU)

What this version does (and why), relative to the first trainer:
  * Multi-GPU DDP + gradient accumulation. Global batch = world_size x micro_batch x grad_accum.
  * Deterministic per-epoch SHUFFLE derived from (seed, step): resuming continues the exact same data order.
  * fp32 master weights for everything trainable + bf16 autocast forward (the first trainer kept parameters AND
    AdamW states in bf16, where small updates round to zero). KL divergences are computed in fp32.
  * Loss masks match the data: the voice/persona PROMPT PREFIX (forced tokens at inference, prefix_len in the
    manifest) is excluded from hard-label CE; KL / bridge / hidden losses see every frame (the student must encode
    the prompt like the teacher). Only text + the AGENT's 8 codebooks are distilled: the user-stream predictions
    are never used at inference (LMGen overwrites them with the real microphone input), and the teacher loader
    copies depformer heads 0-7 -> 8-15 when missing, so teacher user-stream logits may not be trained predictions.
  * On-policy rollouts (P3/P4) now (a) start after the longest prompt prefix in the batch (the old start,
    chunk_frames//4, fell INSIDE the prompt on this data), (b) keep feeding the RECORDED user channel (the old code
    fed the prompt-phase SINE token as "user"), (c) are frame-aligned exactly like LMGen (the old rollout dropped
    frame 0 and shifted everything by one), (d) are verified against the data on every forced frame, and
    (e) get CE only on real-data frames (the old code applied CE to the student's own samples).
  * Warmup + cosine LR; weight decay only on matrices (not norms / embeddings / L_hidden projections).
  * Phase 4 depformer fine-tuning is kept: checkpoints and the final export include `depformer.*` once trained.
  * DDP-safe skip of non-finite steps (all ranks agree), all-reduced loss normalizers, SIGTERM-safe stop.
  * Validation loop (val split), best / latest / periodic checkpoints, structured logs under <output>/logs.
  * --smoke-test: every phase incl. on-policy rollouts, eval, checkpoint, export in a handful of steps.
"""

import argparse
import contextlib
import csv
import datetime
import json
import logging
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import typing as tp
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from moshi.models import loaders
from moshi.models.lm import LMModel, LMGen, AUDIO_TOKENS_PER_STREAM, _delay_sequence, _undelay_sequence
from moshi.utils.compile import no_cuda_graph

from . import checkpoint as ckpt
from .config import load_student_config
from .data.dataset import TeacherTokenDataset, collate_chunks
from .init_from_teacher import initialize_student
from .losses import RunningNorm, HiddenProjection, bridge_loss, frame_weights, _codebook_weights, TEXT_KL_TEMPERATURE
from .schedule import TrainingSchedule, PhaseSpec, PHASES, phase_at
from .student_model import StudentLMModel, build_student_lm, FROZEN_PREFIXES

logger = logging.getLogger("distill.train")
N_AGENT_CODEBOOKS = AUDIO_TOKENS_PER_STREAM          # 8


def _patch_kv_cache_dtype_for_autocast():
    """Streaming KV caches take their dtype from the attention WEIGHTS. With fp32 master weights + bf16 autocast
    (on-policy rollouts), keys/values are bf16 while the cache would be fp32 -> index_copy_ dtype error. Inside this
    training process only, create the cache in the active autocast dtype (bf16 -- exactly what inference uses).
    Without autocast nothing changes; moshi's source (server / offline / teacher) is untouched."""
    from moshi.modules import transformer as mt
    from . import gqa_attention as ga
    for cls in (mt.StreamingMultiheadAttention, ga.GQAStreamingMultiheadAttention):
        orig = cls._init_streaming_state
        if getattr(orig, "_ppx_autocast_kv", False):
            continue

        def patched(self, batch_size, _orig=orig):
            state = _orig(self, batch_size)
            cache = state.kv_cache.cache
            if torch.is_autocast_enabled(cache.device.type):
                want = torch.get_autocast_dtype(cache.device.type)
                if cache.dtype != want:
                    state.kv_cache.cache = cache.to(want)
            return state
        patched._ppx_autocast_kv = True
        cls._init_streaming_state = patched


_patch_kv_cache_dtype_for_autocast()


# ================================================================================================ forward capture
class ForwardCapture(tp.NamedTuple):
    logits: torch.Tensor
    logits_mask: torch.Tensor
    text_logits: torch.Tensor
    text_logits_mask: torch.Tensor
    raw_hidden: torch.Tensor
    layer_hidden: dict[int, torch.Tensor]


def run_forward_train_with_hooks(model: LMModel, codes: torch.Tensor, layer_indices: list[int]) -> ForwardCapture:
    """`LMModel.forward_train` (delay / undelay glue) with forward hooks that capture the per-layer temporal
    outputs (L_hidden) and the input of `out_norm` (L_bridge: the teacher's raw transformer output / the student's
    Bridge output). Codes [B, 17, T] -> logits aligned with codes (no extra shift needed for CE / KL)."""
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
    return ForwardCapture(logits, logits_mask, text_logits, text_logits_mask, raw_hidden_box["raw"], captured_layers)


class StudentWithProjections(nn.Module):
    """What DDP wraps: the student + the L_hidden projection heads, so ONE forward call covers every trainable
    parameter (DDP needs the forward to go through the wrapped module to synchronize gradients)."""

    def __init__(self, student: StudentLMModel, projections: nn.ModuleList):
        super().__init__()
        self.student = student
        self.projections = projections

    def forward(self, codes: torch.Tensor) -> dict:
        cap = run_forward_train_with_hooks(self.student, codes, list(range(len(self.projections))))
        projected = [self.projections[i](cap.layer_hidden[i].float()) for i in range(len(self.projections))]
        return {"logits": cap.logits, "logits_mask": cap.logits_mask, "text_logits": cap.text_logits,
                "text_logits_mask": cap.text_logits_mask, "raw_hidden": cap.raw_hidden, "projected": projected}


# ================================================================================================ losses
def kl_fp32(student_logits, teacher_logits, mask, temperature, weight=1.0):
    """KL(teacher || student) at `temperature` (x tau^2), computed in fp32, averaged over valid positions.
    NaNs in the undelay tail are masked out (and scrubbed so nan*0 can't leak)."""
    s = torch.nan_to_num(student_logits.float(), nan=0.0) / temperature
    t = torch.nan_to_num(teacher_logits.float(), nan=0.0) / temperature
    log_s = F.log_softmax(s, dim=-1)
    log_t = F.log_softmax(t, dim=-1)
    kl = (log_t.exp() * (log_t - log_s)).sum(dim=-1) * (temperature ** 2)
    kl = kl * weight
    m = mask.to(kl.dtype)
    return (kl * m).sum() / m.sum().clamp_min(1.0)


def masked_ce(logits, targets, mask):
    """Hard-label CE averaged over mask==True positions; exactly 0 (not NaN) when no position is valid."""
    lg = torch.nan_to_num(logits.float(), nan=0.0)
    tg = targets.masked_fill(~mask, -100)
    loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1), ignore_index=-100, reduction="sum")
    return loss / mask.sum().clamp_min(1)


def hidden_term(projected: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor:
    """== distill.losses.hidden_loss on already-projected student states (L2-normalize, MSE + (1 - cosine)), in a
    memory-lean form. For unit vectors p, t in R^D: MSE = mean_pos(|p - t|^2) / D = 2 (1 - mean_pos(p.t)) / D and
    cosine = p.t, so the loss is (1 + 2/D) * (1 - mean_pos(p.t)). Same value, far fewer fp32 [B, T, 4096]
    intermediates kept for backward (x 20 layers this was a large share of activation memory)."""
    p = F.normalize(projected.float(), dim=-1)
    with torch.no_grad():
        t = F.normalize(teacher_hidden.float(), dim=-1)
    dot = (p * t).sum(dim=-1).mean()
    return (1.0 + 2.0 / p.shape[-1]) * (1.0 - dot)


def loss_terms(s_out: dict, t_cap: ForwardCapture, codes: torch.Tensor, transition_mask: torch.Tensor,
               prefix_lens: torch.Tensor, ce_end: tp.Optional[torch.Tensor], selected_layers: list[int],
               args, initial_audio_id: int, text_initial_id: int) -> dict[str, torch.Tensor]:
    B, K, T = codes.shape
    dev = codes.device
    frames = torch.arange(T, device=dev)[None, :]
    conv = frames >= prefix_lens[:, None].to(dev)                    # [B, T] conversation (non-prompt) frames
    ce_frames = conv if not args.ce_on_prefix else torch.ones_like(conv)
    if ce_end is not None:                                           # on-policy: CE only on real-data frames
        ce_frames = ce_frames & (frames < ce_end[:, None].to(dev))
    kl_frames = torch.ones_like(conv) if args.kl_on_prefix else conv

    na = 2 * N_AGENT_CODEBOOKS if args.distill_user_stream else N_AGENT_CODEBOOKS
    s_audio, t_audio = s_out["logits"][:, :na], t_cap.logits[:, :na]
    a_mask = s_out["logits_mask"][:, :na]
    t_mask = s_out["text_logits_mask"]                               # [B, 1, T]

    cb_w = _codebook_weights(na, dev, torch.float32).view(1, -1, 1)
    cb1 = torch.nan_to_num(t_cap.logits[:, 0].float(), nan=0.0)      # teacher's first agent codebook
    fw = frame_weights(transition_mask.to(dev), cb1).unsqueeze(1)    # [B, 1, T]
    kl_audio = kl_fp32(s_audio, t_audio, a_mask & kl_frames[:, None], args.audio_kl_temperature, weight=cb_w * fw)
    kl_text = kl_fp32(s_out["text_logits"], t_cap.text_logits, t_mask & kl_frames[:, None], TEXT_KL_TEMPERATURE)

    text_tgt = codes[:, :1]
    text_ok = t_mask & ce_frames[:, None] & (text_tgt != text_initial_id)
    audio_tgt = codes[:, 1:1 + na]
    audio_ok = a_mask & ce_frames[:, None] & (audio_tgt != initial_audio_id)
    ce = masked_ce(s_out["text_logits"], text_tgt, text_ok) + masked_ce(s_audio, audio_tgt, audio_ok)

    br = bridge_loss(s_out["raw_hidden"], t_cap.raw_hidden)
    hid = torch.stack([hidden_term(s_out["projected"][i], t_cap.layer_hidden[selected_layers[i]])
                       for i in range(len(selected_layers))]).mean()
    return {"ce": ce, "kl_audio": kl_audio, "kl_text": kl_text, "kl": kl_audio + kl_text, "bridge": br, "hidden": hid,
            "ce_frames": ce_frames.float().sum(), "kl_frames": kl_frames.float().sum()}


class SyncedNorm:
    """RunningNorm whose statistic is updated with the all-reduced value, so every rank scales identically."""

    def __init__(self):
        self.norm = RunningNorm()

    def scale(self, local_value: float) -> float:
        m = self.norm._mean
        return (m if m is not None else max(abs(local_value), 1e-6)) + self.norm.eps

    def update(self, global_value: float):
        self.norm(torch.tensor(global_value))


# ================================================================================================ distributed
class DistInfo(tp.NamedTuple):
    rank: int
    world: int
    local_rank: int
    device: torch.device

    @property
    def main(self):
        return self.rank == 0


def setup_distributed(device_arg: str) -> DistInfo:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", "0"))
        if device_arg.startswith("cuda"):
            torch.cuda.set_device(local)
            device = torch.device("cuda", local)
            backend = "nccl"
        else:
            device, backend = torch.device("cpu"), "gloo"
        # generous timeout: rank 0 alone runs the student initialization while the others wait
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=90))
        return DistInfo(rank, world, local, device)
    device = torch.device(device_arg)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    return DistInfo(0, 1, 0, device)


def barrier(di: DistInfo):
    if di.world > 1:
        dist.barrier()


def all_reduce(t: torch.Tensor, di: DistInfo, op=None) -> torch.Tensor:
    if di.world > 1:
        dist.all_reduce(t, op=op or dist.ReduceOp.SUM)
    return t


# ================================================================================================ data
class GlobalBatchSampler:
    """Global step s consumes positions [s*G, (s+1)*G) of an infinite stream made of per-epoch permutations
    (numpy RNG seeded with (seed, epoch)). Fully determined by the step -> exact resume, no sampler state."""

    def __init__(self, n: int, global_batch: int, seed: int):
        self.n, self.g, self.seed = n, global_batch, seed
        self._cache: dict[int, np.ndarray] = {}

    def perm(self, epoch: int) -> np.ndarray:
        if epoch not in self._cache:
            if len(self._cache) > 3:
                self._cache.pop(min(self._cache))
            self._cache[epoch] = np.random.default_rng([self.seed, epoch]).permutation(self.n)
        return self._cache[epoch]

    def global_indices(self, step: int) -> list[int]:
        out = []
        for p in range(step * self.g, (step + 1) * self.g):
            e, o = divmod(p, self.n)
            out.append(int(self.perm(e)[o]))
        return out

    def epoch_of(self, step: int) -> float:
        return step * self.g / self.n


def load_batch(ds: TeacherTokenDataset, idxs: list[int], chunk: int, device) -> dict:
    b = collate_chunks([ds[i] for i in idxs], chunk)
    if len(b["sample_ids"]) != len(idxs):
        raise RuntimeError(f"{len(idxs) - len(b['sample_ids'])} sample(s) shorter than --chunk-frames {chunk} "
                           "(collate_chunks drops them) -- the dataset analysis at startup should have caught this")
    return {"codes": b["codes"].to(device, non_blocking=True), "transition_mask": b["transition_mask"].to(device),
            "prefix_lens": b["prefix_lens"].to(device), "sample_ids": b["sample_ids"]}


def analyze_dataset(ds: TeacherTokenDataset, chunk: int, name: str) -> dict:
    ents = ds.entries
    frames = np.array([e["num_frames"] for e in ents])
    pls = np.array([int(e.get("prefix_len", 0)) for e in ents])
    too_short = int((frames < chunk).sum())
    rep = {"name": name, "samples": len(ents), "frames_min": int(frames.min()), "frames_max": int(frames.max()),
           "chunk_frames": chunk, "samples_shorter_than_chunk": too_short,
           "hours_total_in_chunk": round(float(np.minimum(frames, chunk).sum()) / 12.5 / 3600, 2),
           "conversation_hours_in_chunk": round(float(np.clip(np.minimum(frames, chunk) - pls, 0, None).sum()) / 12.5 / 3600, 2),
           "prefix_len_min": int(pls.min()), "prefix_len_max": int(pls.max()), "prefix_len_mean": round(float(pls.mean()), 1),
           "has_prefix_len": bool(pls.max() > 0),
           "scenarios": {k: int(v) for k, v in zip(*np.unique([e.get("scenario", "?") for e in ents], return_counts=True))}}
    return rep


# ================================================================================================ on-policy rollout
@torch.no_grad()
def student_rollout(student: StudentLMModel, codes: torch.Tensor, start: int, n_total: int, autocast_ctx,
                    sampling: dict) -> torch.Tensor:
    """Returns [B, 17, n_total]: frames < start are the data (teacher-forced on every channel); from `start` on the
    student samples text + agent audio while the user channel keeps receiving the RECORDED user codes.

    LMGen overwrites call 0 with the initial token and returns frame f at call f+1 (max_delay=1), exactly as in
    data generation where recorded codes[:, j] == call j+1. So call 0 is a dummy and call j+1 carries codes[:, j];
    n_total + 2 calls return exactly frames 0 .. n_total-1, aligned with `codes`."""
    B, K, T = codes.shape
    u = student.audio_offset + AUDIO_TOKENS_PER_STREAM             # first user-stream row (9)
    was_training = student.training
    student.eval()
    lm_gen = LMGen(student, device=codes.device, use_sampling=True, **sampling)
    out = []
    try:
        with no_cuda_graph(), autocast_ctx, lm_gen.streaming(B):
            for c in range(n_total + 2):
                j = min(max(c - 1, 0), T - 1)                     # frame provided at this call (call 0: dummy)
                user = codes[:, u:, j:j + 1]
                if c - 1 < start:
                    tok = lm_gen.step(input_tokens=user, moshi_tokens=codes[:, 1:u, j:j + 1], text_token=codes[:, 0, j])
                else:
                    tok = lm_gen.step(input_tokens=user)
                if tok is not None:
                    out.append(tok[:, :, 0])
    finally:
        student.train(was_training)
    return torch.stack(out, dim=-1)[:, :, :n_total].contiguous()


# ================================================================================================ trainer
TRAIN_CSV_COLUMNS = ["step", "phase", "total", "ce_raw", "kl_audio_raw", "kl_text_raw", "bridge_raw", "hidden_raw",
                     "seconds_per_step", "lr", "grad_norm", "onpolicy", "frames_per_s", "peak_mem_gb", "skipped_total",
                     "epoch"]
VAL_CSV_COLUMNS = ["step", "phase", "val_kl_text", "val_kl_audio", "val_ce_text", "val_ce_audio", "val_bridge",
                   "val_hidden", "val_agree_text", "val_agree_cb0", "val_samples", "best"]


class RunLogger:
    """<output>/logs: train_rank<r>.log (every rank), metrics.jsonl, events.jsonl, val_log.csv, heartbeat.json;
    <output>/train_log.csv (same columns as before + extras). Only rank 0 writes the shared files."""

    def __init__(self, out_dir: Path, di: DistInfo):
        self.di = di
        self.dir = out_dir / "logs"
        self.dir.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter(f"%(asctime)s %(levelname)s [rank{di.rank}] %(message)s")
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        for h in list(root.handlers):
            root.removeHandler(h)
        fh = logging.FileHandler(self.dir / f"train_rank{di.rank}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        if di.main:
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            root.addHandler(sh)
        logging.captureWarnings(True)
        self.train_csv = out_dir / "train_log.csv"
        self.val_csv = self.dir / "val_log.csv"

    def _csv(self, path: Path, cols: list[str], row: dict):
        new = not path.exists()
        if not new:
            with open(path, encoding="utf-8") as f:
                header = f.readline().strip().split(",")
            if header != cols:                     # a log from an older trainer version: keep it, start fresh
                path.replace(path.with_suffix(f".old{int(time.time())}.csv"))
                new = True
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

    def train_row(self, row: dict):
        if self.di.main:
            self._csv(self.train_csv, TRAIN_CSV_COLUMNS, row)
            self.jsonl("metrics.jsonl", row)

    def val_row(self, row: dict):
        if self.di.main:
            self._csv(self.val_csv, VAL_CSV_COLUMNS, row)
            self.jsonl("metrics.jsonl", {"type": "val", **row})

    def jsonl(self, name: str, obj: dict):
        if self.di.main:
            with open(self.dir / name, "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **obj}) + "\n")

    def event(self, kind: str, **info):
        logger.info("EVENT %s %s", kind, json.dumps(info)[:2000])
        self.jsonl("events.jsonl", {"event": kind, **info})

    def write_json(self, name: str, obj: dict):
        if self.di.main:
            tmp = self.dir / f".{name}.tmp"
            tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
            tmp.replace(self.dir / name)


def load_teacher(args, device) -> LMModel:
    if args.teacher_kwargs_json:                 # tests only: a tiny synthetic teacher
        from safetensors.torch import load_file
        kw = json.loads(args.teacher_kwargs_json)
        teacher = LMModel(device=device, dtype=torch.bfloat16, **kw)
        teacher.load_state_dict(load_file(args.teacher_checkpoint, device="cpu"), strict=True)
        # LMModel.__init__ builds text_linear/out_norm/depformer_in/linears without a dtype; get_moshi_lm fixes
        # that with a trailing .to() -- do the same here.
        teacher = teacher.to(device=device, dtype=torch.bfloat16)
    else:
        # Load on CPU, then move to THIS rank's GPU. get_moshi_lm(device=cuda:N) calls safetensors'
        # load_file(device="cuda") without the index, which safetensors maps to cuda:0 -- under DDP every rank would
        # first put the whole 16.7 GB teacher on GPU 0 (and the caching allocator keeps it there): OOM on rank 0.
        teacher = loaders.get_moshi_lm(args.teacher_checkpoint, device="cpu", dtype=torch.bfloat16)
        teacher = teacher.to(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


TRAINABLE_TOP = ("transformer", "emb", "text_emb", "bridge")


def param_groups(student: StudentLMModel, projections: nn.ModuleList, weight_decay: float, depth_mult: float):
    """Decay only matrices; never norms, biases, embeddings or the L_hidden projections. The depformer group
    exists from the start (params frozen until Phase 4, so AdamW simply skips them) -> stable group layout."""
    dec, nodec, dep_dec, dep_nodec = [], [], [], []
    for n, p in student.named_parameters():
        top = n.split(".")[0]
        if n.startswith("depformer."):
            (dep_dec if p.ndim >= 2 else dep_nodec).append(p)
        elif n.startswith(FROZEN_PREFIXES):
            continue
        elif top in ("emb", "text_emb") or p.ndim < 2:
            nodec.append(p)
        else:
            dec.append(p)
    groups = [{"name": "decay", "params": dec, "weight_decay": weight_decay, "lr_mult": 1.0},
              {"name": "no_decay", "params": nodec, "weight_decay": 0.0, "lr_mult": 1.0},
              {"name": "hidden_proj", "params": list(projections.parameters()), "weight_decay": 0.0, "lr_mult": 1.0},
              {"name": "depformer_decay", "params": dep_dec, "weight_decay": weight_decay, "lr_mult": depth_mult},
              {"name": "depformer_no_decay", "params": dep_nodec, "weight_decay": 0.0, "lr_mult": depth_mult}]
    return [g for g in groups if g["params"]]


def lr_factor(step: int, total: int, warmup: int, min_ratio: float) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    prog = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))


def phase_for(step: int, total: int, fixed: tp.Optional[str]) -> PhaseSpec:
    if fixed:
        return next(p for p in PHASES if p.name == fixed)
    return phase_at(min(1.0, step / max(1, total)))


def git_rev(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def train(args):
    di = setup_distributed(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rl = RunLogger(out, di)
    torch.manual_seed(args.seed + di.rank)
    np.random.seed(args.seed + di.rank)
    if di.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---------------------------------------------------------------- smoke overrides
    if args.smoke_test:
        args.total_steps = args.smoke_steps
        args.eval_every = max(2, args.smoke_steps // 3)
        args.save_every = max(2, args.smoke_steps // 3)
        args.keep_every = 0
        args.val_samples = min(args.val_samples, 4 * di.world)
        args.warmup_steps = min(args.warmup_steps, 2)
        args.log_every = 1
        args.calib_batches = min(args.calib_batches, 2)
        args.calib_frames = min(args.calib_frames, 300)
        args.onpolicy_fraction = 1.0          # every P3/P4 smoke step exercises a rollout
        if di.device.type == "cpu":           # CPU tests only; on GPU keep the REAL rollout length so the smoke
            args.rollout_frames = min(args.rollout_frames, 24)          # run measures true on-policy step time
            args.onpolicy_context_frames = min(args.onpolicy_context_frames, 8)

    # ---------------------------------------------------------------- batch geometry
    micro = args.micro_batch
    if args.batch_size % (di.world * micro) != 0:
        raise SystemExit(f"--batch-size {args.batch_size} must be a multiple of world_size x --micro-batch "
                         f"= {di.world} x {micro}")
    accum = args.batch_size // (di.world * micro)

    # ---------------------------------------------------------------- data
    train_ds = TeacherTokenDataset(args.data_dir)
    val_ds = TeacherTokenDataset(args.val_dir) if args.val_dir and Path(args.val_dir, "manifest.jsonl").exists() else None
    rep = {"train": analyze_dataset(train_ds, args.chunk_frames, args.data_dir)}
    if val_ds is not None:
        rep["val"] = analyze_dataset(val_ds, args.chunk_frames, args.val_dir)
    if rep["train"]["samples_shorter_than_chunk"]:
        raise SystemExit(f"{rep['train']['samples_shorter_than_chunk']} training samples are shorter than "
                         f"--chunk-frames {args.chunk_frames}; use the dataset's frame count (main: 1000, long: 1500)")
    if val_ds is not None and rep["val"]["samples_shorter_than_chunk"]:
        raise SystemExit("validation samples shorter than --chunk-frames")
    if not rep["train"]["has_prefix_len"]:
        logger.warning("manifest has no prefix_len (older dataset format): the prompt prefix can't be masked and "
                       "on-policy rollouts start after --onpolicy-context-frames only")
    student_cfg = load_student_config(args.student_config)
    if args.chunk_frames + 2 > student_cfg.max_frames:
        raise SystemExit(f"--chunk-frames {args.chunk_frames} exceeds the student's context ({student_cfg.max_frames})")
    sampler = GlobalBatchSampler(len(train_ds), args.batch_size, args.seed)
    rep["geometry"] = {"world_size": di.world, "micro_batch": micro, "grad_accum": accum,
                       "global_batch": args.batch_size, "steps_per_epoch": round(len(train_ds) / args.batch_size, 2),
                       "epochs_total": round(args.total_steps * args.batch_size / len(train_ds), 2)}
    rl.write_json("data_report.json", rep)
    logger.info("data: %s", json.dumps(rep))

    # ---------------------------------------------------------------- models
    teacher = load_teacher(args, di.device)
    tkw = json.loads(args.teacher_kwargs_json) if args.teacher_kwargs_json else None
    student = build_student_lm(student_cfg, args.teacher_checkpoint, device=di.device, dtype=torch.bfloat16,
                               teacher_kwargs=tkw)
    ckpt_path = out / "student_checkpoint.pt"
    payload = None
    start_step = 0
    warm_has_depformer = False
    if args.resume and ckpt_path.exists():
        payload = ckpt.load_training_payload(str(ckpt_path))
        if payload.get("format") != "ppx_student_train_v2":
            raise SystemExit(f"{ckpt_path} was written by the old trainer; use --init-from {ckpt_path} to warm-start "
                             "from its weights in a NEW --output-dir")
        start_step = int(payload["step"])
        selected_layers = payload["selected_teacher_layers"]
        rl.event("resume", step=start_step, checkpoint=str(ckpt_path))
    elif args.init_from:
        warm = ckpt.load_training_payload(args.init_from)
        warm_has_depformer = bool(warm.get("depformer_trained"))
        selected_layers = warm["selected_teacher_layers"]
        ckpt.load_trainable_state_dict(student, warm["student_state_dict"])
        rl.event("init_from", path=args.init_from, source_step=warm.get("step"),
                 depformer_included=bool(warm.get("depformer_trained")))
    else:
        # --------- data-driven init from the teacher on rank 0 only; DDP then broadcasts rank 0's weights
        init_path = out / "init_student.pt"
        if di.main:
            if init_path.exists():
                ini = ckpt.load_training_payload(str(init_path))
                ckpt.load_trainable_state_dict(student, ini["student_state_dict"])
                selected_layers = ini["selected_teacher_layers"]
                rl.event("init_reused", path=str(init_path))
            else:
                t0 = time.time()
                calib = []
                g = np.random.default_rng([args.seed, 12345]).permutation(len(train_ds))
                for b in range(args.calib_batches):
                    idx = [int(g[(b * 4 + i) % len(train_ds)]) for i in range(4)]
                    calib.append(collate_chunks([train_ds[i] for i in idx], args.calib_frames)["codes"].to(di.device))
                diag = initialize_student(student, teacher, calib, student_cfg.init.keep_first,
                                          student_cfg.init.keep_last)
                selected_layers = [int(x) for x in diag["selected_layers"]]
                torch.save({"student_state_dict": ckpt.trainable_state_dict(student),
                            "selected_teacher_layers": selected_layers, "diag": {k: v for k, v in diag.items()
                                                                                 if k != "layer_scores"}},
                           init_path)
                rl.event("student_initialized", seconds=round(time.time() - t0, 1), selected_layers=selected_layers,
                         bridge_residual=diag.get("bridge_lstsq_residual"))
                del calib
        obj = [selected_layers if di.main else None]
        if di.world > 1:
            dist.broadcast_object_list(obj, src=0)
        selected_layers = obj[0]
    if len(selected_layers) != student_cfg.num_hidden_layers:
        raise SystemExit(f"selected_teacher_layers has {len(selected_layers)} entries, student has "
                         f"{student_cfg.num_hidden_layers} layers")

    # ---------------------------------------------------------------- precision, depformer state, projections
    def unfreeze_depformer():
        student.unfreeze_depth_transformer()
        if args.master_dtype == "fp32":
            student.depformer.float()          # in-place dtype change keeps the Parameter objects (optimizer refs)

    # include_depformer: the depformer weights differ from the teacher's (trained in P4 now or in the run we
    #   warm-started from) -> they must be saved in checkpoints and shipped in the export.
    # depformer_unfrozen: the depformer is trainable in THIS run (Phase 4 of this schedule).
    include_depformer = bool((payload or {}).get("depformer_trained")) or warm_has_depformer
    phase0 = phase_for(start_step, args.total_steps, args.fixed_phase)
    depformer_unfrozen = False
    if phase0.depth_transformer_unfrozen:
        unfreeze_depformer()
        depformer_unfrozen = include_depformer = True
    if args.master_dtype == "fp32":
        for name in TRAINABLE_TOP:
            getattr(student, name).float()
    if payload is not None:
        ckpt.load_trainable_state_dict(student, payload["student_state_dict"])
    student.transformer.gradient_checkpointing = bool(args.grad_checkpointing)
    student.train()

    projections = nn.ModuleList([HiddenProjection(student_cfg.hidden_size, student_cfg.teacher_reference.dim)
                                 for _ in selected_layers]).to(device=di.device, dtype=torch.float32)
    if payload is not None:
        projections.load_state_dict(payload["hidden_projections_state_dict"])

    groups = param_groups(student, projections, args.weight_decay, 0.1)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                                  fused=(di.device.type == "cuda"))
    for g in optimizer.param_groups:
        g["base_lr"] = args.lr * g["lr_mult"]
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for g in optimizer.param_groups:                 # base lr follows the CURRENT --lr
            g["base_lr"] = args.lr * g["lr_mult"]

    norms = {k: SyncedNorm() for k in ("ce", "kl", "bridge", "hidden")}
    if payload is not None:
        for k, st in (payload.get("running_norm_states") or {}).items():
            if k in norms and st:
                norms[k].norm.load_state_dict(st)

    wrapper = StudentWithProjections(student, projections)

    def wrap_ddp(find_unused: bool):
        if di.world == 1:
            return wrapper
        return DDP(wrapper, device_ids=[di.local_rank] if di.device.type == "cuda" else None,
                   find_unused_parameters=find_unused, broadcast_buffers=False)
    # After Phase 4 unfreezes the depformer, its steps for the (undistilled) user codebooks get no gradient ->
    # find_unused_parameters is required from then on.
    model = wrap_ddp(find_unused=depformer_unfrozen and not args.distill_user_stream)

    if di.device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info(di.device)
        static = {"device": str(di.device), "allocated_gb": round(torch.cuda.memory_allocated(di.device) / 1e9, 2),
                  "reserved_gb": round(torch.cuda.memory_reserved(di.device) / 1e9, 2),
                  "device_free_gb": round(free_b / 1e9, 2), "device_total_gb": round(total_b / 1e9, 2)}
        logger.info("static GPU memory after model setup: %s", json.dumps(static))
        rl.event("static_memory_rank0", **static)
    trainable = [p for g in optimizer.param_groups for p in g["params"]]
    n_params = {"student_trainable": sum(p.numel() for p in student.parameters() if p.requires_grad),
                "hidden_projections": sum(p.numel() for p in projections.parameters()),
                "teacher": sum(p.numel() for p in teacher.parameters())}
    autocast_ctx = (lambda: torch.autocast(device_type=di.device.type, dtype=torch.bfloat16)) \
        if args.master_dtype == "fp32" else contextlib.nullcontext
    sampling = {"temp": 0.8, "temp_text": 0.7, "top_k": 250, "top_k_text": 25}

    run_cfg = {"args": vars(args), "world_size": di.world, "grad_accum": accum, "host": socket.gethostname(),
               "torch": torch.__version__, "cuda": torch.version.cuda, "git": git_rev(Path(__file__).parents[1]),
               "selected_teacher_layers": selected_layers, "params": n_params, "start_step": start_step,
               "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
               if torch.cuda.is_available() else []}
    rl.write_json("run_config.json", run_cfg)
    rl.event("start", **{k: run_cfg[k] for k in ("world_size", "grad_accum", "start_step", "params")},
             total_steps=args.total_steps)

    stop = {"flag": False}

    def on_signal(signum, frame):
        stop["flag"] = True
        logger.warning("signal %s received: saving a checkpoint and stopping after this step", signum)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def check_ranks_in_sync(step: int):
        """All DDP ranks must hold identical weights. Compare a float64 checksum of every trainable tensor."""
        with torch.no_grad():
            chk = torch.zeros(1, device=di.device, dtype=torch.float64)
            for p in trainable:
                chk += p.detach().double().sum()
            if di.world > 1:
                lo, hi = chk.clone(), chk.clone()
                dist.all_reduce(lo, op=dist.ReduceOp.MIN)
                dist.all_reduce(hi, op=dist.ReduceOp.MAX)
                ok = bool((hi - lo).abs().item() <= 1e-6 * max(1.0, abs(hi.item())))
            else:
                ok, lo, hi = True, chk, chk
        logger.info("param checksum step %d: %.12e (ranks in sync: %s)", step, chk.item(), ok)
        if not ok:
            rl.event("rank_desync", step=step, min=lo.item(), max=hi.item())
        return ok

    def save(step: int, tag: str = "latest"):
        check_ranks_in_sync(step)
        if di.main:
            ckpt.save_training_state(
                str(ckpt_path), step=step, student=student, hidden_projections=projections, optimizer=optimizer,
                lr_scheduler=_LRState(step), schedule_state={"phase": phase_for(max(0, step - 1), args.total_steps,
                                                                                 args.fixed_phase).name},
                running_norm_states={k: v.norm.state_dict() for k, v in norms.items()},
                selected_teacher_layers=selected_layers, depformer_trained=include_depformer,
                train_args=vars(args), extra={"best_val": best["value"], "tag": tag})
            if tag == "keep":
                (out / "checkpoints").mkdir(exist_ok=True)
                shutil.copy2(ckpt_path, out / "checkpoints" / f"step_{step:07d}.pt")
            if tag == "best":
                shutil.copy2(ckpt_path, out / "best.pt")
            rl.event("checkpoint", step=step, tag=tag)
        barrier(di)

    best = {"value": (payload or {}).get("extra", {}).get("best_val") if payload else None}

    @torch.no_grad()
    def evaluate(step: int, phase: PhaseSpec):
        if val_ds is None:
            return
        wrapper.eval()
        idx = list(range(min(args.val_samples, len(val_ds))))[di.rank::di.world]
        keys = ["kl_text", "kl_audio", "ce_text", "ce_audio", "bridge", "hidden", "agree_text", "agree_cb0"]
        sums = torch.zeros(len(keys) + 1, device=di.device, dtype=torch.float64)
        for k in range(0, len(idx), micro):
            b = load_batch(val_ds, idx[k:k + micro], args.chunk_frames, di.device)
            codes, pl = b["codes"], b["prefix_lens"]
            t_cap = run_forward_train_with_hooks(teacher, codes, selected_layers)
            with autocast_ctx():
                s_out = wrapper(codes)
            T = codes.shape[-1]
            conv = (torch.arange(T, device=di.device)[None] >= pl[:, None])
            tm = s_out["text_logits_mask"] & conv[:, None]
            am = s_out["logits_mask"][:, :N_AGENT_CODEBOOKS] & conv[:, None]
            s_a, t_a = s_out["logits"][:, :N_AGENT_CODEBOOKS], t_cap.logits[:, :N_AGENT_CODEBOOKS]
            vals = [kl_fp32(s_out["text_logits"], t_cap.text_logits, tm, TEXT_KL_TEMPERATURE),
                    kl_fp32(s_a, t_a, am, args.audio_kl_temperature),
                    masked_ce(s_out["text_logits"], codes[:, :1], tm & (codes[:, :1] != student.text_initial_token_id)),
                    masked_ce(s_a, codes[:, 1:1 + N_AGENT_CODEBOOKS], am & (codes[:, 1:1 + N_AGENT_CODEBOOKS] != student.initial_token_id)),
                    bridge_loss(s_out["raw_hidden"], t_cap.raw_hidden),
                    torch.stack([hidden_term(s_out["projected"][i], t_cap.layer_hidden[selected_layers[i]])
                                 for i in range(len(selected_layers))]).mean()]
            st = torch.nan_to_num(s_out["text_logits"].float(), nan=0).argmax(-1)
            tt = torch.nan_to_num(t_cap.text_logits.float(), nan=0).argmax(-1)
            sa = torch.nan_to_num(s_a[:, 0].float(), nan=0).argmax(-1)
            ta = torch.nan_to_num(t_a[:, 0].float(), nan=0).argmax(-1)
            vals.append(((st == tt) & tm).sum().float() / tm.sum().clamp_min(1))
            vals.append(((sa == ta) & am[:, 0]).sum().float() / am[:, 0].sum().clamp_min(1))
            n = codes.shape[0]
            for i, v in enumerate(vals):
                sums[i] += float(v) * n
            sums[-1] += n
        all_reduce(sums, di)
        wrapper.train()
        if sums[-1] == 0:
            return
        res = {k: float(sums[i] / sums[-1]) for i, k in enumerate(keys)}
        score = res["kl_text"] + res["kl_audio"]
        is_best = best["value"] is None or score < best["value"]
        row = {"step": step, "phase": phase.name, **{f"val_{k}": round(v, 5) for k, v in res.items()},
               "val_samples": int(sums[-1]), "best": int(is_best)}
        rl.val_row(row)
        logger.info("VAL step %d %s", step, json.dumps(row))
        if is_best:
            best["value"] = score
            save(step, "best")

    # ---------------------------------------------------------------- training loop
    skipped_total, consecutive_skips = 0, 0
    phase_times: dict[str, list[float]] = {}
    peak_all = 0.0
    last_phase = None
    step = start_step
    initial_audio_id, text_initial_id = student.initial_token_id, student.text_initial_token_id
    t_run = time.time()
    while step < args.total_steps:
        phase = phase_for(step, args.total_steps, args.fixed_phase)
        if phase.name != last_phase:
            rl.event("phase", step=step, phase=phase.name, weights={k: getattr(phase, k) for k in
                                                                   ("alpha", "beta", "gamma", "delta", "epsilon")},
                     on_policy=phase.on_policy, depformer_unfrozen=phase.depth_transformer_unfrozen)
            last_phase = phase.name
        if phase.depth_transformer_unfrozen and not depformer_unfrozen:
            unfreeze_depformer()
            depformer_unfrozen = include_depformer = True
            trainable = [p for g in optimizer.param_groups for p in g["params"]]
            model = wrap_ddp(find_unused=not args.distill_user_stream)
            rl.event("depformer_unfrozen", step=step, lr_mult=0.1)
        onpolicy = phase.on_policy and args.onpolicy_fraction > 0 and \
            ((step * 2654435761) % 1000) / 1000.0 < args.onpolicy_fraction      # same decision on every rank
        f = lr_factor(step, args.total_steps, args.warmup_steps, args.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * f
        if di.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        gidx = sampler.global_indices(step)
        mine = gidx[di.rank * micro * accum:(di.rank + 1) * micro * accum]
        optimizer.zero_grad(set_to_none=True)
        acc = {k: 0.0 for k in ("total", "ce", "kl_audio", "kl_text", "bridge", "hidden")}
        local_ok = True
        n_frames = 0
        for m in range(accum):
            b = load_batch(train_ds, mine[m * micro:(m + 1) * micro], args.chunk_frames, di.device)
            codes, tmask, pl = b["codes"], b["transition_mask"], b["prefix_lens"]
            ce_end = None
            if onpolicy:
                start = int(pl.max()) + args.onpolicy_context_frames
                n_total = min(codes.shape[-1], start + args.rollout_frames)
                if n_total - start >= 8:
                    torch.manual_seed(args.seed * 1_000_003 + step * 997 + di.rank * 31 + m)
                    gen = student_rollout(student, codes, start, n_total, autocast_ctx(), sampling)
                    if not (torch.equal(gen[:, :, :start], codes[:, :, :start])
                            and torch.equal(gen[:, 9:, :n_total], codes[:, 9:, :n_total])):
                        raise RuntimeError("on-policy rollout is misaligned with the data (forced frames differ) -- "
                                           "this is a bug; refusing to train on it")
                    codes, tmask = gen, tmask[:, :n_total]
                    ce_end = torch.full((codes.shape[0],), start, device=di.device, dtype=torch.long)
            n_frames += codes.numel() // codes.shape[1]
            sync = model.no_sync() if (di.world > 1 and m < accum - 1) else contextlib.nullcontext()
            with sync:
                with torch.no_grad():
                    t_cap = run_forward_train_with_hooks(teacher, codes, selected_layers)
                with autocast_ctx():
                    s_out = model(codes)
                terms = loss_terms(s_out, t_cap, codes, tmask, pl, ce_end, selected_layers, args,
                                   initial_audio_id, text_initial_id)
                total = (phase.alpha * terms["ce"] / norms["ce"].scale(terms["ce"].item())
                         + phase.beta * terms["kl"] / norms["kl"].scale(terms["kl"].item())
                         + phase.gamma * terms["bridge"] / norms["bridge"].scale(terms["bridge"].item())
                         + phase.delta * terms["hidden"] / norms["hidden"].scale(terms["hidden"].item()))
                # L_speaker (phase.epsilon) is a validation metric only in this repo (see losses.py) -> not trained.
                if not torch.isfinite(total):
                    local_ok = False
                (total / accum).backward()
            for k in ("ce", "kl_audio", "kl_text", "bridge", "hidden"):
                acc[k] += float(terms[k].detach()) / accum
            acc["total"] += float(total.detach()) / accum
            del t_cap, s_out, terms, total

        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in trainable if p.grad is not None], args.max_grad_norm)
        flag = torch.tensor([1.0 if (local_ok and torch.isfinite(grad_norm)) else 0.0], device=di.device)
        all_reduce(flag, di, dist.ReduceOp.MIN if di.world > 1 else None)
        if flag.item() > 0:
            optimizer.step()
            consecutive_skips = 0
        else:
            optimizer.zero_grad(set_to_none=True)
            skipped_total += 1
            consecutive_skips += 1
            rl.event("skipped_nonfinite_step", step=step, local_ok=local_ok, grad_norm=float(grad_norm))
            if consecutive_skips >= args.max_consecutive_skips:
                save(step, "latest")
                raise RuntimeError(f"{consecutive_skips} consecutive non-finite steps -- aborting (checkpoint saved)")

        # ---- metrics: all-reduce means, update the shared loss normalizers
        vec = torch.tensor([acc[k] for k in ("total", "ce", "kl_audio", "kl_text", "bridge", "hidden")] + [n_frames],
                           device=di.device, dtype=torch.float64)
        all_reduce(vec, di)
        vals = (vec[:6] / di.world).tolist()
        g_frames = float(vec[6])
        if flag.item() > 0:
            norms["ce"].update(vals[1])
            norms["kl"].update(vals[2] + vals[3])
            norms["bridge"].update(vals[4])
            norms["hidden"].update(vals[5])
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9 if di.device.type == "cuda" else 0.0
        pk = torch.tensor([peak], device=di.device)
        all_reduce(pk, di, dist.ReduceOp.MAX if di.world > 1 else None)
        peak_all = max(peak_all, float(pk))
        key = f"{phase.name}{'_onpolicy' if onpolicy else ''}"
        phase_times.setdefault(key, []).append(dt)
        step += 1
        if step % args.log_every == 0 or step == args.total_steps or step == start_step + 1:
            row = {"step": step, "phase": phase.name, "total": round(vals[0], 5), "ce_raw": round(vals[1], 5),
                   "kl_audio_raw": round(vals[2], 5), "kl_text_raw": round(vals[3], 5), "bridge_raw": round(vals[4], 5),
                   "hidden_raw": round(vals[5], 5), "seconds_per_step": round(dt, 3), "lr": f"{args.lr * f:.3e}",
                   "grad_norm": round(float(grad_norm), 4), "onpolicy": int(onpolicy),
                   "frames_per_s": round(g_frames / max(dt, 1e-6), 1), "peak_mem_gb": round(float(pk), 2),
                   "skipped_total": skipped_total, "epoch": round(sampler.epoch_of(step), 3)}
            rl.train_row(row)
            logger.info("step %d/%d %s total=%.4f ce=%.4f kl=%.4f+%.4f bridge=%.4f hidden=%.4f gn=%.3f lr=%s "
                        "%.2fs/step peak=%.1fGB%s", step, args.total_steps, phase.name, vals[0], vals[1], vals[2],
                        vals[3], vals[4], vals[5], float(grad_norm), row["lr"], dt, float(pk),
                        " [on-policy]" if onpolicy else "")
            # ETA from measured per-phase step times
            eta = 0.0
            for s2 in range(step, args.total_steps, max(1, (args.total_steps - step) // 200 or 1)):
                ph = phase_for(s2, args.total_steps, args.fixed_phase).name
                cand = [v for k2, v in phase_times.items() if k2.startswith(ph)]
                mean = np.mean([x for v in cand for x in v[-50:]]) if cand else dt
                eta += mean * max(1, (args.total_steps - step) // 200 or 1)
            rl.write_json("heartbeat.json", {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": step,
                                             "total_steps": args.total_steps, "phase": phase.name,
                                             "eta_hours": round(eta / 3600, 2), "skipped_total": skipped_total,
                                             "peak_mem_gb": round(peak_all, 2), "best_val": best["value"]})
        stop_t = torch.tensor([1.0 if stop["flag"] else 0.0], device=di.device)
        all_reduce(stop_t, di, dist.ReduceOp.MAX if di.world > 1 else None)
        if args.eval_every and (step % args.eval_every == 0 or step == args.total_steps):
            evaluate(step, phase)
        if stop_t.item() > 0:
            save(step, "latest")
            rl.event("stopped_by_signal", step=step)
            break
        if step % args.save_every == 0 or step == args.total_steps:
            save(step, "latest")
        if args.keep_every and step % args.keep_every == 0:
            save(step, "keep")

    # ---------------------------------------------------------------- export + summary
    finished = step >= args.total_steps
    if finished and di.main:
        exp_dir = out / "export"
        exp_dir.mkdir(exist_ok=True)
        from .export import export_bf16
        export_bf16(student, str(exp_dir / f"{student_cfg.name}_bf16.safetensors"), selected_layers,
                    include_depformer=include_depformer)
        if (out / "best.pt").exists():
            # the final weights are already exported above, so the live model can be overwritten with the best
            # checkpoint's weights (no second 17 GB teacher load into CPU RAM). If best.pt predates Phase 4 it has
            # no depformer keys -> exported without depformer -> inference uses the teacher's original one, exactly
            # the state that checkpoint was evaluated in.
            bp = ckpt.load_training_payload(str(out / "best.pt"))
            ckpt.load_trainable_state_dict(student, bp["student_state_dict"])
            export_bf16(student, str(exp_dir / f"{student_cfg.name}_best_bf16.safetensors"), selected_layers,
                        include_depformer=bool(bp.get("depformer_trained")))
        rl.event("exported", dir=str(exp_dir), depformer_included=include_depformer)
    summary = {"finished": finished, "steps_done": step, "start_step": start_step, "skipped_nonfinite": skipped_total,
               "peak_mem_gb": round(peak_all, 2), "best_val_kl": best["value"], "world_size": di.world,
               "seconds": round(time.time() - t_run, 1),
               "mean_step_seconds_by_phase": {k: round(float(np.mean(v)), 3) for k, v in phase_times.items()},
               "depformer_included": include_depformer, "depformer_trained_this_run": depformer_unfrozen}
    rl.write_json("smoke_summary.json" if args.smoke_test else "run_summary.json", summary)
    rl.event("end", **summary)
    barrier(di)
    if di.world > 1:
        dist.destroy_process_group()


class _LRState:
    """The LR is a pure function of the step (lr_factor), so the 'scheduler state' is just the step."""

    def __init__(self, step):
        self.step = step

    def state_dict(self):
        return {"step": self.step}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- original arguments (unchanged meaning, except --batch-size is now the GLOBAL batch)
    ap.add_argument("--student-config", required=True, help="e.g. student_ppx_s")
    ap.add_argument("--teacher-checkpoint", required=True)
    ap.add_argument("--data-dir", required=True, help="training split dir (manifest.jsonl + codes/)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--total-steps", type=int, default=200_000, help="optimizer steps")
    ap.add_argument("--batch-size", type=int, default=8, help="GLOBAL batch = world x micro-batch x grad-accum")
    ap.add_argument("--chunk-frames", type=int, default=1000, help="must equal the dataset's frames (main 1000 / long 1500)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--audio-kl-temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--smoke-test", action="store_true", help="all phases (incl. on-policy), eval, checkpoint, export in --smoke-steps steps")
    # --- new
    ap.add_argument("--val-dir", default=None)
    ap.add_argument("--micro-batch", type=int, default=1, help="samples per GPU per forward")
    ap.add_argument("--master-dtype", choices=["fp32", "bf16"], default="fp32",
                    help="fp32 = fp32 master weights + bf16 autocast (recommended); bf16 = legacy, lower memory")
    ap.add_argument("--grad-checkpointing", action="store_true", help="recompute student layer activations in backward")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--val-samples", type=int, default=256)
    ap.add_argument("--keep-every", type=int, default=5000, help="also keep checkpoints/step_N.pt every N steps (0 = off)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--onpolicy-fraction", type=float, default=0.5, help="share of P3/P4 steps that use student rollouts")
    ap.add_argument("--rollout-frames", type=int, default=250, help="frames the student generates per rollout (20 s)")
    ap.add_argument("--onpolicy-context-frames", type=int, default=50,
                    help="real conversation frames after the longest prefix before the student takes over")
    ap.add_argument("--ce-on-prefix", action="store_true", help="also apply hard-label CE on prompt-prefix frames")
    ap.add_argument("--no-kl-on-prefix", dest="kl_on_prefix", action="store_false",
                    help="exclude prompt-prefix frames from KL too (default: included)")
    ap.add_argument("--distill-user-stream", action="store_true",
                    help="also distill the teacher's user-stream codebooks (off: never used at inference, see docstring)")
    ap.add_argument("--calib-batches", type=int, default=16, help="init calibration batches (4 samples each)")
    ap.add_argument("--calib-frames", type=int, default=500, help="frames per calibration sample")
    ap.add_argument("--init-from", default=None, help="warm-start weights from a checkpoint (new run, new optimizer)")
    ap.add_argument("--fixed-phase", default=None, choices=[p.name for p in PHASES],
                    help="use one phase for the whole run (e.g. P2_behavior for the long-context fine-tune)")
    ap.add_argument("--max-consecutive-skips", type=int, default=50)
    ap.add_argument("--smoke-steps", type=int, default=12)
    ap.add_argument("--teacher-kwargs-json", default=None, help=argparse.SUPPRESS)   # tests only
    args = ap.parse_args()
    try:
        train(args)
    except Exception:
        logging.getLogger("distill.train").error("FATAL:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
