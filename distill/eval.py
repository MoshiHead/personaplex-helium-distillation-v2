# SPDX-License-Identifier: MIT
"""Evaluation metrics for a distilled student: WER, WavLM-TDNN speaker
similarity, UTMOS, barge-in latency, and RTF against the 80ms frame budget.

No numbers are hardcoded anywhere in this file -- every metric is actually
measured against whatever model/audio is passed in. Optional heavy
dependencies (`transformers` for Whisper/WavLM, `speechmos` for UTMOS) are
imported lazily so importing this module doesn't require them unless the
corresponding metric is actually used.
"""

from dataclasses import dataclass, asdict
import logging
import time
import typing as tp

import numpy as np
import torch

from moshi.models.lm import LMGen
from moshi.models.compression import MimiModel

from .losses import WavLMSpeakerSimilarity

logger = logging.getLogger(__name__)

FRAME_BUDGET_MS = 80.0  # 12.5 Hz Mimi frame rate


@dataclass
class LatencyStats:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float

    @staticmethod
    def from_samples(samples_ms: list[float]) -> "LatencyStats":
        arr = np.array(samples_ms)
        return LatencyStats(
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            mean_ms=float(arr.mean()),
        )


class WhisperWER:
    """Word error rate via `openai/whisper-large-v3` (lazy-loaded)."""

    MODEL_NAME = "openai/whisper-large-v3"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        from transformers import pipeline

        self._pipe = pipeline("automatic-speech-recognition", model=self.MODEL_NAME, device=self.device)

    def transcribe(self, wav: np.ndarray, sample_rate: int) -> str:
        self._load()
        return self._pipe({"array": wav, "sampling_rate": sample_rate})["text"].strip()

    @staticmethod
    def word_error_rate(reference: str, hypothesis: str) -> float:
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        # Standard Levenshtein edit distance over words.
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
        d[:, 0] = np.arange(len(ref_words) + 1)
        d[0, :] = np.arange(len(hyp_words) + 1)
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
        return float(d[-1, -1]) / max(1, len(ref_words))

    def wer(self, reference_text: str, wav: np.ndarray, sample_rate: int) -> float:
        hypothesis = self.transcribe(wav, sample_rate)
        return self.word_error_rate(reference_text, hypothesis)


class UTMOSPredictor:
    """Mean opinion score prediction via the `speechmos` package's UTMOS model
    (`pip install speechmos`), lazy-loaded."""

    def __init__(self):
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return
        import speechmos.utmos as utmos

        self._predictor = utmos

    def score(self, wav: np.ndarray, sample_rate: int) -> float:
        self._load()
        result = self._predictor.compute(wav, sample_rate)
        return float(result["mos"])


@torch.no_grad()
def measure_barge_in_latency(
    lm_gen: LMGen,
    mimi: MimiModel,
    interrupt_input_track: torch.Tensor,
    onset_frame: int,
    silence_token_check: tp.Callable[[torch.Tensor], bool],
    max_lookahead_frames: int = 50,
) -> tp.Optional[float]:
    """Frames from a scripted interruption onset to the first output frame that
    breaks from the pre-interruption (silence/backchannel-only) pattern,
    converted to milliseconds via the Mimi frame rate. Returns None if the
    model never reacts within `max_lookahead_frames`.

    `silence_token_check(tokens)` should return True if `tokens` (the model's
    sampled agent-side codebooks for one frame) look like "no reaction yet"
    (e.g. still emitting the exact silence tokens) -- see
    distill/init_from_teacher.py's SILENCE_TOKENS-based check for the pattern.
    """
    frame_ms = 1000.0 / mimi.frame_rate
    for offset in range(max_lookahead_frames):
        c = onset_frame + offset
        if c >= interrupt_input_track.shape[-1]:
            break
        tokens = lm_gen.step(input_tokens=interrupt_input_track[:, :, c:c + 1])
        if tokens is not None and not silence_token_check(tokens):
            return offset * frame_ms
    return None


@torch.no_grad()
def measure_rtf(
    lm_gen: LMGen,
    mimi: MimiModel,
    num_frames: int,
    device: str,
) -> dict[str, LatencyStats]:
    """Per-component p50/p95/p99 latency (mimi_encode, temporal, bridge+depth,
    mimi_decode, end-to-end). RTF = mean(end_to_end_ms) / FRAME_BUDGET_MS.

    "temporal" and "bridge+depth" are split by inlining `LMGen.step` (which
    normally times as one opaque call) into its two underlying stages: the main
    transformer (`state.graphed_main`, covers the student's GQA transformer +
    Bridge + frozen out_norm/text_linear for the student, or just the plain
    transformer for the teacher) and the depth transformer
    (`state.graphed_depth`, always frozen). This mirrors exactly what
    `LMGen.step` / `LMGen.process_transformer_output` already do internally --
    no changes to moshi/models/lm.py.
    """
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    timings = {"mimi_encode": [], "temporal": [], "bridge_depth": [], "mimi_decode": [], "end_to_end": []}

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    silence = torch.zeros(1, 1, frame_size, device=device)
    state = lm_gen._streaming_state
    for _ in range(num_frames):
        sync()
        t0 = time.perf_counter()
        codes = mimi.encode(silence)
        sync()
        t1 = time.perf_counter()

        prepared = lm_gen.prepare_step_input(codes[:, :, 0:1])
        if prepared is None:
            sync()
            t2 = t3 = t4 = time.perf_counter()
        else:
            input_, provided_, target_, model_input_position, target_position = prepared
            transformer_out, text_logits = state.graphed_main(input_)
            sync()
            t2 = time.perf_counter()
            tokens = lm_gen.process_transformer_output(
                transformer_out, text_logits, provided_, target_, model_input_position, target_position,
            )
            sync()
            t3 = time.perf_counter()
            if isinstance(tokens, tuple):
                tokens = tokens[0]
            if tokens is not None:
                _ = mimi.decode(tokens[:, 1:9])
            sync()
            t4 = time.perf_counter()

        timings["mimi_encode"].append((t1 - t0) * 1000)
        timings["temporal"].append((t2 - t1) * 1000)
        timings["bridge_depth"].append((t3 - t2) * 1000)
        timings["mimi_decode"].append((t4 - t3) * 1000)
        timings["end_to_end"].append((t4 - t0) * 1000)

    return {name: LatencyStats.from_samples(vals) for name, vals in timings.items()}


def print_metrics_table(metrics: dict[str, tp.Any]):
    print(f"{'metric':<28}{'value':>16}")
    print("-" * 44)
    for name, value in metrics.items():
        if isinstance(value, LatencyStats):
            for field, v in asdict(value).items():
                print(f"{name + '.' + field:<28}{v:>16.3f}")
        elif isinstance(value, float):
            print(f"{name:<28}{value:>16.4f}")
        else:
            print(f"{name:<28}{str(value):>16}")
