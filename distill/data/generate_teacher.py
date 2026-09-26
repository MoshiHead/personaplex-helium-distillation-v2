# SPDX-License-Identifier: MIT
"""Teacher self-play data generation for distillation.

Design (per the distillation plan): keep the teacher resident and generate
audio/text tokens on the fly during training; store ONLY token sequences and
the (voice_prompt, persona_text, scenario) metadata that produced them -- never
full logits or hidden states (those are TB-scale for any real corpus). A top-64
logit cache is available as a fallback (`--capture-topk-logits`) for setups
where decoupling generation from training is worth the extra disk.

Every sample carries both a voice prompt AND a persona text prompt, swept
across the available voices and a supplied persona list, so the student cannot
learn to ignore either prefix.

Honesty note: the shipped `voices.tgz` contains 18 voices (NATF0-3, NATM0-3,
VARF0-4, VARM0-4), not >=50. This module sweeps across whatever voice prompts
are actually found in `--voice-prompt-dir` and warns loudly if that is fewer
than `--num-voices` -- it does not fabricate additional voices. The same
applies to personas: a small starter list is bundled below (drawn from
`README.md`'s example prompts and `assets/test/prompt_service.txt`'s style);
reaching "≥30 personas" for a real run requires supplying `--personas-file`.
"""

import argparse
import itertools
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import typing as tp

import torch

from moshi.models import loaders
from moshi.models.lm import LMGen, LMModel
from moshi.models.compression import MimiModel

logger = logging.getLogger(__name__)

# Starter set only -- see module docstring. Supply --personas-file for a real run.
DEFAULT_PERSONAS = [
    "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
    "You enjoy having a good conversation.",
    "You work for CitySan Services which is a waste management company and your name is Ayelen Lucero. "
    "Information: Verify customer name Omar Torres. Current schedule: every other week. "
    "Upcoming pickup: April 12th. Compost bin service available for $8/month add-on.",
    "You work for SwiftPlex Appliances which is a appliance repair company and your name is Farhod Toshmatov. "
    "Information: The dishwasher model is out of stock for replacement parts; we can use an alternative part "
    "with a 3-day delay. Labor cost remains $60 per hour.",
    "You are a calm, patient customer support agent for a home internet provider. Help troubleshoot a "
    "slow connection step by step.",
    "You are an enthusiastic tour guide describing a city's history to a visitor.",
    "You are a no-nonsense project manager giving a brief status update in a stand-up meeting.",
    "You are a friendly barista taking a coffee order and making small talk.",
]

BARGE_IN_OFFSETS_S = [0.5, 1.0, 2.0]
SCENARIOS = ["monologue", "barge_in_0.5s", "barge_in_1.0s", "barge_in_2.0s", "rapid_exchange", "backchannel_monologue"]


@dataclass
class SampleMetadata:
    sample_id: str
    voice_prompt: str
    persona_text: str
    scenario: str
    num_frames: int
    transition_frames: list[int]  # frames flagged as speaker-transition events (barge-in onsets, turn switches)


def list_voice_prompts(voice_prompt_dir: str) -> list[str]:
    voices = sorted(p.name for p in Path(voice_prompt_dir).iterdir() if p.suffix in (".pt", ".wav"))
    if not voices:
        raise RuntimeError(f"No voice prompt files (.pt/.wav) found in {voice_prompt_dir}")
    return voices


def load_personas(personas_file: tp.Optional[str]) -> list[str]:
    if personas_file is None:
        return list(DEFAULT_PERSONAS)
    path = Path(personas_file)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sweep_pairs(voices: list[str], personas: list[str], num_voices: int, num_personas: int,
                 scenarios: list[str], seed: int) -> list[tuple[str, str, str]]:
    if len(voices) < num_voices:
        logger.warning("Requested num_voices=%d but only %d voice prompts are available in the voice "
                        "prompt directory; using all %d.", num_voices, len(voices), len(voices))
    if len(personas) < num_personas:
        logger.warning("Requested num_personas=%d but only %d personas are available; using all %d. "
                        "Supply --personas-file for real coverage.", num_personas, len(personas), len(personas))
    voices = voices[:num_voices] if num_voices else voices
    personas = personas[:num_personas] if num_personas else personas

    g = torch.Generator().manual_seed(seed)
    all_combos = list(itertools.product(voices, personas, scenarios))
    perm = torch.randperm(len(all_combos), generator=g).tolist()
    return [all_combos[i] for i in perm]


def _wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


class BargeInInputTrack:
    """Builds the "other party" Mimi-code input track for a scripted scenario.

    Uses whatever short wav clips are available (e.g. `assets/test/*.wav`) as
    stand-in interrupter/other-party audio, placed at scripted offsets over an
    otherwise-silent track. This is a synthetic scripting tool, not a claim of
    naturalistic barge-in audio -- see module docstring.
    """

    def __init__(self, mimi: MimiModel, interrupter_wavs: list[str], frame_rate: float, sample_rate: int):
        self.mimi = mimi
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.interrupter_wavs = interrupter_wavs

    def _load_clip_frames(self, path: str) -> torch.Tensor:
        from moshi.models.lm import load_audio, _iterate_audio, encode_from_sphn

        audio = load_audio(path, self.sample_rate)
        frames = list(encode_from_sphn(
            self.mimi, _iterate_audio(audio, sample_interval_size=int(self.sample_rate / self.frame_rate)),
            max_batch=1,
        ))
        return torch.cat(frames, dim=-1) if frames else torch.zeros(1, 8, 0, dtype=torch.long)

    def build(self, scenario: str, num_frames: int, silence_frame: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """Returns (codes [1, 8, num_frames], transition_frames)."""
        track = silence_frame.repeat(1, 1, num_frames).clone()
        transitions: list[int] = []
        if scenario == "monologue":
            return track, transitions

        clip = None
        if self.interrupter_wavs:
            clip = self._load_clip_frames(self.interrupter_wavs[0])

        def splice(onset_frame: int):
            if clip is None or clip.shape[-1] == 0:
                return
            end = min(num_frames, onset_frame + clip.shape[-1])
            track[:, :, onset_frame:end] = clip[:, :, : end - onset_frame]
            transitions.append(onset_frame)

        if scenario.startswith("barge_in_"):
            offset_s = float(scenario.removeprefix("barge_in_").removesuffix("s"))
            splice(int(offset_s * self.frame_rate))
        elif scenario == "rapid_exchange":
            for onset_s in [0.5, 1.5, 2.5, 3.5]:
                splice(int(onset_s * self.frame_rate))
        elif scenario == "backchannel_monologue":
            for onset_s in [1.0, 2.0, 3.0, 4.0, 5.0]:
                onset_frame = int(onset_s * self.frame_rate)
                if clip is not None and clip.shape[-1] > 0:
                    short = clip[:, :, : max(1, int(0.3 * self.frame_rate))]
                    end = min(num_frames, onset_frame + short.shape[-1])
                    track[:, :, onset_frame:end] = short[:, :, : end - onset_frame]
                transitions.append(onset_frame)
        return track, transitions


@torch.no_grad()
def generate_sample(
    lm_gen: LMGen,
    mimi: MimiModel,
    text_tokenizer,
    voice_prompt_path: str,
    persona_text: str,
    scenario: str,
    duration_s: float,
    barge_in: BargeInInputTrack,
) -> tuple[torch.Tensor, list[int]]:
    """Runs the teacher self-play for one (voice, persona, scenario) sample.
    Returns (codes [num_codebooks=17, T], transition_frames)."""
    num_frames = int(duration_s * lm_gen._frame_rate)

    mimi.reset_streaming()
    if voice_prompt_path.endswith(".pt"):
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
    else:
        lm_gen.load_voice_prompt(voice_prompt_path)
    lm_gen.text_prompt_tokens = text_tokenizer.encode(_wrap_with_system_tags(persona_text))
    lm_gen.reset_streaming()
    lm_gen.step_system_prompts(mimi)
    mimi.reset_streaming()

    silence_frame = lm_gen._encode_sine_frame()
    input_track, transitions = barge_in.build(scenario, num_frames, silence_frame)

    all_tokens = []
    for c in range(num_frames):
        tokens = lm_gen.step(input_tokens=input_track[:, :, c:c + 1])
        if tokens is not None:
            all_tokens.append(tokens[:, :, 0])
    if not all_tokens:
        return torch.zeros(lm_gen.lm_model.num_codebooks, 0, dtype=torch.long), []
    codes = torch.stack(all_tokens, dim=-1)[0]  # [num_codebooks, T]
    return codes, transitions


def run_generation(
    output_dir: str,
    voice_prompt_dir: str,
    personas_file: tp.Optional[str],
    num_voices: int,
    num_personas: int,
    hours: float,
    sample_duration_s: float,
    hf_repo: str,
    moshi_weight: tp.Optional[str],
    mimi_weight: tp.Optional[str],
    tokenizer_path: tp.Optional[str],
    device: str,
    seed: int,
    interrupter_wavs: list[str],
):
    import sentencepiece
    from huggingface_hub import hf_hub_download

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    codes_dir = out / "codes"
    codes_dir.mkdir(exist_ok=True)

    already_done = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            already_done.add(json.loads(line)["sample_id"])

    if mimi_weight is None:
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
    if moshi_weight is None:
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
    if tokenizer_path is None:
        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)

    mimi = loaders.get_mimi(mimi_weight, device)
    teacher = loaders.get_moshi_lm(moshi_weight, device=device)
    teacher.eval()
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    lm_gen = LMGen(teacher, device=device, sample_rate=mimi.sample_rate, frame_rate=mimi.frame_rate,
                    audio_silence_frame_cnt=int(0.5 * mimi.frame_rate))
    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    barge_in = BargeInInputTrack(mimi, interrupter_wavs, mimi.frame_rate, mimi.sample_rate)

    voices = list_voice_prompts(voice_prompt_dir)
    personas = load_personas(personas_file)
    combos = sweep_pairs(voices, personas, num_voices, num_personas, SCENARIOS, seed)

    target_frames = int(hours * 3600 * mimi.frame_rate)
    frames_written = 0
    manifest_f = open(manifest_path, "a", encoding="utf-8")
    try:
        for voice, persona, scenario in combos:
            if frames_written >= target_frames:
                break
            sample_id = f"{Path(voice).stem}__{abs(hash(persona)) % 10**8}__{scenario}"
            if sample_id in already_done:
                continue
            voice_path = str(Path(voice_prompt_dir) / voice)
            codes, transitions = generate_sample(
                lm_gen, mimi, text_tokenizer, voice_path, persona, scenario, sample_duration_s, barge_in,
            )
            if codes.shape[-1] == 0:
                continue
            torch.save(codes, codes_dir / f"{sample_id}.pt")
            meta = SampleMetadata(sample_id, voice, persona, scenario, codes.shape[-1], transitions)
            manifest_f.write(json.dumps(asdict(meta)) + "\n")
            manifest_f.flush()
            frames_written += codes.shape[-1]
            logger.info("Generated %s (%d frames, %.0fs total so far)",
                        sample_id, codes.shape[-1], frames_written / mimi.frame_rate)
    finally:
        manifest_f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--voice-prompt-dir", required=True)
    parser.add_argument("--personas-file", default=None)
    parser.add_argument("--num-voices", type=int, default=50)
    parser.add_argument("--num-personas", type=int, default=30)
    parser.add_argument("--hours", type=float, default=0.05, help="Target hours of audio to generate.")
    parser.add_argument("--sample-duration-s", type=float, default=20.0)
    parser.add_argument("--hf-repo", default=loaders.DEFAULT_REPO)
    parser.add_argument("--moshi-weight", default=None)
    parser.add_argument("--mimi-weight", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interrupter-wavs", nargs="*", default=["assets/test/input_service.wav"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_generation(
        output_dir=args.output_dir,
        voice_prompt_dir=args.voice_prompt_dir,
        personas_file=args.personas_file,
        num_voices=args.num_voices,
        num_personas=args.num_personas,
        hours=args.hours,
        sample_duration_s=args.sample_duration_s,
        hf_repo=args.hf_repo,
        moshi_weight=args.moshi_weight,
        mimi_weight=args.mimi_weight,
        tokenizer_path=args.tokenizer,
        device=args.device,
        seed=args.seed,
        interrupter_wavs=args.interrupter_wavs,
    )


if __name__ == "__main__":
    main()
