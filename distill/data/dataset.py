# SPDX-License-Identifier: MIT
"""Reads the manifest + token-sequence files produced by `generate_teacher.py`.

Primary path: teacher logits are computed on the fly at train time from the
resident teacher model (see distill/train.py) -- this dataset only serves
token sequences and prompt/scenario metadata, matching the "store only token
sequences + prompts" storage policy.

Fallback path: `TopKLogitCache` lets a training run consume pre-computed
top-64 teacher logits instead, for setups where decoupling generation from
training is worth the extra disk (this is NOT the default).
"""

from dataclasses import dataclass
from pathlib import Path
import json
import typing as tp

import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    sample_id: str
    voice_prompt: str
    persona_text: str
    scenario: str
    codes: torch.Tensor  # [num_codebooks, T]
    transition_frames: list[int]

    def transition_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.codes.shape[-1], dtype=torch.bool)
        for f in self.transition_frames:
            if 0 <= f < mask.shape[0]:
                mask[f] = True
        return mask


class TeacherTokenDataset(Dataset):
    """One item per generated sample. `__getitem__` returns a `Sample`; batching
    (padding to a common length, chunking long sequences into training windows)
    is left to a `collate_fn` / sampler in `distill/train.py` since the right
    chunking policy depends on the training phase (teacher-forced windows vs.
    full-episode on-policy rollouts).
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.jsonl in {data_dir} -- run distill/data/generate_teacher.py first."
            )
        self.entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Sample:
        entry = self.entries[idx]
        codes = torch.load(self.data_dir / "codes" / f"{entry['sample_id']}.pt", map_location="cpu")
        return Sample(
            sample_id=entry["sample_id"],
            voice_prompt=entry["voice_prompt"],
            persona_text=entry["persona_text"],
            scenario=entry["scenario"],
            codes=codes,
            transition_frames=entry["transition_frames"],
        )

    def total_frames(self) -> int:
        return sum(e["num_frames"] for e in self.entries)


def collate_chunks(samples: list[Sample], chunk_frames: int) -> dict[str, torch.Tensor]:
    """Pads/truncates a list of `Sample`s to `chunk_frames` and stacks them into
    a batch. Samples shorter than `chunk_frames` are dropped (a training run
    should pick `chunk_frames` well below the shortest generated sample)."""
    usable = [s for s in samples if s.codes.shape[-1] >= chunk_frames]
    if not usable:
        raise ValueError(f"No samples have >= {chunk_frames} frames in this batch.")
    codes = torch.stack([s.codes[:, :chunk_frames] for s in usable], dim=0)
    transition_mask = torch.stack([s.transition_mask()[:chunk_frames] for s in usable], dim=0)
    return {"codes": codes, "transition_mask": transition_mask,
            "sample_ids": [s.sample_id for s in usable]}


class TopKLogitCache:
    """Fallback storage for pre-computed top-64 teacher logits, keyed by
    `sample_id`. Layout on disk: `<cache_dir>/<sample_id>.pt` holding a dict with
    `text_topk_ids/text_topk_vals` [T, 64] and `audio_topk_ids/audio_topk_vals`
    [K, T, 64]. NOT populated by default -- see module docstring.
    """

    TOPK = 64

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _topk(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vals, ids = torch.topk(logits.float(), TopKLogitCache.TOPK, dim=-1)
        return ids.to(torch.int32), vals.to(torch.float16)

    def write(self, sample_id: str, text_logits: torch.Tensor, audio_logits: torch.Tensor):
        text_ids, text_vals = self._topk(text_logits)
        audio_ids, audio_vals = self._topk(audio_logits)
        torch.save(
            {"text_topk_ids": text_ids, "text_topk_vals": text_vals,
             "audio_topk_ids": audio_ids, "audio_topk_vals": audio_vals},
            self.cache_dir / f"{sample_id}.pt",
        )

    def read(self, sample_id: str) -> tp.Optional[dict[str, torch.Tensor]]:
        path = self.cache_dir / f"{sample_id}.pt"
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu")

    @staticmethod
    def to_dense(ids: torch.Tensor, vals: torch.Tensor, card: int, fill_value: float = -1e4) -> torch.Tensor:
        """Reconstructs a dense (very negative outside top-k) logits tensor from a
        sparse top-k cache, so it can be dropped into `losses.kl_distillation_loss`
        as if it were the real teacher logits."""
        shape = ids.shape[:-1] + (card,)
        dense = torch.full(shape, fill_value, dtype=vals.dtype)
        dense.scatter_(-1, ids.long(), vals)
        return dense
