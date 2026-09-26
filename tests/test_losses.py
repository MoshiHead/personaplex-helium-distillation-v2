# SPDX-License-Identifier: MIT
"""Fast, self-contained shape/value sanity checks for distill/losses.py."""

import torch

from distill.losses import (
    kl_distillation_loss, ce_loss, bridge_loss, hidden_loss, HiddenProjection,
    dilate_mask, normalized_entropy, frame_weights, RunningNorm, _codebook_weights,
)


def test_kl_zero_when_distributions_match():
    logits = torch.randn(2, 3, 5, 10)
    mask = torch.ones(2, 3, 5, dtype=torch.bool)
    loss = kl_distillation_loss(logits, logits.clone(), mask, temperature=1.0)
    assert loss.item() < 1e-5


def test_kl_ignores_nan_padded_positions():
    student = torch.randn(1, 1, 4, 6)
    teacher = torch.randn(1, 1, 4, 6)
    mask = torch.ones(1, 1, 4, dtype=torch.bool)
    mask[:, :, -1] = False
    student[:, :, -1] = float("nan")
    teacher[:, :, -1] = float("nan")
    loss = kl_distillation_loss(student, teacher, mask, temperature=1.0)
    assert torch.isfinite(loss)


def test_ce_matches_manual_cross_entropy():
    logits = torch.randn(2, 1, 3, 5)
    targets = torch.randint(0, 5, (2, 1, 3))
    got = ce_loss(logits, targets, ignore_index=-100)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1))
    assert torch.allclose(got, expected, atol=1e-5)


def test_ce_ignore_index_excludes_position():
    logits = torch.zeros(1, 1, 2, 4)
    logits[0, 0, 0, 0] = 100.0  # position 0: confident correct prediction for class 0
    logits[0, 0, 1, 1] = -100.0  # position 1: confident WRONG prediction, but ignored
    targets = torch.tensor([[[0, 3]]])
    targets_ignored = targets.clone()
    targets_ignored[0, 0, 1] = -100
    loss_with_ignore = ce_loss(logits, targets_ignored, ignore_index=-100)
    assert loss_with_ignore.item() < 0.01


def test_bridge_loss_zero_when_identical():
    x = torch.randn(2, 5, 16)
    assert bridge_loss(x, x.clone()).item() < 1e-5


def test_hidden_loss_uses_projection_and_normalizes():
    proj = HiddenProjection(student_dim=8, teacher_dim=16)
    student_hidden = torch.randn(2, 4, 8)
    teacher_hidden = torch.randn(2, 4, 16)
    loss = hidden_loss(student_hidden, teacher_hidden, proj)
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)


def test_dilate_mask_expands_by_radius():
    mask = torch.zeros(1, 10)
    mask[0, 5] = 1.0
    dilated = dilate_mask(mask, radius=2)
    assert dilated[0, 3:8].bool().all()
    assert not dilated[0, 0].bool()
    assert not dilated[0, 9].bool()


def test_dilate_mask_noop_at_radius_zero():
    mask = torch.zeros(1, 10)
    mask[0, 5] = 1.0
    assert torch.equal(dilate_mask(mask, radius=0), mask)


def test_normalized_entropy_uniform_is_one():
    logits = torch.zeros(1, 5, 100)  # uniform distribution -> max entropy
    entropy = normalized_entropy(logits)
    assert torch.allclose(entropy, torch.ones_like(entropy), atol=1e-3)


def test_frame_weights_boosts_transitions():
    transition_mask = torch.zeros(1, 20)
    transition_mask[0, 10] = 1.0
    logits = torch.randn(1, 20, 50)
    weights = frame_weights(transition_mask, logits, radius=2, transition_boost=3.0, entropy_boost=0.0)
    assert weights[0, 10].item() > weights[0, 0].item()


def test_codebook_weights_tile_and_truncate():
    w = _codebook_weights(16, torch.device("cpu"), torch.float32)
    assert w.shape == (16,)
    assert torch.equal(w[:8], w[8:16])


def test_running_norm_converges_toward_stable_scale():
    norm = RunningNorm(momentum=0.5)
    val = torch.tensor(10.0)
    outputs = [norm(val).item() for _ in range(20)]
    assert abs(outputs[-1] - 1.0) < 0.05


def test_running_norm_survives_nan_input():
    """Regression test: a single non-finite call used to permanently corrupt
    `self._mean` to NaN (`momentum*mean + (1-momentum)*nan == nan` forever after),
    silently normalizing every FUTURE, otherwise-healthy call by NaN too. This
    happened in practice when a degenerate student init produced an inf bridge
    loss on step 0 of P1 -- every subsequent step's ce/bridge/hidden terms came
    out NaN as a result, even though only the FIRST step's data was bad.

    The bad call's own OUTPUT is still (correctly) non-finite -- dividing a NaN
    numerator by a finite mean is still NaN, and the caller (distill/train.py) is
    separately responsible for checking total_loss and skipping the optimizer
    step on it. What must NOT happen is `self._mean` itself becoming NaN, which
    would corrupt every subsequent, otherwise-healthy call too.
    """
    norm = RunningNorm(momentum=0.5)
    norm(torch.tensor(10.0))
    mean_before = norm._mean

    norm(torch.tensor(float("nan")))  # this call's own output is expected to be NaN
    assert norm._mean == mean_before, "a non-finite call must not update the running mean"

    norm(torch.tensor(float("inf")))
    assert norm._mean == mean_before

    # A healthy call afterward should behave exactly as if the bad calls never happened.
    recovered = norm(torch.tensor(10.0)).item()
    assert torch.isfinite(torch.tensor(recovered))
    assert 0.0 < recovered < 10.0


def test_running_norm_handles_nan_as_first_call():
    """Same guard, but for the case where the VERY FIRST call is non-finite (no
    prior good mean to fall back on) -- must not raise, and must leave `_mean`
    unset (None) rather than NaN, so the next healthy call initializes cleanly."""
    norm = RunningNorm(momentum=0.5)
    norm(torch.tensor(float("nan")))
    assert norm._mean is None

    recovered = norm(torch.tensor(10.0)).item()
    assert torch.isfinite(torch.tensor(recovered))
