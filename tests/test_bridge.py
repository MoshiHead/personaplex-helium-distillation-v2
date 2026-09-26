# SPDX-License-Identifier: MIT
"""Regression test for a real bug: `Bridge.__init__` passed `device=device` to
`self.linear` but not to `self.norm` (via `create_norm_fn`), so `self.norm.alpha`
silently stayed on the default device while everything else in the model moved
to the requested one. Invisible in eager execution (RMSNorm's `alpha.to(var)`
just does a slower cross-device cast); a hard CUDA failure ("operation not
permitted when stream is capturing") the moment it runs inside LMGen's
CUDAGraphed capture, since a cross-device transfer cannot be captured into a
CUDA graph. Uses `device="meta"` (available on any machine, no GPU needed) to
exercise the same class of bug: a real CUDA machine would use "cuda", but the
underlying question -- does `device` reach every submodule? -- is identical.
"""

import torch

from distill.bridge import Bridge


def test_bridge_norm_matches_linear_device():
    bridge = Bridge(in_dim=8, out_dim=16, device="meta", dtype=torch.float32)
    assert bridge.linear.weight.device.type == "meta"
    assert bridge.norm.alpha.device.type == "meta", (
        "Bridge.norm.alpha did not follow the requested device -- "
        "create_norm_fn was called without device=, reintroducing the cross-device bug."
    )


def test_bridge_default_device_is_cpu():
    bridge = Bridge(in_dim=8, out_dim=16)
    assert bridge.linear.weight.device.type == "cpu"
    assert bridge.norm.alpha.device.type == "cpu"


def test_bridge_forward_shape():
    bridge = Bridge(in_dim=8, out_dim=16, dtype=torch.float32)
    x = torch.randn(2, 3, 8)
    out = bridge(x)
    assert out.shape == (2, 3, 16)
