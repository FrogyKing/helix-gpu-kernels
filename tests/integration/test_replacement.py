import sys
import types

import pytest
import torch

from helix import reference
from helix.integration.smollm import replace_rmsnorm


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(7))
        self.variance_epsilon = 1e-5

    def forward(self, x):
        return reference.rmsnorm(x, self.weight, self.variance_epsilon)


def test_replacement_preserves_weights_and_epsilon(monkeypatch):
    # CPU test checks adapter plumbing, not custom GPU execution.
    module = types.ModuleType("triton_kernels.rmsnorm")
    module.rmsnorm = reference.rmsnorm
    monkeypatch.setitem(sys.modules, "triton_kernels.rmsnorm", module)
    model = torch.nn.Sequential(LlamaRMSNorm(), LlamaRMSNorm())
    original_weight = model[0].weight
    x = torch.randn(2, 3, 7)
    expected = model(x)
    originals = replace_rmsnorm(model, "triton")
    assert len(originals) == 2
    assert model[0].weight is original_weight
    assert model[0].variance_epsilon == 1e-5
    torch.testing.assert_close(model(x), expected)


def test_unsupported_model_rejected():
    with pytest.raises(ValueError, match="no supported"):
        replace_rmsnorm(torch.nn.Linear(3, 4), "triton")
