import math

import pytest
import torch

from helix import reference as ref


@pytest.mark.parametrize("width", [1, 3, 33, 257, 1025])
def test_stable_softmax(width):
    x = torch.linspace(-10000, 10000, width).repeat(3, 1)
    result = ref.softmax(x)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.sum(-1), torch.ones(3))
    torch.testing.assert_close(result, torch.softmax(x.double(), -1).float())


def test_softmax_nonfinite_contract():
    x = torch.tensor([[0.0, -math.inf], [-math.inf, -math.inf], [math.inf, 1.0]])
    result = ref.softmax(x)
    torch.testing.assert_close(result[0], torch.tensor([1.0, 0.0]))
    assert torch.isnan(result[1:]).all()


@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_rmsnorm_zero_and_scale(eps):
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    w = torch.tensor([2.0, -1.0])
    expected = (x.double() / torch.sqrt(x.double().square().mean(-1, True) + eps)) * w
    torch.testing.assert_close(ref.rmsnorm(x, w, eps), expected.float())


@pytest.mark.parametrize("eps", [0, -1, math.inf, math.nan])
def test_rmsnorm_bad_eps(eps):
    with pytest.raises(ValueError):
        ref.rmsnorm(torch.ones(2, 3), torch.ones(3), eps)


@pytest.mark.parametrize("causal", [False, True])
def test_attention_matches_sdpa(causal):
    torch.manual_seed(42)
    q, k, v = (torch.randn(2, 3, 17, 7) for _ in range(3))
    torch.testing.assert_close(ref.attention(q, k, v, causal), ref.sdpa(q, k, v, causal))


def test_causal_first_position_cannot_see_future():
    q, k, v = (torch.randn(1, 1, 5, 3) for _ in range(3))
    result = ref.attention(q, k, v, True)
    torch.testing.assert_close(result[..., 0, :], v[..., 0, :])


def test_quantization_error_bound_and_zero_row():
    torch.manual_seed(7)
    w = torch.randn(9, 33)
    w[0].zero_()
    qw, scales = ref.quantize_weight(w)
    assert qw.dtype == torch.int8
    assert scales[0] == 1
    assert not qw[0].any()
    assert ((qw.float() * scales[:, None] - w).abs() <= scales[:, None] / 2 + 1e-6).all()
    x = torch.randn(5, 33)
    actual = ref.quantized_linear(x, qw, scales)
    bound = x.abs().sum(-1, keepdim=True) * scales[None, :] / 2
    assert ((actual - x @ w.T).abs() <= bound + 1e-5).all()


def test_quantization_rejects_nonfinite():
    with pytest.raises(ValueError, match="finite"):
        ref.quantize_weight(torch.tensor([[float("nan")]]))


def test_quantization_rejects_underflowing_scale():
    with pytest.raises(ValueError, match="underflows"):
        ref.quantize_weight(torch.tensor([[torch.finfo(torch.float32).tiny * 1e-7]]))
