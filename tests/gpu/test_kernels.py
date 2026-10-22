"""Explicit GPU correctness suite. A skip is not GPU validation evidence."""

import importlib

import pytest
import torch

from helix import reference as ref

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def implementation(backend, operation):
    module = "quantization" if operation == "quantized_linear" else operation
    path = "helix.runtime.cuda" if backend == "cuda" else f"triton_kernels.{module}"
    return getattr(importlib.import_module(path), operation)


def tolerance(dtype, reduction=False):
    # FP16/BF16 output rounding dominates; FP32 GEMM reduction order also differs.
    if dtype == torch.bfloat16:
        return {"rtol": 2e-2, "atol": 2e-2}
    if dtype == torch.float16:
        return {"rtol": 3e-3, "atol": 3e-3}
    return {"rtol": 3e-4 if reduction else 2e-5, "atol": 3e-4 if reduction else 2e-6}


@pytest.fixture(autouse=True)
def precision():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1729)
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1), (7, 33), (13, 257), (2, 1025), (2, 8193)])
def test_softmax(backend, dtype, shape):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    x[0] *= 20
    torch.testing.assert_close(
        implementation(backend, "softmax")(x), ref.softmax(x), **tolerance(dtype)
    )


@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [1, 33, 1025, 8193])
def test_rmsnorm(backend, dtype, width):
    x = torch.randn(5, width, device="cuda", dtype=dtype)
    x[0].zero_()
    w = torch.randn(width, device="cuda", dtype=dtype)
    torch.testing.assert_close(
        implementation(backend, "rmsnorm")(x, w), ref.rmsnorm(x, w), **tolerance(dtype)
    )


@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(1, 1, 1), (17, 31, 19), (64, 128, 65)])
def test_gemm(backend, dtype, dims):
    m, n, k = dims
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    expected = (a.double() @ b.double()).to(dtype)
    fn = implementation(backend, "gemm")
    torch.testing.assert_close(fn(a, b), expected, **tolerance(dtype, True))
    if backend == "cuda":
        torch.testing.assert_close(fn(a, b, variant="naive"), expected, **tolerance(dtype, True))


@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1, 1, 16), (1, 2, 33, 32), (2, 1, 65, 64), (1, 1, 129, 128)])
@pytest.mark.parametrize("causal", [False, True])
def test_attention(backend, dtype, shape, causal):
    q, k, v = (torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3))
    actual = implementation(backend, "attention")(q, k, v, causal)
    torch.testing.assert_close(actual, ref.attention(q, k, v, causal), **tolerance(dtype, True))


@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(1, 1, 1), (7, 19, 33), (64, 128, 65)])
def test_quantized_linear(backend, dtype, dims):
    m, n, k = dims
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    w = torch.randn(n, k, device="cuda", dtype=dtype)
    w[0].zero_()
    qw, scales = ref.quantize_weight(w)
    torch.testing.assert_close(
        implementation(backend, "quantized_linear")(x, qw, scales),
        ref.quantized_linear(x, qw, scales),
        **tolerance(dtype, True),
    )


@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_softmax_masked_nonfinite(backend):
    x = torch.tensor(
        [[0.0, -float("inf"), 1.0], [-float("inf")] * 3, [float("inf"), 1.0, 2.0]],
        device="cuda",
    )
    torch.testing.assert_close(
        implementation(backend, "softmax")(x), ref.softmax(x), equal_nan=True
    )


@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_forward_only_and_noncontiguous_rejected(backend):
    fn = implementation(backend, "softmax")
    with pytest.raises((RuntimeError, ValueError), match="contiguous"):
        fn(torch.randn(3, 7, device="cuda").T)
    with pytest.raises((RuntimeError, ValueError), match="grad|forward|autograd"):
        fn(torch.randn(3, 7, device="cuda", requires_grad=True))


@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_current_cuda_stream(backend):
    fn = implementation(backend, "softmax")
    # Compile before testing stream ordering, so JIT initialization cannot mask bugs.
    fn(torch.ones(3, 7, device="cuda"))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(33, 257, device="cuda")
        expected = ref.softmax(x)
        actual = fn(x)
    stream.synchronize()
    torch.testing.assert_close(actual, expected)
