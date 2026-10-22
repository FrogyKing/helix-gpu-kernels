"""Weight-only INT8 linear: dequantize tiles in registers, accumulate FP32."""

import torch
import triton
import triton.language as tl

from ._validation import floating


@triton.jit
def _linear(
    X,
    W,
    S,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    scale = tl.load(S + cols, cols < N, other=0).to(tl.float32)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        idx = start * BK + ks
        x = tl.load(
            X + rows[:, None] * K + idx[None, :], (rows[:, None] < M) & (idx[None, :] < K), other=0
        ).to(tl.float32)
        w = tl.load(
            W + cols[None, :] * K + idx[:, None], (cols[None, :] < N) & (idx[:, None] < K), other=0
        ).to(tl.float32)
        w = w * scale[None, :]
        acc = tl.dot(x, w, acc, input_precision="ieee")
    tl.store(Y + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))


def quantized_linear(x, qweight, scales):
    """Compute x @ dequantized_weight.T without a full dequantized allocation.

    Weights are signed INT8 [out,in], scales FP32 [out]. This is weight storage
    compression and fused FP32 dequantization, not integer Tensor Core GEMM.
    """
    floating(x)
    if x.ndim < 1 or qweight.ndim != 2 or x.shape[-1] != qweight.shape[1]:
        raise ValueError("expected x[...,in] and qweight[out,in]")
    if qweight.dtype != torch.int8 or qweight.device != x.device or not qweight.is_contiguous():
        raise ValueError("qweight must be contiguous INT8 on the input CUDA device")
    if (
        scales.dtype != torch.float32
        or scales.device != x.device
        or not scales.is_contiguous()
        or scales.shape != (qweight.shape[0],)
    ):
        raise ValueError("scales must be contiguous FP32 [out] on the input CUDA device")
    if scales.requires_grad and torch.is_grad_enabled():
        raise ValueError("forward only: detach scales or use torch.inference_mode()")
    k = x.shape[-1]
    if k == 0:
        raise ValueError("input feature dimension must be positive")
    m = x.numel() // k
    n = qweight.shape[0]
    y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    if m and n:
        with torch.cuda.device(x.device):
            _linear[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
                x, qweight, scales, y, m, n, k, 32, 32, 32, num_warps=4
            )
    return y
