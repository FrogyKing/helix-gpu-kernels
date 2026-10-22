"""Fused RMS normalization and elementwise learned scale."""

import math

import torch
import triton
import triton.language as tl

from ._validation import floating, last_dimension


@triton.jit
def _rmsnorm(X, W, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=0).to(tl.float32)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)
    tl.store(Y + row * N + col, x * inv * w, col < N)


def rmsnorm(x, weight, eps=1e-6):
    """Accumulate mean square in FP32, matching the documented FP32 reference."""
    floating(x, weight)
    last_dimension(x)
    if weight.ndim != 1 or weight.numel() != x.shape[-1]:
        raise ValueError("weight must be a vector matching the final dimension")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    y = torch.empty_like(x)
    if x.numel():
        block = triton.next_power_of_2(x.shape[-1])
        warps = 4 if block <= 2048 else 8 if block <= 8192 else 16
        with torch.cuda.device(x.device):
            _rmsnorm[(x.numel() // x.shape[-1],)](
                x, weight, y, x.shape[-1], eps, block, num_warps=warps
            )
    return y
