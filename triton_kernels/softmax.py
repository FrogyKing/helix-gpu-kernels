"""One program per row: register reduction with FP32 exponentiation."""

import torch
import triton
import triton.language as tl

from ._validation import floating, last_dimension


@triton.jit
def _softmax(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=-float("inf")).to(tl.float32)
    e = tl.exp(x - tl.max(x, axis=0))
    y = e / tl.sum(e, axis=0)
    tl.store(Y + row * N + col, y, col < N)


def softmax(x):
    """Stable final-axis softmax; NaN and infinity behavior follows torch.softmax."""
    floating(x)
    last_dimension(x)
    y = torch.empty_like(x)
    if x.numel():
        block = triton.next_power_of_2(x.shape[-1])
        warps = 4 if block <= 2048 else 8 if block <= 8192 else 16
        with torch.cuda.device(x.device):
            _softmax[(x.numel() // x.shape[-1],)](x, y, x.shape[-1], block, num_warps=warps)
    return y
