"""Fixed-tile GEMM with FP32 accumulation and no hidden TF32 truncation."""

import torch
import triton
import triton.language as tl

from ._validation import floating


@triton.jit
def _gemm(
    A,
    B,
    C,
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
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        idx = start * BK + ks
        a = tl.load(
            A + rows[:, None] * K + idx[None, :], (rows[:, None] < M) & (idx[None, :] < K), other=0
        )
        b = tl.load(
            B + idx[:, None] * N + cols[None, :], (idx[:, None] < K) & (cols[None, :] < N), other=0
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(C + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))


def gemm(a, b):
    """Multiply contiguous [M,K] and [K,N]; cast FP32 sum to the input dtype."""
    floating(a, b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected [M,K] and [K,N] matrices")
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if m and n:
        with torch.cuda.device(a.device):
            _gemm[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
                a, b, c, m, n, k, 32, 32, 32, num_warps=4
            )
    return c
