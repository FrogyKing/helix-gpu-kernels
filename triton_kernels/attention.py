"""Online softmax attention with fixed tiles and no quadratic score allocation.

The implementation uses the FlashAttention online recurrence; it is not the
production FlashAttention library. P @ V uses FP32 IEEE dot products, deliberately
preserving FP32 probabilities instead of rounding them to a Tensor Core dtype.
"""

import math

import torch
import triton
import triton.language as tl

from ._validation import floating


@triton.jit
def _attention(
    Q,
    K,
    V,
    OUT,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1)
    ds = tl.arange(0, BD)
    js = tl.arange(0, BN)
    q = tl.load(
        Q + head * NQ * D + rows[:, None] * D + ds[None, :],
        (rows[:, None] < NQ) & (ds[None, :] < D),
        other=0,
    )
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.full((BM,), 0, tl.float32)
    numerator = tl.full((BM, BD), 0, tl.float32)
    end = NK
    if CAUSAL:
        end = tl.minimum(NK, (tl.program_id(0) + 1) * BM)
    for start in range(tl.cdiv(end, BN)):
        cols = start * BN + js
        k = tl.load(
            K + head * NK * D + ds[:, None] + cols[None, :] * D,
            (ds[:, None] < D) & (cols[None, :] < NK),
            other=0,
        )
        logits = tl.dot(q, k, input_precision="ieee") * SCALE
        valid = (rows[:, None] < NQ) & (cols[None, :] < NK)
        if CAUSAL:
            valid = valid & (cols[None, :] <= rows[:, None])
        logits = tl.where(valid, logits, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(logits, axis=1))
        # Padded query rows have no valid keys; keep their accumulator finite.
        safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
        correction = tl.exp(maximum - safe_maximum)
        probabilities = tl.exp(logits - safe_maximum[:, None])
        v = tl.load(
            V + head * NK * D + cols[:, None] * D + ds[None, :],
            (cols[:, None] < NK) & (ds[None, :] < D),
            other=0,
        ).to(tl.float32)
        numerator = numerator * correction[:, None]
        numerator = tl.dot(probabilities, v, numerator, input_precision="ieee")
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        maximum = new_maximum
    output = numerator / denominator[:, None]
    tl.store(
        OUT + head * NQ * D + rows[:, None] * D + ds[None, :],
        output,
        (rows[:, None] < NQ) & (ds[None, :] < D),
    )


def attention(q, k, v, causal=False):
    """Forward attention on [batch,heads,sequence,dim], with upper-left causal mask.

    Q sequence length may differ from K/V. Head dimensions 1..128 are padded to
    a minimum of 16 for dot instructions. Finite inputs are the supported domain.
    Dropout, arbitrary masks, GQA, autograd and KV-cache decoding are out of scope.
    """
    floating(q, k, v)
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected Q, K, V shaped [batch,heads,sequence,dim]")
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("batch, heads, and dimensions must match; K and V must match")
    if not 1 <= q.shape[-1] <= 128 or k.shape[-2] == 0:
        raise ValueError("head dimension must be 1..128 and K/V sequence nonempty")
    if not isinstance(causal, bool):
        raise ValueError("causal must be boolean")
    output = torch.empty_like(q)
    if q.numel():
        nq, d = q.shape[-2:]
        nk = k.shape[-2]
        bd = max(16, triton.next_power_of_2(d))
        with torch.cuda.device(q.device):
            _attention[(triton.cdiv(nq, 16), q.shape[0] * q.shape[1])](
                q, k, v, output, nq, nk, d, 1 / math.sqrt(d), causal, 16, 32, bd, num_warps=4
            )
    return output
