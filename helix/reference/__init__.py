"""PyTorch forward references; low precision reductions accumulate in float32."""

import math

import torch
import torch.nn.functional as F


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Native stable softmax; all-minus-infinity rows follow PyTorch (NaN)."""
    return torch.softmax(x, dim=-1)


def rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, round_normalized: bool = False
) -> torch.Tensor:
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    if x.ndim < 1 or x.shape[-1] == 0 or weight.shape != (x.shape[-1],):
        raise ValueError("weight must match a nonempty last dimension")
    xf = x.float()
    normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    if round_normalized:
        normalized = normalized.to(x.dtype).float()
    return (normalized * weight.float()).to(x.dtype)


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """Materialized mathematical reference, square self-attention [B,H,N,D]."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have identical [B,H,N,D] shapes")
    if q.shape[-1] == 0 or q.shape[-2] == 0:
        raise ValueError("sequence and head dimensions must be nonempty")
    scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        mask = torch.ones((q.shape[-2], q.shape[-2]), device=q.device, dtype=torch.bool)
        scores = scores.masked_fill(mask.triu(1), float("-inf"))
    return (torch.softmax(scores, dim=-1) @ v.float()).to(q.dtype)


def sdpa(q, k, v, causal=False):
    """Optimized framework reference; selected backend is hardware dependent."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric nearest-even INT8 per output channel; no zero point."""
    if weight.ndim != 2 or min(weight.shape) == 0 or not weight.is_floating_point():
        raise ValueError("weight must be a nonempty floating [out,in] matrix")
    if not torch.isfinite(weight).all():
        raise ValueError("quantization requires finite weights")
    wf = weight.float()
    if not torch.isfinite(wf).all():
        raise ValueError("weights must be representable in float32")
    maxima = wf.abs().amax(dim=1)
    scales = torch.where(maxima == 0, torch.ones_like(maxima), maxima / 127)
    if (scales == 0).any():
        raise ValueError("nonzero weight channel scale underflows float32")
    qweight = torch.round(wf / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return qweight.contiguous(), scales.contiguous()


def quantized_linear(x, qweight, scales):
    """Dequantize then matmul baseline; measures storage compression separately."""
    weight = qweight.float() * scales.float()[:, None]
    return (x.float() @ weight.T).to(x.dtype)
