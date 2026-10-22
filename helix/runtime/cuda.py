"""Lazy, forward-only CUDA extension loading.

Importing this module is safe on a CPU development machine. Building requires a
CUDA toolkit, a compatible PyTorch wheel, and a C++17 compiler. Compilation is
performed only on the first custom CUDA operation, normally inside Modal.
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Any


@functools.lru_cache(maxsize=1)
def load_extension() -> Any:
    """Build/load kernels from PyTorch's versioned extension cache.

    Set HELIX_CUDA_VERBOSE=1 to expose compiler diagnostics and MAX_JOBS to
    control compilation parallelism. No fast-math flag is used: precision changes
    must be explicit experiments, not hidden differences against the reference.
    """
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if not torch.cuda.is_available():
        raise RuntimeError("Helix CUDA requires a CUDA-enabled PyTorch runtime and GPU")
    if CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit not found; compile inside the Modal devel image")
    root = Path(__file__).resolve().parents[2]
    if not (root / "cuda" / "include").is_dir():
        root = Path(sys.prefix) / "share" / "helix"
    sources = [root / "cuda" / "bindings" / "bindings.cpp"]
    sources.extend(sorted((root / "cuda" / "src").rglob("*.cu")))
    if len(sources) != 6 or not all(source.is_file() for source in sources):
        raise RuntimeError(f"Expected Helix CUDA source tree at {root / 'cuda'}")
    os.environ.setdefault("MAX_JOBS", "2")
    return load(
        name="helix_cuda_v1",
        sources=[str(source) for source in sources],
        extra_include_paths=[str(root / "cuda" / "include")],
        extra_cflags=["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "--ptxas-options=-v"],
        verbose=os.environ.get("HELIX_CUDA_VERBOSE") == "1",
    )


def softmax(x: Any) -> Any:
    """Stable softmax over the final dimension; FP32 reduction for all dtypes."""
    return load_extension().softmax(x)


def rmsnorm(x: Any, weight: Any, eps: float = 1e-6) -> Any:
    """RMSNorm with a matching-dtype weight vector and FP32 accumulation."""
    return load_extension().rmsnorm(x, weight, eps)


def gemm(a: Any, b: Any, variant: str = "tiled") -> Any:
    """2D GEMM using SIMT FMA; neither implementation uses Tensor Cores."""
    if variant not in {"naive", "tiled"}:
        raise ValueError("variant must be 'naive' or 'tiled'")
    return load_extension().gemm(a, b, variant == "tiled")


def attention(q: Any, k: Any, v: Any, causal: bool = False) -> Any:
    """Online-softmax attention for [B,H,M,D], D<=256; rectangular causal is top-left.

    K and V have shape [B,H,N,D]. No attention score matrix is materialized.
    No dropout, GQA, arbitrary mask, KV cache, or backward pass is provided.
    """
    return load_extension().attention(q, k, v, causal)


def quantized_linear(x: Any, qweight: Any, scales: Any) -> Any:
    """Weight-only INT8 linear with qweight[N,K], FP32 scales[N], x[...,K].

    Inputs remain floating point. This is a storage-compression experiment using
    SIMT floating-point FMA, not an INT8 Tensor Core arithmetic implementation.
    """
    return load_extension().quantized_linear(x, qweight, scales)
