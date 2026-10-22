"""Host validation; no synchronization or data-dependent GPU checks."""

import torch

FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def floating(*tensors):
    for tensor in tensors:
        if not tensor.is_cuda or tensor.dtype not in FLOAT_DTYPES:
            raise ValueError("expected CUDA FP16, BF16, or FP32 tensors")
        if not tensor.is_contiguous():
            raise ValueError("contiguous tensors are required")
        if tensor.requires_grad and torch.is_grad_enabled():
            raise ValueError("forward only: use torch.inference_mode() or detach inputs")
        if (
            tensor.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(tensor.device)[0] < 8
        ):
            raise ValueError("BF16 requires compute capability 8.0 or newer")
    if any(t.device != tensors[0].device or t.dtype != tensors[0].dtype for t in tensors):
        raise ValueError("floating inputs must have the same dtype and CUDA device")


def last_dimension(x):
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("a nonempty final dimension is required")
    if x.shape[-1] > 65536:
        raise ValueError("single-program reduction supports widths up to 65536")
