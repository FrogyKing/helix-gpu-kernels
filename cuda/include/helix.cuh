#pragma once

#include "helix.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <math_constants.h>

namespace helix {

inline void check_tensor(const torch::Tensor &x, const char *name) {
  TORCH_CHECK(!(at::GradMode::is_enabled() && x.requires_grad()),
              "Helix kernels are forward-only; use torch.inference_mode()");
  TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                  x.scalar_type() == at::kBFloat16,
              name, " must have dtype float32, float16, or bfloat16");
}

inline void check_same(const torch::Tensor &a, const torch::Tensor &b) {
  TORCH_CHECK(a.device() == b.device(), "tensors must share a CUDA device");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "tensors must share dtype");
}

// Unlike fmaxf alone, this operator preserves NaNs through the max reduction.
__device__ __forceinline__ float nan_max(float a, float b) {
  return (isnan(a) || isnan(b)) ? CUDART_NAN_F : fmaxf(a, b);
}

template <bool Maximum> __device__ __forceinline__ float block_reduce(float value) {
  __shared__ float warp_values[32];
  // A preceding reduction's broadcast must be consumed before reusing scratch.
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xffffffff, value, offset);
    value = Maximum ? nan_max(value, other) : value + other;
  }
  if (lane == 0)
    warp_values[warp] = value;
  __syncthreads();
  value = threadIdx.x < (blockDim.x >> 5) ? warp_values[lane] : (Maximum ? -CUDART_INF_F : 0.0f);
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other = __shfl_down_sync(0xffffffff, value, offset);
      value = Maximum ? nan_max(value, other) : value + other;
    }
    if (lane == 0)
      warp_values[0] = value;
  }
  __syncthreads();
  return warp_values[0];
}

} // namespace helix
