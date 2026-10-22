#include "helix.cuh"

namespace helix {
namespace {
constexpr int Tile = 16;

template <typename T>
__global__ void int8_linear_kernel(const T *__restrict__ x, const int8_t *__restrict__ weight,
                                   const float *__restrict__ scales, T *__restrict__ output,
                                   int64_t m, int64_t n, int64_t k) {
  const int64_t row = static_cast<int64_t>(blockIdx.y) * Tile + threadIdx.y;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * Tile + threadIdx.x;
  __shared__ float x_tile[Tile][Tile];
  // [output channel][input feature] gives contiguous global INT8 weight loads.
  __shared__ float weight_tile[Tile][Tile + 1];
  float accumulator = 0.0f;
  for (int64_t base = 0; base < k; base += Tile) {
    x_tile[threadIdx.y][threadIdx.x] = row < m && base + threadIdx.x < k
                                           ? static_cast<float>(x[row * k + base + threadIdx.x])
                                           : 0.0f;
    const int64_t channel = static_cast<int64_t>(blockIdx.x) * Tile + threadIdx.y;
    weight_tile[threadIdx.y][threadIdx.x] =
        channel < n && base + threadIdx.x < k
            ? static_cast<float>(weight[channel * k + base + threadIdx.x])
            : 0.0f;
    __syncthreads();
#pragma unroll
    for (int index = 0; index < Tile; ++index)
      accumulator = fmaf(x_tile[threadIdx.y][index], weight_tile[threadIdx.x][index], accumulator);
    __syncthreads();
  }
  // A per-output-channel scale is constant over the reduction, so apply it once.
  if (row < m && col < n)
    output[row * n + col] = static_cast<T>(accumulator * scales[col]);
}
} // namespace

torch::Tensor quantized_linear(torch::Tensor x, torch::Tensor weight, torch::Tensor scales) {
  check_tensor(x, "x");
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0, "x requires a positive final dimension");
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == at::kChar,
              "quantized weight must be contiguous CUDA INT8");
  TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.scalar_type() == at::kFloat,
              "scales must be contiguous CUDA FP32");
  TORCH_CHECK(x.device() == weight.device() && x.device() == scales.device(),
              "x, weight, and scales must share a device");
  TORCH_CHECK(weight.dim() == 2 && weight.size(1) == x.size(-1) && scales.dim() == 1 &&
                  scales.numel() == weight.size(0),
              "require x [...,K], weight [N,K], scales [N]");
  const int64_t k = x.size(-1), n = weight.size(0), m = x.numel() / k;
  TORCH_CHECK((m + Tile - 1) / Tile <= 65535 &&
                  (n + Tile - 1) / Tile <= std::numeric_limits<int>::max(),
              "linear dimensions exceed launch-grid bounds");
  const c10::cuda::CUDAGuard guard(x.device());
  auto shape = x.sizes().vec();
  shape.back() = n;
  auto output = torch::empty(shape, x.options());
  if (m == 0 || n == 0)
    return output;
  const dim3 block(Tile, Tile), grid((n + Tile - 1) / Tile, (m + Tile - 1) / Tile);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "helix_int8_linear", [&] {
        int8_linear_kernel<scalar_t><<<grid, block, 0, stream>>>(
            x.data_ptr<scalar_t>(), weight.data_ptr<int8_t>(), scales.data_ptr<float>(),
            output.data_ptr<scalar_t>(), m, n, k);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

} // namespace helix
