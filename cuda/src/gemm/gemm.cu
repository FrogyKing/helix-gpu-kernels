#include "helix.cuh"

namespace helix {
namespace {
constexpr int Tile = 16;

template <typename T, bool Tiled>
__global__ void gemm_kernel(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ c,
                            int64_t m, int64_t n, int64_t k) {
  const int64_t row = static_cast<int64_t>(blockIdx.y) * Tile + threadIdx.y;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * Tile + threadIdx.x;
  float sum = 0.0f;
  if constexpr (Tiled) {
    __shared__ T a_tile[Tile][Tile];
    __shared__ T b_tile[Tile][Tile];
    for (int64_t base = 0; base < k; base += Tile) {
      a_tile[threadIdx.y][threadIdx.x] = row < m && base + threadIdx.x < k
                                             ? a[row * k + base + threadIdx.x]
                                             : static_cast<T>(0.0f);
      b_tile[threadIdx.y][threadIdx.x] = base + threadIdx.y < k && col < n
                                             ? b[(base + threadIdx.y) * n + col]
                                             : static_cast<T>(0.0f);
      __syncthreads();
#pragma unroll
      for (int index = 0; index < Tile; ++index)
        sum = fmaf(static_cast<float>(a_tile[threadIdx.y][index]),
                   static_cast<float>(b_tile[index][threadIdx.x]), sum);
      __syncthreads();
    }
  } else {
    if (row < m && col < n)
      for (int64_t index = 0; index < k; ++index)
        sum = fmaf(static_cast<float>(a[row * k + index]), static_cast<float>(b[index * n + col]),
                   sum);
  }
  if (row < m && col < n)
    c[row * n + col] = static_cast<T>(sum);
}
} // namespace

torch::Tensor gemm(torch::Tensor a, torch::Tensor b, bool tiled) {
  check_tensor(a, "a");
  check_tensor(b, "b");
  check_same(a, b);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
              "GEMM requires [M,K] and [K,N]");
  const int64_t m = a.size(0), n = b.size(1), k = a.size(1);
  TORCH_CHECK((m + Tile - 1) / Tile <= 65535 &&
                  (n + Tile - 1) / Tile <= std::numeric_limits<int>::max(),
              "GEMM dimensions exceed launch-grid bounds");
  const c10::cuda::CUDAGuard guard(a.device());
  auto output = torch::empty({m, n}, a.options());
  if (m == 0 || n == 0)
    return output;
  const dim3 block(Tile, Tile), grid((n + Tile - 1) / Tile, (m + Tile - 1) / Tile);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, a.scalar_type(), "helix_gemm", [&] {
    if (tiled)
      gemm_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
          a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), m, n, k);
    else
      gemm_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
          a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), m, n, k);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

} // namespace helix
