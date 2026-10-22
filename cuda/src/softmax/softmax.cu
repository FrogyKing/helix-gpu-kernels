#include "helix.cuh"

namespace helix {

template <typename T>
__global__ void softmax_rows(const T *__restrict__ x, T *__restrict__ y, int64_t width) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * width;
  float maximum = -CUDART_INF_F;
  for (int64_t j = threadIdx.x; j < width; j += blockDim.x)
    maximum = nan_max(maximum, static_cast<float>(x[base + j]));
  maximum = block_reduce<true>(maximum);
  float denominator = 0.0f;
  for (int64_t j = threadIdx.x; j < width; j += blockDim.x)
    denominator += expf(static_cast<float>(x[base + j]) - maximum);
  denominator = block_reduce<false>(denominator);
  for (int64_t j = threadIdx.x; j < width; j += blockDim.x)
    y[base + j] = static_cast<T>(expf(static_cast<float>(x[base + j]) - maximum) / denominator);
}

torch::Tensor softmax(torch::Tensor x) {
  check_tensor(x, "x");
  TORCH_CHECK(x.dim() >= 1, "softmax requires at least one dimension");
  const c10::cuda::CUDAGuard guard(x.device());
  auto output = torch::empty_like(x);
  if (x.numel() == 0)
    return output;
  const int64_t width = x.size(-1);
  const int64_t rows = x.numel() / width;
  TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "too many softmax rows");
  const int threads = width <= 128 ? 128 : 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "helix_softmax", [&] {
    softmax_rows<scalar_t>
        <<<rows, threads, 0, stream>>>(x.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), width);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

} // namespace helix
