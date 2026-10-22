#include "helix.cuh"

namespace helix {

template <typename T>
__global__ void rmsnorm_rows(const T *__restrict__ x, const T *__restrict__ weight,
                             T *__restrict__ output, int64_t width, float eps) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * width;
  float squares = 0.0f;
  for (int64_t j = threadIdx.x; j < width; j += blockDim.x) {
    const float value = static_cast<float>(x[base + j]);
    squares += value * value;
  }
  const float inverse = rsqrtf(block_reduce<false>(squares) / width + eps);
  for (int64_t j = threadIdx.x; j < width; j += blockDim.x)
    output[base + j] =
        static_cast<T>(static_cast<float>(x[base + j]) * inverse * static_cast<float>(weight[j]));
}

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps) {
  check_tensor(x, "x");
  check_tensor(weight, "weight");
  check_same(x, weight);
  TORCH_CHECK(x.dim() >= 1 && weight.dim() == 1 && x.size(-1) == weight.numel(),
              "weight must be a vector matching x's final dimension");
  TORCH_CHECK(x.size(-1) > 0, "RMSNorm width must be positive");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0 && static_cast<float>(eps) > 0.0f &&
                  std::isfinite(static_cast<float>(eps)),
              "eps must be finite, positive, and representable in FP32");
  const c10::cuda::CUDAGuard guard(x.device());
  auto output = torch::empty_like(x);
  if (x.numel() == 0)
    return output;
  const int64_t width = x.size(-1);
  const int64_t rows = x.numel() / width;
  TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "too many RMSNorm rows");
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "helix_rmsnorm", [&] {
    rmsnorm_rows<scalar_t>
        <<<rows, 256, 0, stream>>>(x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                                   output.data_ptr<scalar_t>(), width, static_cast<float>(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

} // namespace helix
