#pragma once

#include <torch/extension.h>

namespace helix {

torch::Tensor softmax(torch::Tensor x);
torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps);
torch::Tensor gemm(torch::Tensor a, torch::Tensor b, bool tiled);
torch::Tensor attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);
torch::Tensor quantized_linear(torch::Tensor x, torch::Tensor weight, torch::Tensor scales);

} // namespace helix
