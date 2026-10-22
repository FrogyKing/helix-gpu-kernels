#include "helix.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax", &helix::softmax, "Stable last-dimension softmax (CUDA)");
  m.def("rmsnorm", &helix::rmsnorm, "Fused RMSNorm (CUDA)");
  m.def("gemm", &helix::gemm, "Naive or shared-memory tiled GEMM (CUDA)");
  m.def("attention", &helix::attention, "Tiled online-softmax attention (CUDA)");
  m.def("quantized_linear", &helix::quantized_linear, "INT8 weight-only linear (CUDA)");
}
