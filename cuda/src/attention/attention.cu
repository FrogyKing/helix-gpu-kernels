#include "helix.cuh"

namespace helix {
namespace {
constexpr int KeysPerTile = 16;
constexpr int Threads = 128;

// One CTA owns one query. K/V tiles are staged in shared memory and discarded.
// This is a transparent SIMT research kernel, not a production FlashAttention clone.
template <typename T>
__global__ void online_attention(const T *__restrict__ q, const T *__restrict__ k,
                                 const T *__restrict__ v, T *__restrict__ output,
                                 int64_t query_length, int64_t key_length, int dim, bool causal) {
  extern __shared__ float scratch[];
  float *query = scratch;
  float *keys = query + dim;
  float *values = keys + KeysPerTile * dim;
  float *scores = values + KeysPerTile * dim;
  float *probabilities = scores + KeysPerTile;
  float *state = probabilities + KeysPerTile; // running maximum, sum, old scaling
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t query_index = blockIdx.x % query_length;
  const int64_t batch_head = blockIdx.x / query_length;
  const int64_t q_base = static_cast<int64_t>(blockIdx.x) * dim;
  const int64_t kv_base = batch_head * key_length * dim;
  const int64_t limit = causal && query_index + 1 < key_length ? query_index + 1 : key_length;
  const float factor = rsqrtf(static_cast<float>(dim));
  float accumulator[2] = {0.0f, 0.0f};
  for (int feature = tid; feature < dim; feature += Threads)
    query[feature] = static_cast<float>(q[q_base + feature]);
  if (tid == 0) {
    state[0] = -CUDART_INF_F;
    state[1] = 0.0f;
  }
  __syncthreads();
  for (int64_t start = 0; start < limit; start += KeysPerTile) {
    for (int index = tid; index < KeysPerTile * dim; index += Threads) {
      const int64_t key_index = start + index / dim;
      const int feature = index % dim;
      keys[index] =
          key_index < limit ? static_cast<float>(k[kv_base + key_index * dim + feature]) : 0.0f;
      values[index] =
          key_index < limit ? static_cast<float>(v[kv_base + key_index * dim + feature]) : 0.0f;
    }
    __syncthreads();
    // Four warps compute sixteen independent Q.K dot products per tile.
    for (int key = warp; key < KeysPerTile; key += Threads / 32) {
      float score = 0.0f;
      for (int feature = lane; feature < dim; feature += 32)
        score = fmaf(query[feature], keys[key * dim + feature], score);
      for (int offset = 16; offset > 0; offset >>= 1)
        score += __shfl_down_sync(0xffffffff, score, offset);
      if (lane == 0)
        scores[key] = start + key < limit ? score * factor : -CUDART_INF_F;
    }
    __syncthreads();
    if (tid == 0) {
      float next_max = state[0];
      for (int key = 0; key < KeysPerTile; ++key)
        next_max = nan_max(next_max, scores[key]);
      const float old_scale = expf(state[0] - next_max);
      float tile_sum = 0.0f;
      for (int key = 0; key < KeysPerTile; ++key) {
        probabilities[key] = start + key < limit ? expf(scores[key] - next_max) : 0.0f;
        tile_sum += probabilities[key];
      }
      state[1] = state[1] * old_scale + tile_sum;
      state[0] = next_max;
      state[2] = old_scale;
    }
    __syncthreads();
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      const int feature = tid + slot * Threads;
      if (feature < dim) {
        float updated = accumulator[slot] * state[2];
#pragma unroll
        for (int key = 0; key < KeysPerTile; ++key)
          updated = fmaf(probabilities[key], values[key * dim + feature], updated);
        accumulator[slot] = updated;
      }
    }
    __syncthreads(); // no thread may overwrite a tile still consumed by a peer
  }
#pragma unroll
  for (int slot = 0; slot < 2; ++slot) {
    const int feature = tid + slot * Threads;
    if (feature < dim)
      output[q_base + feature] = static_cast<T>(accumulator[slot] / state[1]);
  }
}
} // namespace

torch::Tensor attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  check_tensor(q, "q");
  check_tensor(k, "k");
  check_tensor(v, "v");
  check_same(q, k);
  check_same(q, v);
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "attention requires contiguous [B,H,S,D] tensors");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1) && k.sizes() == v.sizes() &&
                  q.size(3) == k.size(3),
              "attention batch/head/feature dimensions must match; K and V shapes must match");
  TORCH_CHECK(q.size(3) > 0 && q.size(3) <= 256, "attention head dimension must be 1..256");
  TORCH_CHECK(k.size(2) > 0, "attention requires at least one key");
  const int64_t rows = q.numel() / q.size(3);
  TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "too many attention query rows");
  const c10::cuda::CUDAGuard guard(q.device());
  auto output = torch::empty_like(q);
  if (q.numel() == 0)
    return output;
  const int dim = static_cast<int>(q.size(3));
  const size_t shared_bytes = (dim + 2 * KeysPerTile * dim + 2 * KeysPerTile + 3) * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, q.scalar_type(), "helix_online_attention", [&] {
        online_attention<scalar_t><<<rows, Threads, shared_bytes, stream>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), q.size(2), k.size(2), dim, causal);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

} // namespace helix
