# Mathematical and numerical foundations

These are analytical arguments, not measured performance results. The GPU suite
must be executed on hardware before these implementations are considered validated.

## Floating point and reductions

For a normal floating-point operation, the usual model is
`fl(a op b) = (a op b)(1 + delta)`, `|delta| <= u`, excluding overflow and
underflow. Unit roundoff is approximately 2^-24 for FP32, 2^-11 for FP16 and
2^-8 for BF16. A sequential sum of n terms has an absolute error bounded by
`gamma_(n-1) * sum(abs(x))`, where `gamma_n = n*u/(1-n*u)` when `n*u < 1`.
A balanced reduction reduces the dependency depth to about log2(n), but changes
the summation order. Different reduction trees need not agree bit-for-bit.

Helix accumulates normalizations, dot products, and online attention state in
FP32. Output conversion to FP16/BF16 still rounds the final value. FP32
accumulation does not prevent overflow of `x*x` for arbitrarily large FP32
RMSNorm inputs. These kernels are finite-input inference implementations;
nonfinite softmax behavior is tested separately against PyTorch. No backward
pass or gradient guarantee is provided.

## Stable softmax

For finite logits, let `m = max_i x_i`. Dividing numerator and denominator by
`exp(m)` gives `p_i = exp(x_i-m) / sum_j exp(x_j-m)`. Every exponential is in
`(0,1]` in exact arithmetic, the denominator is at least one, and the formula
avoids positive exponential overflow. Extremely negative differences can
underflow to zero; this loses tiny probabilities rather than overflowing large
ones. A row of all `-inf` or a row containing `+inf` produces NaNs under the
PyTorch subtraction convention. A finite row mixed with `-inf` is well defined.
Padding lanes must contribute `-inf` to the maximum and zero to the sum.

One fused row kernel reads input and writes output once in an ideal traffic
model: `2*rows*width*sizeof(dtype)` bytes. This is a lower bound, not measured
DRAM traffic. Wide rows may require extra loads or reduce occupancy through
register pressure. Compare against native `torch.softmax`, not an intentionally
unfused chain of Python operations.

## Exact blockwise online attention

For one query, write logits `s_j = q dot k_j / sqrt(d)`. After visiting a set A,
maintain `m_A = max_(j in A) s_j`,
`l_A = sum_(j in A) exp(s_j-m_A)`, and
`o_A = sum_(j in A) exp(s_j-m_A)*v_j`. The unnormalized weighted sum is `o_A`.
For a new disjoint block B, let `m = max(m_A, max_(j in B) s_j)` and
`alpha = exp(m_A-m)`. Then

```text
l_new = alpha*l_A + sum_(j in B) exp(s_j-m)
o_new = alpha*o_A + sum_(j in B) exp(s_j-m)*v_j
m_new = m
output = o_final / l_final
```

The identity `exp(s_j-m) = exp(s_j-m_A)*exp(m_A-m)` rescales every previously
visited contribution by the same factor. Induction over blocks proves the state
equals a stable softmax over all visited keys. The final division therefore
equals materialized softmax attention in exact arithmetic. Floating-point
rescaling, reordered sums and dot-product precision introduce differences.

Causal masking sets future logits to `-inf`. For square self-attention every
query has at least its own valid key. A first block with no valid keys requires
special handling to avoid `exp(-inf - -inf)`; padded query rows must not be stored.
Fully masked user-defined rows are outside the current attention API. Rectangular
custom attention uses upper-left causal alignment; the inference integration
replaces RMSNorm only and retains PyTorch's KV-cache attention semantics.

Materializing scores requires `O(B*H*N^2)` extra storage. Tile processing avoids
that matrix, with state proportional to the query tile times head dimension.
Both algorithms still perform `O(B*H*N^2*d)` arithmetic. Calling this approach
FlashAttention-like describes an algorithmic principle, not parity with the
production library's scheduling, backward kernels or architecture specialization.

## RMSNorm

For width d, `y_i = w_i*x_i / sqrt(sum_j x_j^2/d + eps)`. Positive epsilon keeps
the all-zero row finite. The implementation converts input to FP32 before
squaring and reducing. The ideal fused traffic model is input read + output
write + weight read (`3*rows*d*bytes` without cache reuse). A cached shared
weight vector can reduce physical traffic, so this estimate must not be reported
as hardware-counter bandwidth. Production reference baselines use the same
weight and epsilon semantics.

## GEMM, arithmetic intensity and roofline

For `A[M,K] @ B[K,N]`, count approximately `2*M*N*K` FLOPs (multiply and add
count separately). Achieved TFLOP/s is `2*M*N*K/(seconds*10^12)`. Small K adds
boundary effects and the conventional count omits loop/index and conversion work.
The minimum read/write traffic is
`bytes_A*M*K + bytes_B*K*N + bytes_C*M*N`, assuming each operand is read once.
Ideal arithmetic intensity is FLOPs divided by that lower-bound traffic.

A shared-memory tile with dimensions BM, BN, BK performs `2*BM*BN*BK` FLOPs
per `BM*BK + BK*BN` elements loaded, amortizing global loads across outputs.
Synchronization, bank behavior, register capacity, occupancy and tail tiles can
limit realized reuse. The roofline bound is
`P <= min(peak_compute, sustained_bandwidth*arithmetic_intensity)`.
Use hardware and precision-specific peak limits, and measured sustained
bandwidth if available. Vendor spec peaks are ceilings, not benchmark outputs.
The current CUDA SIMT tiled GEMM is educational and does not use Tensor Cores;
the Triton dot path may use them for low precision. FP32 IEEE and TF32 are distinct
comparisons. Tensor Core eligibility must not be inferred from TFLOP/s alone.

## INT8 weight-only quantization

For output channel o, set `s_o = max(abs(w_o))/127`,
`q_oi = clamp(round(w_oi/s_o), -127,127)`, and reconstruct
`w_hat_oi = s_o*q_oi`. The zero row uses `s_o=1`, `q_oi=0`.
Rounding is nearest-even. For finite inputs, without clipping or scale
underflow, `abs(w_hat_oi-w_oi) <= s_o/2` up to FP32 rounding. Consequently,

```text
abs(y_hat_mo-y_mo) <= (s_o/2) * sum_i abs(x_mi)
```

The quantizer rejects weights that overflow FP32 conversion and nonzero
channels whose scale underflows to zero, instead of silently dividing by zero.

The bound separates storage quantization error from kernel accumulation error.
Tests compare kernels with the dequantized reference to isolate implementation
error, and separately test the quantizer's bound against original weights.
INT8 storage plus one FP32 scale per channel costs `O*I + 4*O` bytes, compared
with `2*O*I` for FP16 storage. Metadata and alignment can change practical memory.
These weight-only kernels multiply floating activations by reconstructed weights;
they are not integer Tensor Core GEMMs. Smaller storage does not prove faster
inference, especially for batches where unpacking and conversion dominate.

## Test tolerances and coverage

CPU reference tests use double precision for selected oracles and explicit
quantization error bounds. GPU FP32 elementwise reductions use `rtol=2e-5,
atol=2e-6`; dot-product paths use `rtol=3e-4, atol=3e-4`. FP16 uses `rtol=3e-3,
atol=3e-3`, BF16 uses `rtol=2e-2, atol=2e-2`. The output rounding and changed
reduction order motivate these tolerances; they are not a guarantee for arbitrary
input ranges or reduction lengths. Near-zero outputs require an absolute bound.
The suite covers odd dimensions, tail masks, zero rows, large finite softmax
differences, single-element rows, both causal modes, and multiple dtypes.
Do not widen tolerances when a test fails without identifying the numerical
cause. Record max absolute error and relative error with a declared denominator
floor separately from an allclose pass/fail decision.

## Sources and attribution

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Triton language documentation](https://triton-lang.org/main/python-api/triton.language.html)
- [FlashAttention, Dao et al. (2022)](https://arxiv.org/abs/2205.14135)
- [Roofline, Williams et al. (2009)](https://doi.org/10.1145/1498765.1498785)
- [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

The code here is an independently written experimental implementation of these
published mathematical principles. No performance result is implied by citation.
