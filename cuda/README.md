# CUDA implementation notes

All kernels are original, forward-only implementations. PyTorch's extension loader
compiles the five CUDA translation units and a small pybind module lazily. The
toolkit and Linux compiler are supplied by the Modal development image; importing
the Python wrapper on Windows does not invoke a compiler.

| Operation | Mapping and storage | Precision and limitations |
| --- | --- | --- |
| Softmax | One CTA per row, warp shuffles plus shared warp partials; three coalesced passes over X | FP32 max/sum, FP32/FP16/BF16 input; no fixed on-chip row-width bound |
| RMSNorm | One CTA per row, shared warp partials, fused scale/store | FP32 squares and sum; same dtype weight; positive finite FP32 epsilon |
| Naive GEMM | One output per thread, no shared data reuse | FP32 FMA accumulation, FP32/FP16/BF16 storage |
| Tiled GEMM | 16x16 CTA, shared 16x16 A/B tiles, zero-fill ragged edges | SIMT FMA; **no Tensor Core use** |
| Attention | One CTA per query, 16-key shared K/V tile, online maximum/sum and register output | FP32 dot/sum/output accumulator; head dimension 1..256, matching K/V shapes; no backward/dropout/GQA |
| INT8 linear | 16x16 CTA, shared X and transposed-channel INT8 weight tiles | Floating point activations, FP32 FMA and per-output FP32 scale; **not INT8 Tensor Core GEMM** |

Tensor arguments must be contiguous on the same CUDA device. Allocation and launch
use PyTorch's current stream under a device guard. The host checks launch errors;
callers must synchronize before interpreting asynchronous execution errors. These
kernels allocate outputs, so wrapper timings include allocation bookkeeping unless
that is accounted for separately. No hidden CPU/GPU transfers occur.

Softmax follows PyTorch's NaN behavior for nonfinite rows: any NaN, any positive
infinity, or an entirely negative-infinite row yields NaNs. Finite values mixed
with negative infinity are supported. Attention is numerically validated on finite
Q/K/V; nonfinite attention inputs have no stronger guarantee. Causal attention uses
the top-left convention `key_index <= query_index`, including rectangular shapes.
Empty leading dimensions are accepted; zero-key attention is rejected explicitly.

RMSNorm uses the conventional FP32 sum-of-squares algorithm and can overflow for
extreme FP32 magnitudes; it is not an overflow-scaled norm implementation. GEMM,
attention, and linear use `fmaf` with FP32 accumulation regardless of storage
dtype. INT8 linear factors a channel's constant scale outside the K reduction;
its rounding order differs from dequantize-to-FP16-then-matmul baselines.

The tiled kernels intentionally expose synchronization and data reuse. They have
no register blocking, asynchronous copy, tensor-memory accelerator, or MMA
instructions. Wide-row softmax rereads inputs instead of keeping an unbounded
register array. Attention repeats K/V loads across query blocks. Benchmark these
limitations honestly against cuBLAS and SDPA; optimized vendor libraries are
expected to win on many workloads.

Compiler register/shared-memory diagnostics are enabled with `--ptxas-options=-v`.
Use `HELIX_CUDA_VERBOSE=1` to display compilation logs. These diagnostics are not
occupancy or throughput measurements. A CUDA-enabled correctness run and
`compute-sanitizer` where available are required before declaring GPU validation.
