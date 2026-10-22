# Initial L4 experiment

Date: 2026-10-22. GPU: NVIDIA L4, SM89, 58 SMs, 23,689,363,456 reported device
bytes. Python 3.12.1, PyTorch 2.7.1+cu126, Triton 3.3.1, toolkit 12.6.85, driver
610.57.04. FP32 TF32 disabled. RNG seed 20261022; 5 warmups, 30 dispatch samples,
10 CUDA Graph replays of 10 captured operators per case.

The expanded correctness job passed **176 tests** (24 CPU/adapter and 152 GPU
cases). The benchmark validated **171 comparisons**, all passing. This is an
initial smoke matrix on one L4 allocation, not a statistically replicated study
across devices, clock states or time periods.

Sources: [measured JSON](../results/benchmark-7a62b743566a490482ef485321ec8b67.json),
[CSV](../results/benchmark-7a62b743566a490482ef485321ec8b67.csv),
[expanded test evidence](../results/test_kernels-83e5ebd77f4c4d238aa623ad7a236009.json),
and [every measured comparison](all-measurements.md).
Benchmark source revision: `2104c05cc1589604edb77ea925403a4e3dc5ffaf` (clean at dispatch).
Remote source SHA256: `40ef347f78af0f561b36bca7c6b68123b1e6757286c1ddbe935a9984fb6977eb`.

## Representative FP16 results

All times below are medians in microseconds. GEMM and linear shapes use M,K,N;
attention uses B,H,N,D. Graph times exclude Python scheduling and output allocation.
These are deliberately representative cases, with both wins and losses; the full
table includes every shape, dtype, backend and causal setting.

| Operation / shape | Backend | Dispatch µs | Graph µs | Graph speedup vs PyTorch |
|---|---|---:|---:|---:|
| Softmax 128×1024 | PyTorch | 21.504 | 4.813 | 1.000× |
| Softmax 128×1024 | Triton | 67.584 | 2.304 | 2.089× |
| Softmax 128×1024 | CUDA | 20.480 | 3.686 | 1.306× |
| RMSNorm 128×1024 | Native PyTorch RMSNorm | 103.936 | 15.565 | 1.000× |
| RMSNorm 128×1024 | Triton | 81.920 | 2.099 | 7.415× |
| RMSNorm 128×1024 | CUDA | 21.504 | 3.277 | 4.750× |
| GEMM 128×768×3072 | PyTorch/cuBLAS | 30.720 | 11.162 | 1.000× |
| GEMM 128×768×3072 | Triton | 81.920 | 21.709 | 0.514× |
| GEMM 128×768×3072 | CUDA tiled | 272.864 | 259.072 | 0.043× |
| GEMM 128×768×3072 | CUDA naive | 369.120 | 352.307 | 0.032× |
| Causal attention 1×4×256×64 | PyTorch SDPA | 43.008 | 9.216 | 1.000× |
| Causal attention 1×4×256×64 | Triton | 94.240 | 11.366 | 0.811× |
| Causal attention 1×4×256×64 | CUDA | 133.104 | 121.648 | 0.076× |
| INT8 linear 32×512×768 | PyTorch dequantize + FP32 matmul | 116.736 | 19.149 | 1.000× |
| INT8 linear 32×512×768 | Triton | 91.136 | 24.986 | 0.766× |
| INT8 linear 32×512×768 | CUDA | 27.648 | 18.432 | 1.039× |
| Original floating linear 32×512×768 | PyTorch FP16 linear | 34.272 | 4.198 | Different weights / accuracy target |

### Interpretation

Softmax demonstrates why dispatch and device measurements must be separated:
Triton wins the graph device comparison but loses the Python dispatch comparison.
Validation/launch work and scheduling gaps materially affect short intervals.
There is no universal softmax speedup claim.

RMSNorm fusion removes a sequence of native operator work in this PyTorch version.
Its large graph improvement does not predict a matching model speedup. CUDA's
three-pass softmax trades extra reads for bounded row storage, while Triton keeps
the padded row resident. Register-pressure effects have not been measured.

The CUDA tiled GEMM improves on its naive implementation (352.307/259.072 ≈
1.36× in graph replay) but remains far behind cuBLAS. Shared-memory reuse alone
does not provide the scheduling, register blocking or Tensor Core arithmetic of
the optimized reference. Its graph throughput is about 2.33 useful TFLOP/s,
versus 27.82 for Triton and 54.11 for cuBLAS on this FP16 case. See the
[roofline model](../../docs/performance/roofline.md) for the relevant precision
ceilings and the distinction between minimum global traffic and CTA load demand.

Online attention avoids the quadratic score allocation, but the custom kernels
do not beat SDPA here. CUDA repeats K/V loads per query and serializes the online
state update in one thread. Triton's FP32 probability/value product prioritizes
accuracy over a fully low precision Tensor Core path. These are design-based
explanations consistent with the timings, not hardware-counter proof of a unique
bottleneck. Materialized attention is retained as an additional math reference;
it is not used as the only performance baseline.

INT8 weights plus scales occupy 396,288 bytes versus 786,432 FP16 bytes in this
case, a 1.98× storage reduction. The original FP16 cuBLAS linear remains much
faster than either custom weight-only kernel in graph replay. CUDA is only about
1.04× faster than dequantize-plus-matmul, while Triton is slower. Storage savings
do not establish inference acceleration. Maximum output error versus original
floating weights is recorded separately from kernel error versus dequantized weights.

## Real pretrained inference

Model: Apache-2.0 `HuggingFaceTB/SmolLM2-135M`, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Replaced 61 RMSNorm modules with the
Triton Llama-compatible mode. Prompt: “Explain why the sky appears blue in one
sentence:”. Ten input tokens and 16 greedy generated tokens; same weights, SDPA,
FP16, KV cache and generation settings in both paths. Two warmups and three
measured generations per path, baseline first.

| Metric | PyTorch | Triton RMSNorm |
|---|---:|---:|
| Median generation ms | 730.797 | 666.775 |
| p95 generation ms | 742.010 | 740.422 |
| Tokens/s | 21.894 | 23.996 |
| Peak allocated bytes | 416,070,144 | 416,070,144 |
| Incremental peak allocated bytes | 2,115,072 | 2,115,072 |

Observed median ratio: **1.096×**. There are only three samples, fixed measurement
order and overlapping tails; this is preliminary evidence, not a statistically
established general speedup. No model memory reduction was observed.

Prefill logit maximum absolute error: 0.046875; RMS error: 0.00940010. The original
`rtol=0.01, atol=0.05` assertion passed. Prefill top-1 and generated-token agreement
were both 100%. Both paths generated: “The sky is blue because it is made up of
tiny particles of air”. This demonstrates successful real-input integration;
it does not establish general language-model quality or long-generation equivalence.

The first integration attempt failed because it used a different intermediate
rounding order. The tolerance was not widened. That failure is preserved in
[the observed failure record](failed-inference.json) and the fix is explained in
[inference numerics](../../docs/mathematics/inference.md). Its microbenchmark
timings were lost when the original uncaught assertion aborted result export;
they have not been reconstructed or included in measured tables.

## Profiling observations

PyTorch profiler recorded five `_softmax` launches with 10.978 µs total device
time, about 2.196 µs per launch under instrumentation. It also recorded host
allocation and launch activity. Profiler timing is separate from event timing.

Nsight Compute was installed but its probe failed with return code 9 while
attempting to lock GPU clock frequencies. The exact driver error is retained
in measured JSON. Counter permission/availability is therefore **unverified**;
no occupancy, warp-efficiency or DRAM-counter measurements are claimed. A future
approved probe can try `--clock-control none`. Nsight Systems was not installed.

## Cost and coverage limits

Six reservations (including the failed hardware import and failed inference)
total **$1.316502**, within the approved $1.35 ceiling and $30 project target.
[Cost ledger report](costs.json). These are conservative timeout-based estimates,
not provider invoices. The successful final benchmark function body took 294.018 s;
boot/import/teardown and CPU image builds have different billing scopes. Actual
provider allocation charges still require reconciliation with Modal.

Implemented and exercised: all five PyTorch/Triton/CUDA operator families, both
CUDA GEMM variants, causal/noncausal attention, FP32/FP16/BF16, local JSON/CSV
export, graph/dispatch timings, pretrained Triton RMSNorm inference, and local
measured dashboard. Windows and Linux CPU CI passed.

Experimental or unverified: repeated independent performance trials, A10/H100,
large transformer configuration, CUDA-backed full-model inference, exhaustive
head dimensions/empty shapes/multi-device testing, memory/race sanitizers, hardware
counters, explicit CUDA Tensor Core GEMM, register blocking/vectorized CUDA loads,
INT4, and optimized integer Tensor Core quantized baselines. No claim of production
FlashAttention parity or universal custom-kernel acceleration is made.
