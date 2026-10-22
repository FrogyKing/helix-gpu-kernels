# Benchmark interpretation

Every comparison uses the same preallocated inputs, shape, dtype and causal flag.
TF32 is disabled for FP32 matrix products. GEMM correctness uses FP64 reference
accumulation then rounds to output dtype. Native torch.softmax, torch.matmul,
torch.nn.functional.rms_norm, and SDPA are primary framework baselines. Explicit
materialized attention and composed RMSNorm are additional references, not the
only performance baselines. Quantization compares dequantize-then-multiply and
the original floating weight linear layer; the latter has a different error target.

The suite records warmup count, repetitions, RNG seed and actual hardware/software
versions. JIT compilation, first-call setup and input construction precede timing.
Individual CUDA event intervals enclose one Python-dispatched operation, including
output allocation/device work; they are **operator dispatch** measurements, not
raw preallocated single-kernel replays. Python scheduling gaps can affect short
intervals. Median, p95, standard deviation, coefficient of variation and every
sample are persisted. There is no event-overhead subtraction or outlier deletion.
One case's warmups follow the previous case; kernel ordering is fixed, not randomized.
Clock locking is not assumed. These limitations matter for very short kernels.

GPU allocator peaks are measured after warmup. They exclude CUDA context and
driver allocations, and are not equivalent to nvidia-smi process memory. Minimum
input/output bytes are analytical lower bounds. CUDA softmax's three reads and
attention's repeated K/V tile reads exceed this lower bound. Effective minimum
I/O GB/s must not be described as measured DRAM throughput. GEMM useful work is
2MNK FLOPs; attention counts QK and PV useful arithmetic with a triangular causal
pair count. Softmax/RMSNorm transcendental work is not assigned arbitrary FLOPs.

Roofline bounds use `min(peak compute, memory bandwidth × arithmetic intensity)`.
No peak number is silently hardcoded: precision and sparse/dense modes must match
the experiment. Measured DRAM traffic and occupancy require hardware counters.
The PyTorch profiler supplies kernel/device intervals and memory activity when
CUPTI is available. Nsight Compute presence alone does not establish permission
to use performance counters. The report must preserve any permission error.

Correctness precedes each timing. A failed comparison is stored with status and
error, excluded from speedup charts, and makes the suite unsuccessful. GPU skips
are not a successful GPU validation run. There is no fabricated fallback dataset;
the dashboard's synthetic demonstration is kept in a separate directory and view.

## Cost model

Prices checked 2026-10-22: L4 $0.000222/s, A10 $0.000306/s, H100 $0.001097/s.
[Official Modal pricing](https://modal.com/pricing). Estimates add two physical
CPU cores, 8 GiB RAM, 120 seconds startup/shutdown allowance and 25% contingency.
The initial L4 plan reserves $0.658251 across three jobs; failed dispatches retain
their reservation. Image builds use CPU resources and actual provider billing may
differ. Reservations are not invoices. The $30 cumulative local threshold and $1
per-run threshold prevent scheduling from this checkout; they do not enforce a
provider spending cap or account for other apps. Account-specific limits remain
unknown if the provider CLI does not expose them. No H100 run is authorized by
approval of an L4 batch.

Record remote function-body time and client wall time separately. Neither is a
complete provider allocation measurement: boot, import, build, queue and teardown
accounting differ. Reconcile with Modal billing before claiming actual dollars spent.
