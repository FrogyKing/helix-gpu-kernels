# Initial implementation review

Reviewed before publication:

- Each GEMM and quantized linear tile loads ragged edges as zero; all threads
  reach both shared-memory barriers, including threads outside the output shape.
- Warp reductions use full warps (128 or 256 threads). Shared reduction scratch
  is synchronized both before reuse and after broadcast.
- Attention has a uniform key-tile loop for each query CTA. Shared K/V, scores,
  normalization state and probabilities are separated and synchronized before
  consumption/reuse. Output feature ownership is unique. Causal keys are excluded
  before normalization. No quadratic score allocation occurs.
- All CUDA launches use the current stream and a device guard. Last launch errors
  are checked. A separate GPU test validates ordering on a nondefault stream.
- FP32 reductions and explicit TF32 settings are documented. Quantization scales
  have defined zero-row behavior; nonfinite or unrepresentable scales are rejected
  by the quantizer. Kernels consume the quantizer's valid scales as a precondition.
- Benchmark failures are persisted and excluded from performance charts. Graph
  replay and dispatch timings have different labels. Synthetic dashboard data
  cannot enter the measured artifact path.
- Cost reservations are written atomically under an exclusive local lock and
  retained after failures. Concurrent clones and unrelated provider usage are
  outside the ledger's guarantees. No provider hard cap is claimed.
- Source uploads enumerate project directories and never upload credentials,
  the home directory or the local virtual environment. No remote code is trusted
  when loading the safetensors model. Git excludes model files and auth files.

Dynamic evidence: the initial L4 job ran 166 tests successfully. This does not
replace a memory/race sanitizer pass, which remains an additional validation step.
Large-grid exhaustion, multiple devices and other GPU architectures have not been
exhaustively exercised. Optimization candidates remain experimental even when
their tested numerical behavior passes.
