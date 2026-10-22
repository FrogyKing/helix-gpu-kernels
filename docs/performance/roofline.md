# L4 performance model

NVIDIA lists a 30.3 TFLOP/s FP32 SIMT peak, 121 TFLOP/s dense FP16 Tensor Core
peak and 300 GB/s memory bandwidth for L4. Sparse Tensor Core figures are not
applicable to these dense matrices. These are vendor ceilings, not measurements
of the Modal allocation. [NVIDIA Ada architecture white paper, Appendix D](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf).

The FP32 SIMT ridge point is `30.3e12 / 300e9 = 101 FLOP/byte`; the dense FP16
Tensor Core ridge is approximately `403 FLOP/byte`. CUDA kernels use the SIMT
ceiling even for FP16 input. A low precision storage type alone does not select
the Tensor Core roof. FP32 IEEE Triton must also not use a TF32 Tensor Core peak.

Consider the configured transformer-shaped GEMM `[128,768] @ [768,3072]`:
it has 603,979,776 useful FLOPs. FP16 input/output minimum traffic is 5,701,632
bytes, giving an optimistic 105.93 FLOP/byte intensity. This assumes each input
element is loaded once across the entire grid. At 300 GB/s its ideal DRAM roof
is about 31.78 TFLOP/s, capped at 30.3 TFLOP/s for the CUDA SIMT path. This is a
coarse classification, not a prediction of achieved performance.

The actual CUDA CTA only reuses data inside a 16×16 tile: each K tile does 8,192
FLOPs while loading 1,024 bytes of FP16 input, or 8 FLOP/byte of CTA load demand
before output stores. FP32 storage gives 4 FLOP/byte. Repeated loads across CTAs,
cache reuse, instruction/convert overhead, synchronization and limited independent
accumulators make the global minimum-traffic roof optimistic. A naive one-output
thread has still less explicit reuse. Measuring DRAM bytes is necessary to place
these kernels at their actual memory roofline position.

Small repeatedly reused inputs can reside in L2, so dividing minimum tensor bytes
by warm-cache timing can exceed external-memory bandwidth without contradiction.
Dispatch intervals can instead be dominated by CPU launch gaps. Compare the
CUDA Graph device intervals alongside dispatch timing, and avoid attributing
every loss to bandwidth or occupancy without counter evidence. The measured
experiment table reports achieved useful TFLOP/s; the model explains candidate
limits rather than proving a unique bottleneck.
