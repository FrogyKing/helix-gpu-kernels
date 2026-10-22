# Research and attribution

These are independent educational engineering implementations. No production
FlashAttention compatibility or performance claim is implied.

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/): execution, synchronization, memory hierarchy and warp primitives.
- [NVIDIA Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): counter interpretation and limitations.
- [Triton language documentation](https://triton-lang.org/main/python-api/triton.language.html): reduction and dot primitives.
- [FlashAttention, Dao et al. (2022)](https://arxiv.org/abs/2205.14135): exact attention with IO-aware tiling and online normalization.
- [Online normalizer calculation, Milakov and Gimelshein (2018)](https://arxiv.org/abs/1805.02867): stable incremental softmax.
- [Roofline model, Williams et al. (2009)](https://doi.org/10.1145/1498765.1498785): arithmetic-intensity performance bounds.
- [RMSNorm, Zhang and Sennrich (2019)](https://arxiv.org/abs/1910.07467): root mean square normalization.
- [PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html): reference semantics and backend dispatch.
- [Modal CUDA guide](https://modal.com/docs/guide/cuda): runtime/toolkit distinction and container setup.
- [SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): pretrained model, architecture and Apache-2.0 license.

Third-party libraries and model weights retain their original licenses. The model
is downloaded at explicit inference invocation and is not redistributed in this repository.
