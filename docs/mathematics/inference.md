# Pretrained transformer experiment

`helix.integration.smollm.run_inference` loads
[HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
at immutable revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
The model card declares Apache-2.0. Its pretrained weights are downloaded only
when this experiment is explicitly invoked. Model files are not redistributed.
`trust_remote_code=False` is used for both model and tokenizer.

The experiment replaces each LlamaRMSNorm module with a forward-only Helix
implementation, preserving its exact learned weight parameter and epsilon.
Attention, rotary embeddings, linears and KV cache retain the framework path.
The same loaded model runs baseline first and then the replacement; timing order
can bias results and should be alternated in a larger study. Generation uses a
real English prompt, greedy decoding, a fixed number of new tokens and caching.

Two generation warmups exclude model initialization and kernel compilation.
Repeated synchronized host wall-clock timings include host scheduling, launch
overhead, prefill and autoregressive decoding. Tokens per second divides the
fixed generated-token count by median generation time. This is an end-to-end
generation measurement, not kernel-only timing or isolated decode throughput.
Reported peak allocated memory includes weights and live tensors. Cached allocator
reservation and device-wide usage are different quantities.

Before the custom generation, the experiment compares prefill logits using max
absolute error, RMS error and top-1 agreement. It records generation token
agreement and decoded output. Floating-point changes can alter greedy choices
when top logits are close; token equality is not a general numerical correctness
proof. Check kernel numerical tests independently. Any result with large logits
error needs investigation before use as performance evidence.
The predeclared prefill-logit assertion uses `rtol=0.01, atol=0.05`, allowing
FP16 rounding to propagate through the decoder while rejecting large adapter
errors. This is an integration acceptance bound, not a kernel accuracy claim.
Actual errors are always reported; generated-token agreement excludes the prompt.

The default experiment has not been executed merely because its local adapter
test passes. The CPU adapter test substitutes the reference operation and checks
weight/epsilon preservation without downloading a model. GPU approval and the
real inference run are required to demonstrate that integration works on hardware.

Expected speedups are bounded by the fraction of execution spent in RMSNorm.
If the baseline fraction is f and RMSNorm speeds up by s, Amdahl's bound is
`1/((1-f)+f/s)`, excluding new overhead. No microbenchmark speedup is automatically
an end-to-end gain. Kernel launch costs, KV attention and matrix multiplication
may dominate this small model's decoding path.
