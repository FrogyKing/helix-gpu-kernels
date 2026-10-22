"""Replace pretrained Llama RMSNorm modules, then measure real greedy generation.

Downloads require explicit invocation. No remote code is trusted or executed.
"""

import importlib
import math
import statistics
import time

import torch

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


class KernelRMSNorm(torch.nn.Module):
    def __init__(self, original, backend):
        super().__init__()
        self.weight = original.weight
        self.variance_epsilon = original.variance_epsilon
        module = "helix.runtime.cuda" if backend == "cuda" else "triton_kernels.rmsnorm"
        self.kernel = importlib.import_module(module).rmsnorm

    def forward(self, hidden_states):
        return self.kernel(hidden_states.contiguous(), self.weight, self.variance_epsilon)


def replace_rmsnorm(model, backend):
    """Return replaced originals so the caller can restore the exact model."""
    if backend not in {"cuda", "triton"}:
        raise ValueError("backend must be cuda or triton")
    originals = []
    for parent in model.modules():
        for name, module in list(parent.named_children()):
            if type(module).__name__ == "LlamaRMSNorm":
                originals.append((parent, name, module))
    if not originals:
        raise ValueError("model has no supported LlamaRMSNorm modules")
    for parent, name, module in originals:
        setattr(parent, name, KernelRMSNorm(module, backend))
    return originals


def run_inference(
    backend="triton",
    prompt="Explain why the sky appears blue in one sentence:",
    new_tokens=16,
    repeats=5,
    revision=REVISION,
):
    """Return measured end-to-end results; includes host scheduling and KV cache.

    Model download/load and JIT warmup are excluded. Baseline and replacement use
    one model, the same weights, prompt, precision and SDPA attention backend.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("inference experiment requires a CUDA device")
    if not 1 <= new_tokens <= 128 or not 2 <= repeats <= 20:
        raise ValueError("new_tokens must be 1..128 and repeats 2..20")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision, trust_remote_code=False)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=revision,
            trust_remote_code=False,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )
        .eval()
        .cuda()
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    if inputs.input_ids.shape[-1] + new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt and generation exceed model context")
    generate_kwargs = dict(
        **inputs,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    def measure():
        for _ in range(2):
            warm = model.generate(**generate_kwargs)
            del warm
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        samples = []
        output = None
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model.generate(**generate_kwargs)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        median = statistics.median(samples)
        result = {
            "samples_ms": samples,
            "median_ms": median,
            "p95_ms": sorted(samples)[math.ceil(0.95 * repeats) - 1],
            "tokens_per_second": new_tokens / (median / 1000),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated() - before,
            "output_token_ids": output[0].tolist(),
            "text": tokenizer.decode(output[0], skip_special_tokens=True),
        }
        del output
        return result

    with torch.inference_mode():
        baseline_logits = model(**inputs).logits.float().cpu()
        baseline = measure()
        originals = replace_rmsnorm(model, backend)
        try:
            candidate_logits = model(**inputs).logits.float().cpu()
            error = (candidate_logits - baseline_logits).abs()
            if not torch.isfinite(candidate_logits).all():
                raise RuntimeError("custom RMSNorm produced nonfinite model logits")
            # FP16 output rounding propagates through all decoder layers. This
            # predeclared bound catches large integration errors; report actual
            # errors and token agreement even when it passes.
            torch.testing.assert_close(candidate_logits, baseline_logits, rtol=1e-2, atol=5e-2)
            candidate = measure()
        finally:
            for parent, name, original in originals:
                setattr(parent, name, original)
    return {
        "kind": "pretrained_inference",
        "measured": True,
        "model": MODEL_ID,
        "model_revision": getattr(model.config, "_commit_hash", revision),
        "model_license": "Apache-2.0",
        "precision": "float16",
        "attention_backend": "sdpa",
        "custom_operation": "rmsnorm",
        "replaced_modules": len(originals),
        "prompt": prompt,
        "input_tokens": inputs.input_ids.shape[-1],
        "generated_tokens": new_tokens,
        "baseline": baseline,
        backend: candidate,
        "end_to_end_speedup": baseline["median_ms"] / candidate["median_ms"],
        "logits_max_abs_error": error.max().item(),
        "logits_rms_error": error.square().mean().sqrt().item(),
        "prefill_top1_agreement": (baseline_logits.argmax(-1) == candidate_logits.argmax(-1))
        .float()
        .mean()
        .item(),
        "generation_token_agreement": sum(
            a == b
            for a, b in zip(
                baseline["output_token_ids"][-new_tokens:],
                candidate["output_token_ids"][-new_tokens:],
            )
        )
        / new_tokens,
        "logits_validation_tolerance": {"rtol": 1e-2, "atol": 5e-2},
        "timing_scope": "host wall clock, synchronized generation; excludes load and warmup",
    }
