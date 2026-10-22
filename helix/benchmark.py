"""Bounded GPU-event experiments; allocation/JIT/input creation excluded from warmup."""

import csv
import importlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from helix import reference
from helix.runtime.metadata import hardware


def timings(fn, warmup, repetitions):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repetitions)
    ]
    for start, end in events:
        start.record()
        end.record()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    samples = []
    # Events enclose one framework dispatch, including its device work. This is
    # distinct from a raw CUDA kernel replay and from model end-to-end latency.
    for start, end in events:
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
        del output
    median = statistics.median(samples)
    return {
        "samples_us": samples,
        "median_us": median,
        "p95_us": float(np.percentile(samples, 95)),
        "stddev_us": statistics.pstdev(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "coefficient_of_variation": statistics.pstdev(samples) / statistics.mean(samples),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated() - before,
        "timing_scope": "cuda_events_single_dispatch_with_output_allocation",
    }


def graph_timings(fn, repetitions):
    """Device execution using an amortized captured graph of ten operations.

    Capture excludes Python and output allocator work from replay. This measures
    all kernels of an operator, not a single instruction or hardware counter.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [fn() for _ in range(10)]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    torch.cuda.synchronize()
    samples = []
    for _ in range(min(repetitions, 10)):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 10)
    del outputs, graph
    return {
        "cuda_graph_samples_us_per_operator": samples,
        "cuda_graph_median_us": statistics.median(samples),
        "cuda_graph_p95_us": float(np.percentile(samples, 95)),
        "cuda_graph_scope": "10 captured operators per replay; allocation and Python excluded",
    }


def custom(backend, operation):
    module = (
        "helix.runtime.cuda"
        if backend == "cuda"
        else ("triton_kernels." + {"quantized_linear": "quantization"}.get(operation, operation))
    )
    return getattr(importlib.import_module(module), operation)


def make_case(operation, shape, dtype, causal=False):
    rand = lambda *s: torch.randn(*s, device="cuda", dtype=dtype)  # noqa: E731
    if operation in {"softmax", "rmsnorm"}:
        x = rand(*shape)
        args = (x,) if operation == "softmax" else (x, rand(shape[-1]))
        ref = getattr(reference, operation)(*args)
        native = (
            (lambda: torch.softmax(x, -1))
            if operation == "softmax"
            else (lambda: F.rms_norm(x, (shape[-1],), args[1], eps=1e-6))
        )
        flops = 0
    elif operation == "gemm":
        m, k, n = shape
        args = (rand(m, k), rand(k, n))
        ref = (args[0].double() @ args[1].double()).to(dtype)
        native = lambda: torch.matmul(*args)  # noqa: E731
        flops = 2 * m * n * k
    elif operation == "attention":
        args = (rand(*shape), rand(*shape), rand(*shape))
        ref = reference.attention(*args, causal=causal)
        native = lambda: reference.sdpa(*args, causal=causal)  # noqa: E731
        b, h, n, d = shape
        flops = 4 * b * h * (n * (n + 1) // 2 if causal else n * n) * d
    else:
        m, k, n = shape
        x, weight = rand(m, k), rand(n, k)
        qw, scales = reference.quantize_weight(weight)
        args = (x, qw, scales)
        ref = reference.quantized_linear(*args)
        native = lambda: reference.quantized_linear(*args)  # noqa: E731
        flops = 2 * m * n * k
    kwargs = {"causal": causal} if operation == "attention" else {}
    functions = {"pytorch": native}
    if operation in {"attention", "rmsnorm"}:
        functions["pytorch_materialized"] = lambda: getattr(reference, operation)(*args, **kwargs)
    if operation == "quantized_linear":
        functions["pytorch_unquantized"] = lambda: F.linear(x, weight)
    for backend in ["triton", "cuda"]:
        fn = custom(backend, operation)
        functions[backend] = lambda fn=fn: fn(*args, **kwargs)
    if operation == "gemm":
        functions["cuda_naive"] = lambda: custom("cuda", operation)(*args, variant="naive")
    input_bytes = sum(t.numel() * t.element_size() for t in args)
    useful_bytes = input_bytes + ref.numel() * ref.element_size()
    extra = {}
    if operation == "quantized_linear":
        reconstructed = qw.float() * scales[:, None]
        extra = {
            "weight_storage_bytes": qw.numel() + scales.numel() * 4,
            "original_weight_bytes": weight.numel() * weight.element_size(),
            "weight_quant_max_abs_error": (reconstructed - weight.float()).abs().max().item(),
            "quant_output_max_abs_error": (ref - F.linear(x, weight)).abs().max().item(),
        }
    return functions, ref, flops, useful_bytes, extra


@torch.inference_mode()
def run_suite(config_path, deadline_seconds=420):
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark suite requires a CUDA GPU; skipping is not success")
    config = json.loads(Path(config_path).read_text())
    if not 1 <= config["warmup"] <= 100 or not 2 <= config["repetitions"] <= 200:
        raise ValueError("Benchmark repetitions outside bounded limits")
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    rows = []
    for operation in ["softmax", "rmsnorm", "gemm", "attention", "quantized_linear"]:
        for shape in config[operation]:
            for dtype_name in config["dtypes"]:
                dtype = getattr(torch, dtype_name)
                for causal in [False, True] if operation == "attention" else [False]:
                    if time.perf_counter() - started > deadline_seconds:
                        return {
                            "success": False,
                            "status": "deadline_reached",
                            "rows": rows,
                            "hardware": hardware(),
                            "config": config,
                            "measured": True,
                        }
                    functions, expected, flops, useful_bytes, extra = make_case(
                        operation, shape, dtype, causal
                    )
                    baseline_us = None
                    baseline_graph_us = None
                    for backend, fn in functions.items():
                        row = {
                            "operation": operation,
                            "backend": backend,
                            "shape": shape,
                            "dtype": dtype_name,
                            "causal": causal,
                            "measured": True,
                            **extra,
                        }
                        try:
                            actual = fn()
                            error = (actual.float() - expected.float()).abs()
                            row["max_abs_error"] = error.max().item()
                            row["max_relative_error_floor_1e_6"] = (
                                (error / expected.float().abs().clamp_min(1e-6)).max().item()
                            )
                            # The original floating weights are an error reference,
                            # not an exact representation of the quantized weights.
                            if backend != "pytorch_unquantized":
                                rtol, atol = {
                                    torch.float32: (3e-4, 3e-4),
                                    torch.float16: (3e-3, 3e-3),
                                    torch.bfloat16: (2e-2, 2e-2),
                                }[dtype]
                                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                            del actual, error
                            row.update(timings(fn, config["warmup"], config["repetitions"]))
                            row.update(graph_timings(fn, config["repetitions"]))
                            row["status"] = "passed"
                            if backend == "pytorch":
                                baseline_us = row["median_us"]
                                baseline_graph_us = row["cuda_graph_median_us"]
                            row["speedup_vs_pytorch"] = baseline_us / row["median_us"]
                            row["cuda_graph_speedup_vs_pytorch"] = (
                                baseline_graph_us / row["cuda_graph_median_us"]
                            )
                            row["useful_flops"] = flops
                            row["achieved_useful_tflops"] = flops / (row["median_us"] * 1e6)
                            row["minimum_io_bytes_analytical"] = useful_bytes
                            row["effective_minimum_io_gbs"] = useful_bytes / (
                                row["median_us"] * 1e3
                            )
                            # These are operator lower bounds, not measured DRAM traffic.
                            row["arithmetic_intensity_ideal_minimum_io"] = flops / useful_bytes
                        except Exception as exc:
                            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                        rows.append(row)
        print(f"Completed {operation}: {len(rows)} cumulative comparisons", flush=True)
    return {
        "success": all(r["status"] == "passed" for r in rows),
        "measured": True,
        "hardware": hardware(),
        "config": config,
        "rows": rows,
        "event_clock_resolution_note": "No event overhead subtraction; short kernels are noisy.",
        "memory_note": "Allocator peak is measured; minimum I/O is analytical, not DRAM traffic.",
    }


def export_csv(path, rows):
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else ["status"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
            )
