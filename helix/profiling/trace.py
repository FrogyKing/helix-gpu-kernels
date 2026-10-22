def probe_nsight_compute():
    import shutil
    import subprocess

    if not shutil.which("ncu"):
        return {"status": "unavailable", "reason": "ncu executable absent"}
    try:
        process = subprocess.run(
            [
                "ncu",
                "--target-processes",
                "all",
                "--kernel-name",
                "regex:softmax",
                "--launch-count",
                "1",
                "--metrics",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "python",
                "-c",
                "import torch; from triton_kernels.softmax import softmax; "
                "softmax(torch.empty(128,1024,device='cuda')); torch.cuda.synchronize()",
            ],
            capture_output=True,
            text=True,
            timeout=35,
        )
        output = process.stdout + process.stderr
        return {
            "status": "completed" if process.returncode == 0 else "unavailable",
            "returncode": process.returncode,
            "output": output,
            "note": "Probe only. Empty input used; not a numerical or timing result.",
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def profile_softmax():
    import torch
    from torch.profiler import ProfilerActivity, profile

    from triton_kernels.softmax import softmax

    x = torch.randn(128, 1024, device="cuda")
    for _ in range(5):
        softmax(x)
    torch.cuda.synchronize()
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            for _ in range(5):
                softmax(x)
            torch.cuda.synchronize()
        return {
            "status": "measured",
            "tool": "torch.profiler",
            "events": [
                {
                    "name": e.key,
                    "calls": e.count,
                    "self_cpu_time_us": e.self_cpu_time_total,
                    "self_device_time_us": e.self_device_time_total,
                    "device_memory_bytes": e.device_memory_usage,
                }
                for e in profiler.key_averages()
            ],
            "limitations": "Trace timing only; no occupancy, DRAM counters or warp efficiency.",
        }
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
